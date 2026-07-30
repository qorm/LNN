// Command ltc-sequence trains a small LTC on a toy bounded-accumulator task:
// at every step the network sees a sign u_t in {-1, +1} and must output the
// clipped running sum
//
//	s_t = clip(s_{t-1} + 0.25*u_t, -1, 1).
//
// The task needs cross-step memory, so it genuinely exercises the liquid
// dynamics rather than a static input/output map. Training is hand-rolled
// SGD (kept hand-rolled deliberately as the readable baseline; the
// lnn/optimizer package offers SGD/Momentum/Adam for production loops)
// with global gradient-norm clipping, and the program doubles as an
// end-to-end integration smoke test: fixed seed, deterministic output,
// loss printed as it falls.
package main

import (
	"fmt"
	"math"
	"math/rand"

	"lnn/autograd"
	"lnn/nn"
	"lnn/tensor"
)

func main() {
	rng := rand.New(rand.NewSource(42))

	const (
		inDim   = 1
		units   = 8
		unfolds = 4
		seqLen  = 12
		batch   = 16
		iters   = 250
		lr      = 0.05
		maxNorm = 1.0  // global gradient-norm clip
		ts      = 1.0  // ODE time span per step
		step    = 0.25 // accumulator increment per input
		clip    = 1.0  // target saturation bound
	)

	cell := nn.NewLTC(inDim, units, nil, unfolds, rng)
	readout := nn.NewLinear(units, 1, rng)
	params := nn.ParametersOf(cell, readout)

	fmt.Printf("LTC accumulator task: inDim=%d units=%d unfolds=%d seqLen=%d batch=%d\n",
		inDim, units, unfolds, seqLen, batch)
	fmt.Printf("trainable parameters: %d tensors\n\n", len(params))

	var first, last float64
	for it := 0; it < iters; it++ {
		// Fresh random sequences every iteration (online SGD).
		xs := make([]*autograd.Variable, seqLen)
		targets := make([]*autograd.Variable, seqLen)
		state := make([]float32, batch)
		for t := 0; t < seqLen; t++ {
			xb := make([]float32, batch)
			yb := make([]float32, batch)
			for b := 0; b < batch; b++ {
				u := float32(1)
				if rng.Intn(2) == 0 {
					u = -1
				}
				xb[b] = u
				s := state[b] + step*u
				if s > clip {
					s = clip
				} else if s < -clip {
					s = -clip
				}
				state[b] = s
				yb[b] = s
			}
			xs[t] = autograd.Var(tensor.FromData(xb, batch, inDim))
			targets[t] = autograd.Var(tensor.FromData(yb, batch, 1))
		}

		// Forward through the unrolled sequence and a linear readout.
		ys, _ := nn.Unroll(cell, xs, nil, ts)
		var acc *autograd.Variable
		for t, y := range ys {
			diff := autograd.Sub(readout.Forward(y), targets[t]) // [batch, 1]
			sq := autograd.Hadamard(diff, diff)
			if t == 0 {
				acc = sq
			} else {
				acc = autograd.Add(acc, sq)
			}
		}
		loss := autograd.Scale(autograd.MeanAll(acc), 1/float32(seqLen))

		// Backward, clip the global gradient norm, step.
		for _, p := range params {
			p.ZeroGrad()
		}
		loss.Backward()

		var norm2 float64
		for _, p := range params {
			if p.Grad == nil {
				continue
			}
			for _, g := range p.Grad.Data {
				norm2 += float64(g) * float64(g)
			}
		}
		scale := lr
		if norm := math.Sqrt(norm2); norm > maxNorm {
			scale = lr * maxNorm / norm
		}
		for _, p := range params {
			if p.Grad == nil {
				continue
			}
			for i := range p.Data.Data {
				p.Data.Data[i] -= float32(scale) * p.Grad.Data[i]
			}
		}

		l := float64(loss.Value())
		if it == 0 {
			first = l
		}
		last = l
		if it%25 == 0 || it == iters-1 {
			fmt.Printf("iter %3d  loss=%.6f\n", it, l)
		}
	}

	fmt.Printf("\nfirst loss %.6f -> final loss %.6f\n", first, last)
	if last >= first {
		fmt.Println("FAIL: loss did not decrease")
		return
	}
	fmt.Println("OK: loss decreased")
}
