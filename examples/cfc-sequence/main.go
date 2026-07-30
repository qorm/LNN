// Command cfc-sequence trains a small CfC (Closed-form Continuous-time cell)
// on the same toy bounded-accumulator task as examples/ltc-sequence: at every
// step the network sees a sign u_t in {-1, +1} and must output the clipped
// running sum
//
//	s_t = clip(s_{t-1} + 0.25*u_t, -1, 1).
//
// The task needs cross-step memory, so it genuinely exercises the cell's
// dynamics rather than a static input/output map. The CfC advances the full
// time span of each step with the Lemma-1 closed-form solution (see
// doc/cfc.md), so there is no ODE-substep unfolds parameter and the graph
// per RNN step is lighter than the LTC's at the same scale.
//
// The loop demonstrates the recommended production form from doc/training.md:
// caller-owned global gradient-norm clipping (maxNorm=1.0, same recipe as
// examples/ltc-sequence) followed by the optimizer package — optimizer.NewSGD
// plus one Step(params) call in place of the hand-rolled update loop. Step
// never calls ZeroGrad, so the loop zeroes gradients explicitly every
// iteration. The program doubles as an end-to-end integration smoke test:
// fixed seed, deterministic output, loss printed as it falls.
package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/nn"
	"github.com/qorm/LNN/optimizer"
	"github.com/qorm/LNN/tensor"
)

func main() {
	rng := rand.New(rand.NewSource(42))

	const (
		inDim   = 1
		units   = 8
		seqLen  = 12
		batch   = 16
		iters   = 250
		lr      = 0.05
		maxNorm = 1.0  // global gradient-norm clip
		ts      = 1.0  // time span per step
		step    = 0.25 // accumulator increment per input
		clip    = 1.0  // target saturation bound
	)

	// No unfolds parameter: the closed-form solution advances the full time
	// span in a single step.
	cell := nn.NewCfC(inDim, units, nil, rng)
	readout := nn.NewLinear(units, 1, rng)
	params := nn.ParametersOf(cell, readout)
	opt := optimizer.NewSGD(lr)

	fmt.Printf("CfC accumulator task: inDim=%d units=%d seqLen=%d batch=%d\n",
		inDim, units, seqLen, batch)
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

		// ZeroGrad is the caller's contract: Step never does it.
		for _, p := range params {
			p.ZeroGrad()
		}
		loss.Backward()

		// Global gradient-norm clipping (caller-owned, as in
		// examples/ltc-sequence and doc/training.md): accumulate the norm in
		// float64, scale gradients in place, then let the optimizer step.
		var norm2 float64
		for _, p := range params {
			if p.Grad == nil {
				continue
			}
			for _, g := range p.Grad.Data {
				norm2 += float64(g) * float64(g)
			}
		}
		if norm := math.Sqrt(norm2); norm > maxNorm {
			s := float32(maxNorm / norm)
			for _, p := range params {
				if p.Grad == nil {
					continue
				}
				for i := range p.Grad.Data {
					p.Grad.Data[i] *= s
				}
			}
		}
		opt.Step(params)

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
