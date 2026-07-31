// Command gradient-inspect trains a small LTC on the same toy
// bounded-accumulator task as examples/ltc-sequence (at every step the
// network sees a sign u_t in {-1, +1} and must output the clipped running
// sum s_t = clip(s_{t-1} + 0.25*u_t, -1, 1)) and prints a gradient
// diagnostic block every K iterations:
//
//   - the gradient L2 norm of every parameter (a nil Grad means the
//     parameter is not wired into the graph at all),
//   - the global maximum absolute gradient value,
//   - the count of NaN/Inf gradient entries,
//   - the global gradient norm before and after clipping, in the same
//     iteration, so the clip's effect is visible side by side,
//   - the parameter-update magnitude: the L2 norm of the change the
//     optimizer step actually applied to each parameter.
//
// It is a template for locating the cause when a loss does not fall. How
// to read the block:
//
//   - Every gradient norm is zero, or a parameter's Grad is nil: the
//     parameter never joined the loss's graph (check the forward wiring —
//     an unused module handed to nn.ParametersOf is the classic case), or
//     the signal vanishes before it reaches the leaves (learning rate and
//     scale sanity checks apply). Note the adjacent failure mode: a
//     missing ZeroGrad doubles the gradients instead of zeroing them, so
//     norms exactly 2x too large point at the loop discipline, not the
//     model (doc/training.md).
//   - The NaN/Inf count is non-zero: float32 overflow, or a degenerate
//     input — for the LTC, a bad ts or the 1/(den+eps) division spike of
//     doc/training.md's divergence checklist. One non-finite value
//     propagates through the whole graph, so the first diagnostic block
//     that reports it pins down the iteration where things broke.
//   - Norms grow from block to block: gradient explosion. Enable (or
//     tighten) caller-owned global gradient-norm clipping before the
//     optimizer step — the raw-vs-clipped comparison below shows exactly
//     what the clip removes.
//
// Run from the repository root:
//
//	go run ./examples/gradient-inspect
//
// The program is a self-checking smoke test: fixed seed, deterministic
// output, and a non-zero exit if the loss does not decrease, any gradient
// is non-finite, or the clip never engages.
package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/nn"
	"github.com/qorm/LNN/optimizer"
	"github.com/qorm/LNN/tensor"
)

// names labels nn.ParametersOf(cell, readout) in its flattening order: the
// LTC's 13 trainable parameters, then the readout's W and B. Kept in sync
// with LTC.Parameters and Linear.Parameters by hand, on purpose — the
// diagnostic output names parameters rather than printing raw indices.
var names = []string{
	"gleak", "vleak", "cm", "mu", "sigma", "w",
	"sMu", "sSigma", "sW", "inW", "inB", "outW", "outB",
	"readout.W", "readout.B",
}

func main() {
	rng := rand.New(rand.NewSource(42))

	const (
		inDim   = 1
		units   = 8 // same scale as examples/ltc-sequence: its early gradient norms exceed 1.0, so the clip below engages
		unfolds = 4
		seqLen  = 12
		batch   = 16
		iters   = 100
		every   = 25   // diagnostic period
		lr      = 0.05 // plain SGD: the diagnostics expose what the update sees
		maxNorm = 1.0  // global gradient-norm clip
		ts      = 1.0  // ODE time span per step
		step    = 0.25 // accumulator increment per input
		clip    = 1.0  // target saturation bound
	)

	cell := nn.NewLTC(inDim, units, nil, unfolds, rng)
	readout := nn.NewLinear(units, 1, rng)
	params := nn.ParametersOf(cell, readout)
	opt := optimizer.NewSGD(lr)

	fmt.Printf("LTC gradient inspection: inDim=%d units=%d unfolds=%d seqLen=%d batch=%d\n",
		inDim, units, unfolds, seqLen, batch)
	fmt.Printf("trainable parameters: %d tensors, diagnostics every %d iterations\n\n", len(params), every)

	var first, last float64
	var totalNonFinite int
	var clipEverEngaged bool
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

		for _, p := range params {
			p.ZeroGrad()
		}
		loss.Backward()

		l := float64(loss.Value())
		if it == 0 {
			first = l
		}
		last = l

		diag := it%every == 0 || it == iters-1
		if diag {
			fmt.Printf("iter %3d  loss=%.6f\n", it, l)
			totalNonFinite += printGradStats(params)
		}

		// Clip the global gradient norm (caller-owned), snapshot the
		// parameters, step, and measure what the step changed. The
		// raw-vs-clipped norm comparison is computed on the same gradient,
		// in the same iteration.
		raw, clipped, engaged := clipGrads(params, maxNorm)
		if engaged {
			clipEverEngaged = true
		}
		before := make([][]float32, len(params))
		for i, p := range params {
			before[i] = append([]float32(nil), p.Data.Data...)
		}
		opt.Step(params)
		if diag {
			status := "not engaged"
			if engaged {
				status = "engaged"
			}
			fmt.Printf("  clip: maxNorm=%.1f  raw norm=%.6f -> clipped norm=%.6f  (%s)\n",
				maxNorm, raw, clipped, status)
			fmt.Println("  parameter update magnitudes:")
			for i, p := range params {
				var d2 float64
				for j, v := range p.Data.Data {
					d := float64(v) - float64(before[i][j])
					d2 += d * d
				}
				fmt.Printf("    param %2d %-9s |Δparam|=%12.6e\n", i, names[i], math.Sqrt(d2))
			}
			fmt.Println()
		}
	}

	fmt.Printf("first loss %.6f -> final loss %.6f\n", first, last)
	switch {
	case last >= first:
		fmt.Println("FAIL: loss did not decrease")
		os.Exit(1)
	case totalNonFinite > 0:
		fmt.Printf("FAIL: %d non-finite gradient values observed\n", totalNonFinite)
		os.Exit(1)
	case !clipEverEngaged:
		fmt.Println("FAIL: gradient clip never engaged")
		os.Exit(1)
	}
	fmt.Println("OK: loss decreased, all gradients finite, clip engaged")
}

// printGradStats prints one gradient L2 norm per parameter plus the global
// maximum absolute gradient and the NaN/Inf count, and returns the count of
// non-finite entries seen. A nil Grad — the parameter is not wired into the
// graph — prints as norm 0 with a note, since that is exactly the signal a
// stalled loss sends.
func printGradStats(params []*autograd.Variable) int {
	fmt.Println("  gradient L2 norms per parameter:")
	var norm2, maxAbs float64
	var nonFinite int
	for i, p := range params {
		var pn2 float64
		wired := p.Grad != nil
		if wired {
			for _, g := range p.Grad.Data {
				fg := float64(g)
				pn2 += fg * fg
				if a := math.Abs(fg); a > maxAbs {
					maxAbs = a
				}
				if math.IsNaN(fg) || math.IsInf(fg, 0) {
					nonFinite++
				}
			}
		}
		norm2 += pn2
		if wired {
			fmt.Printf("    param %2d %-9s grad L2=%12.6e\n", i, names[i], math.Sqrt(pn2))
		} else {
			fmt.Printf("    param %2d %-9s grad L2=%12.6e  (nil grad: not in graph)\n", i, names[i], 0.0)
		}
	}
	fmt.Printf("  global: max|grad|=%.6e  NaN/Inf count=%d  (all-parameter grad L2=%.6f)\n",
		maxAbs, nonFinite, math.Sqrt(norm2))
	return nonFinite
}

// clipGrads rescales every gradient in place so the global gradient norm
// does not exceed maxNorm, and returns the norm before clipping, the norm
// after, and whether the clip engaged. The post-clip norm is recomputed
// from the rescaled gradients rather than assumed, so the comparison the
// caller prints is measured, not derived.
func clipGrads(params []*autograd.Variable, maxNorm float64) (raw, clipped float64, engaged bool) {
	var norm2 float64
	for _, p := range params {
		if p.Grad == nil {
			continue
		}
		for _, g := range p.Grad.Data {
			norm2 += float64(g) * float64(g)
		}
	}
	raw = math.Sqrt(norm2)
	clipped = raw
	if raw > maxNorm {
		s := float32(maxNorm / raw)
		for _, p := range params {
			if p.Grad == nil {
				continue
			}
			for i := range p.Grad.Data {
				p.Grad.Data[i] *= s
			}
		}
		var c2 float64
		for _, p := range params {
			if p.Grad == nil {
				continue
			}
			for _, g := range p.Grad.Data {
				c2 += float64(g) * float64(g)
			}
		}
		clipped = math.Sqrt(c2)
		engaged = true
	}
	return raw, clipped, engaged
}
