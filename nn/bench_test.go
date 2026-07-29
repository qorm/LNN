package nn

import (
	"math/rand"
	"testing"

	"lnn/autograd"
	"lnn/tensor"
)

// benchSink keeps the compiler from optimizing benchmark results away.
var benchSink *autograd.Variable

// BenchmarkLTCStep times a single LTC RNN step: the input affine map, the
// sensory synapses, unfolds ODE substeps (each with the full recurrent
// synapse fan-in and the num/(den+eps) membrane update), and the output
// affine map. in=4, units=16, unfolds=4, batch=8.
func BenchmarkLTCStep(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	cell := NewLTC(4, 16, nil, 4, rng)
	x := autograd.Var(tensor.Uniform(rng, -1, 1, 8, 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, _ := cell.Step(x, nil, 0.1)
		benchSink = out
	}
}

// BenchmarkUnrollBackward replicates the examples/ltc-sequence workload:
// a 12-step sequence unrolled through an LTC (inDim=1, units=8, unfolds=4,
// batch=16) feeding a linear readout, an MSE-style loss over all steps, and
// one Backward through the entire unrolled graph. Inputs and targets are
// drawn from a fixed seed once, outside the timed loop.
func BenchmarkUnrollBackward(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	const (
		inDim  = 1
		units  = 8
		seqLen = 12
		batch  = 16
	)
	cell := NewLTC(inDim, units, nil, 4, rng)
	readout := NewLinear(units, 1, rng)
	params := ParametersOf(cell, readout)

	xs := make([]*autograd.Variable, seqLen)
	targets := make([]*autograd.Variable, seqLen)
	for t := 0; t < seqLen; t++ {
		xb := make([]float32, batch)
		yb := make([]float32, batch)
		for i := range xb {
			if rng.Intn(2) == 0 {
				xb[i] = -1
			} else {
				xb[i] = 1
			}
			yb[i] = float32(rng.Intn(5)-2) * 0.25 // bounded target in [-0.5, 0.5]
		}
		xs[t] = autograd.Var(tensor.FromData(xb, batch, inDim))
		targets[t] = autograd.Var(tensor.FromData(yb, batch, 1))
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, p := range params {
			p.ZeroGrad()
		}
		ys, _ := Unroll(cell, xs, nil, 1.0)
		var acc *autograd.Variable
		for t, y := range ys {
			diff := autograd.Sub(readout.Forward(y), targets[t])
			sq := autograd.Hadamard(diff, diff)
			if t == 0 {
				acc = sq
			} else {
				acc = autograd.Add(acc, sq)
			}
		}
		loss := autograd.Scale(autograd.MeanAll(acc), 1/float32(seqLen))
		loss.Backward()
		benchSink = loss
	}
}
