package nn

import (
	"math/rand"
	"runtime"
	"testing"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/tensor"
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

// BenchmarkCfCStep times a single CfC RNN step: the input affine map, the
// sensory and recurrent synaptic drives (the sparse fold contraction), the
// closed-form membrane update (decay rate + exprel decay factor), and the
// output affine map. in=4, units=16, batch=8 — the BenchmarkLTCStep dims.
func BenchmarkCfCStep(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	cell := NewCfC(4, 16, nil, rng)
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

// rematBenchSetup builds the shared fixtures for the remat benchmarks:
// the same 12-step LTC/CfC sequence workload BenchmarkUnrollBackward uses.
func rematBenchSetup(cfc bool) (cell Cell, params []*autograd.Variable, readout *Linear, xs, targets []*autograd.Variable) {
	rng := rand.New(rand.NewSource(42))
	const (
		inDim  = 1
		units  = 8
		seqLen = 12
		batch  = 16
	)
	if cfc {
		c := NewCfC(inDim, units, nil, rng)
		cell = c
	} else {
		c := NewLTC(inDim, units, nil, 4, rng)
		cell = c
	}
	readout = NewLinear(units, 1, rng)
	params = ParametersOf(cell.(Module), readout)
	xs = make([]*autograd.Variable, seqLen)
	targets = make([]*autograd.Variable, seqLen)
	for t := 0; t < seqLen; t++ {
		xb := make([]float32, batch)
		yb := make([]float32, batch)
		for i := range xb {
			if rng.Intn(2) == 0 {
				xb[i] = -1
			} else {
				xb[i] = 1
			}
			yb[i] = float32(rng.Intn(5)-2) * 0.25
		}
		xs[t] = autograd.Var(tensor.FromData(xb, batch, inDim))
		targets[t] = autograd.Var(tensor.FromData(yb, batch, 1))
	}
	return cell, params, readout, xs, targets
}

// rematBenchLoss is BenchmarkUnrollBackward's loss over a ys slice.
func rematBenchLoss(readout *Linear, targets []*autograd.Variable) func(ys []*autograd.Variable) *autograd.Variable {
	return func(ys []*autograd.Variable) *autograd.Variable {
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
		return autograd.Scale(autograd.MeanAll(acc), 1/float32(len(ys)))
	}
}

// BenchmarkUnrollRemat times the rematerialized counterpart of
// BenchmarkUnrollBackward on the identical workload (chunkSize 4): two
// forwards and one backward per unit sweep plus the σ sweep the LTC's
// spine class requires — the recompute premium paid for O(chunk) peak
// graph memory instead of O(T).
func BenchmarkUnrollRemat(b *testing.B) {
	cell, params, readout, xs, targets := rematBenchSetup(false)
	lossFn := rematBenchLoss(readout, targets)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, p := range params {
			p.ZeroGrad()
		}
		_, _, loss := UnrollRemat(cell, params, xs, nil, 1.0, 4, lossFn)
		benchSink = loss
	}
}

// BenchmarkUnrollRematCfC is BenchmarkUnrollRemat for the CfC cell, whose
// spine-free step graph takes the rest sweep alone (two forwards, one
// backward).
func BenchmarkUnrollRematCfC(b *testing.B) {
	cell, params, readout, xs, targets := rematBenchSetup(true)
	lossFn := rematBenchLoss(readout, targets)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, p := range params {
			p.ZeroGrad()
		}
		_, _, loss := UnrollRemat(cell, params, xs, nil, 1.0, 4, lossFn)
		benchSink = loss
	}
}

// liveHeapMB returns the live heap in MiB after a full GC.
func liveHeapMB() float64 {
	runtime.GC()
	runtime.GC()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return float64(m.HeapAlloc) / (1 << 20)
}

// BenchmarkUnrollPeakMemory512 measures the live heap the full-unroll
// graph pins between forward and Backward at T=512 — the O(T × step
// graph) memory remat eliminates. Reported as the custom liveMB metric;
// no assertion is made (GC timing is machine-dependent).
func BenchmarkUnrollPeakMemory512(b *testing.B) {
	rng := rand.New(rand.NewSource(7))
	const (
		inDim  = 1
		units  = 8
		seqLen = 512
		batch  = 16
	)
	cell := NewLTC(inDim, units, nil, 4, rng)
	readout := NewLinear(units, 1, rng)
	xs := make([]*autograd.Variable, seqLen)
	targets := make([]*autograd.Variable, seqLen)
	for t := 0; t < seqLen; t++ {
		xs[t] = autograd.Var(tensor.Uniform(rng, -1, 1, batch, inDim))
		targets[t] = autograd.Var(tensor.Uniform(rng, -0.5, 0.5, batch, 1))
	}
	lossFn := rematBenchLoss(readout, targets)
	for i := 0; i < b.N; i++ {
		base := liveHeapMB()
		ys, _ := Unroll(cell, xs, nil, 1.0)
		loss := lossFn(ys)
		peak := liveHeapMB()
		b.ReportMetric(peak-base, "liveMB")
		loss.Backward() // release before the next iteration
		benchSink = loss
	}
}

// BenchmarkUnrollRematPeakMemory512 measures UnrollRemat's peak live heap
// on the same T=512 workload as the sum it is bounded by: the retained
// small tensors (detached outputs/states, seeds, accumulated gradients,
// the loss graph) measured after the call, plus one recompute unit's
// transient graph (chunkSize 16), measured by holding one unit's outputs
// without detaching. Reported as retainedMB and unitMB; no assertion.
func BenchmarkUnrollRematPeakMemory512(b *testing.B) {
	rng := rand.New(rand.NewSource(7))
	const (
		inDim  = 1
		units  = 8
		seqLen = 512
		batch  = 16
		chunk  = 16
	)
	cell := NewLTC(inDim, units, nil, 4, rng)
	readout := NewLinear(units, 1, rng)
	params := ParametersOf(cell, readout)
	xs := make([]*autograd.Variable, seqLen)
	targets := make([]*autograd.Variable, seqLen)
	for t := 0; t < seqLen; t++ {
		xs[t] = autograd.Var(tensor.Uniform(rng, -0.5, 0.5, batch, inDim))
		targets[t] = autograd.Var(tensor.Uniform(rng, -0.5, 0.5, batch, 1))
	}
	lossFn := rematBenchLoss(readout, targets)
	for i := 0; i < b.N; i++ {
		for _, p := range params {
			p.ZeroGrad()
		}
		base := liveHeapMB()
		ys, hN, loss := UnrollRemat(cell, params, xs, nil, 1.0, chunk, lossFn)
		retained := liveHeapMB()
		b.ReportMetric(retained-base, "retainedMB")
		// One recompute unit's transient graph, upper bound of the sweep
		// peak on top of the retained set.
		ubase := liveHeapMB()
		h := statesForMemoryProbe(cell, xs, chunk)
		ysU := make([]*autograd.Variable, chunk)
		for k := 0; k < chunk; k++ {
			ysU[k], h = cell.Step(xs[k], h, 1.0)
		}
		benchSink = ysU[chunk-1] // keep the unit graph live through the measurement
		unit := liveHeapMB()
		b.ReportMetric(unit-ubase, "unitMB")
		benchSink = ys[len(ys)-1]
		_, _ = hN, loss
	}
}

// statesForMemoryProbe builds a detached mid-sequence state to start a
// one-unit recompute from (mirroring the sweep's boundary leaves).
func statesForMemoryProbe(cell Cell, xs []*autograd.Variable, k int) *autograd.Variable {
	var h *autograd.Variable
	for i := 0; i < k; i++ {
		_, h = cell.Step(xs[i], h, 1.0)
		h = autograd.Detach(h)
	}
	return h
}
