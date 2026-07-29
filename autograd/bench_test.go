package autograd

import (
	"math/rand"
	"testing"

	"lnn/tensor"
)

// benchSink keeps the compiler from optimizing benchmark results away.
var benchSink *Variable

// BenchmarkChainForwardBackward times an end-to-end forward+backward pass
// over a medium-sized graph: a deep chain of alternating Hadamard/Add layers
// reduced to a scalar loss, then Backward through the whole chain into the
// leaves. The graph is rebuilt each iteration, so the measurement covers
// graph construction (forward) and gradient propagation together.
func BenchmarkChainForwardBackward(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	const depth = 16
	weights := make([]*Variable, depth)
	for i := range weights {
		weights[i] = Var(tensor.Uniform(rng, 0.9, 1.1, 64, 64))
	}
	x := Var(tensor.Randn(rng, 64, 64))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x.ZeroGrad()
		for _, w := range weights {
			w.ZeroGrad()
		}
		v := x
		for _, w := range weights {
			v = Add(Hadamard(v, w), x)
		}
		loss := MeanAll(v)
		loss.Backward()
	}
}

// BenchmarkGatherRowsBackward times GatherRows plus its backward pass: a
// per-row gather out of a wide table, summed to a scalar, then Backward
// scattering the gradient back into the [rows, cols] leaf.
func BenchmarkGatherRowsBackward(b *testing.B) {
	rng := rand.New(rand.NewSource(2))
	const rows, cols = 1024, 32
	table := Var(tensor.Randn(rng, rows, cols))
	idx := make([]int, rows)
	for i := range idx {
		idx[i] = rng.Intn(cols)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		table.ZeroGrad()
		picked := GatherRows(table, idx)
		loss := SumAll(picked)
		loss.Backward()
	}
}

// BenchmarkDivDenLoop mimics the LTC update loop, whose hot path repeatedly
// composes divisions of the form v = num / (den + eps): one seed division
// followed by unfolds rounds of v = (v + a_t) / (b_t + eps). It serves as
// the regression baseline for any future closed-form Div implementation
// (Div currently lowers to Hadamard(a, Pow(b, -1))).
func BenchmarkDivDenLoop(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	const unfolds = 8
	as := make([]*Variable, unfolds+1)
	bs := make([]*Variable, unfolds+1)
	for i := 0; i <= unfolds; i++ {
		as[i] = Var(tensor.Uniform(rng, 0.1, 1, 32, 32))
		bs[i] = Var(tensor.Uniform(rng, 0.1, 1, 32, 32))
	}
	eps := Const(tensor.FromData([]float32{1e-8}, 1))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, a := range as {
			a.ZeroGrad()
		}
		for _, bv := range bs {
			bv.ZeroGrad()
		}
		eps.ZeroGrad()
		v := Div(as[0], Add(bs[0], eps))
		for t := 1; t <= unfolds; t++ {
			v = Div(Add(v, as[t]), Add(bs[t], eps))
		}
		loss := MeanAll(v)
		loss.Backward()
	}
}
