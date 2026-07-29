package tensor

import (
	"math/rand"
	"testing"
)

// benchSink keeps the compiler from optimizing benchmark results away.
var benchSink *Tensor

// Benchmarks for the representative hot paths of the tensor package. Sizes
// reflect the library's 1D/2D focus (matrix ops up to 128x128). Every
// benchmark constructs its inputs once before the timed loop so only the
// operation under test is measured.

func BenchmarkMatMul64(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	a := Randn(rng, 64, 64)
	c := Randn(rng, 64, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSink = MatMul(a, c)
	}
}

func BenchmarkMatMul128(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	a := Randn(rng, 128, 128)
	c := Randn(rng, 128, 128)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSink = MatMul(a, c)
	}
}

func BenchmarkSoftmaxRows(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	a := Randn(rng, 64, 128)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSink = SoftmaxRows(a)
	}
}

// BenchmarkAddBroadcastRow exercises limited broadcasting: a full matrix
// plus a 1D row vector replicated across every row.
func BenchmarkAddBroadcastRow(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	a := Randn(rng, 128, 128)
	row := Randn(rng, 128)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSink = Add(a, row)
	}
}

func BenchmarkHadamard(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	a := Randn(rng, 128, 128)
	c := Randn(rng, 128, 128)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSink = Hadamard(a, c)
	}
}

func BenchmarkSumRows(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	a := Randn(rng, 128, 128)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSink = SumRows(a)
	}
}

func BenchmarkSumCols(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	a := Randn(rng, 128, 128)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSink = SumCols(a)
	}
}

func BenchmarkTranspose(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	a := Randn(rng, 128, 128)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSink = Transpose(a)
	}
}
