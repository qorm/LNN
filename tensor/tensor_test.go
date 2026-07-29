package tensor

import (
	"math"
	"math/rand"
	"testing"
)

const tol = 1e-6

func almostEq(a, b, eps float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= eps
}

func checkData(t *testing.T, got *Tensor, want []float32, eps float32) {
	t.Helper()
	if len(got.Data) != len(want) {
		t.Fatalf("data length %d, want %d (shape %v)", len(got.Data), len(want), got.Shape)
	}
	for i := range want {
		if !almostEq(got.Data[i], want[i], eps) {
			t.Fatalf("data[%d] = %v, want %v (full: %v)", i, got.Data[i], want[i], got.Data)
		}
	}
}

func checkShape(t *testing.T, got *Tensor, want ...int) {
	t.Helper()
	if len(got.Shape) != len(want) {
		t.Fatalf("shape %v, want %v", got.Shape, want)
	}
	for i := range want {
		if got.Shape[i] != want[i] {
			t.Fatalf("shape %v, want %v", got.Shape, want)
		}
	}
}

func TestNewAndFromData(t *testing.T) {
	z := New(2, 3)
	checkShape(t, z, 2, 3)
	checkData(t, z, make([]float32, 6), 0)

	d := FromData([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	checkShape(t, d, 2, 3)
	if got := d.At(1, 2); got != 6 {
		t.Fatalf("At(1,2) = %v, want 6", got)
	}
	d.Set(9, 0, 1)
	if got := d.At(0, 1); got != 9 {
		t.Fatalf("Set/At round trip failed, got %v", got)
	}
}

func TestFromDataPanicsOnMismatch(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on shape/data mismatch")
		}
	}()
	FromData([]float32{1, 2, 3}, 2, 2)
}

func TestCloneIndependence(t *testing.T) {
	a := FromData([]float32{1, 2, 3, 4}, 2, 2)
	b := a.Clone()
	b.Data[0] = 99
	if a.Data[0] != 1 {
		t.Fatal("Clone shares backing array")
	}
}

func TestMatMul(t *testing.T) {
	a := FromRows([]float32{1, 2, 3}, []float32{4, 5, 6})          // 2x3
	b := FromRows([]float32{7, 8}, []float32{9, 10}, []float32{11, 12}) // 3x2
	got := MatMul(a, b)
	checkShape(t, got, 2, 2)
	checkData(t, got, []float32{58, 64, 139, 154}, tol)

	// Identity multiplication.
	id := FromRows([]float32{1, 0}, []float32{0, 1})
	got = MatMul(FromRows([]float32{3, 4}, []float32{5, 6}), id)
	checkData(t, got, []float32{3, 4, 5, 6}, tol)
}

func TestTranspose(t *testing.T) {
	a := FromRows([]float32{1, 2, 3}, []float32{4, 5, 6})
	got := Transpose(a)
	checkShape(t, got, 3, 2)
	checkData(t, got, []float32{1, 4, 2, 5, 3, 6}, 0)
}

func TestBroadcastArithmetic(t *testing.T) {
	a := FromRows([]float32{1, 2, 3}, []float32{4, 5, 6})

	tests := []struct {
		name string
		op   func(x, y *Tensor) *Tensor
		b    *Tensor
		want []float32
	}{
		{"add same shape", Add, FromRows([]float32{10, 20, 30}, []float32{40, 50, 60}), []float32{11, 22, 33, 44, 55, 66}},
		{"add scalar", Add, FromData([]float32{10}, 1), []float32{11, 12, 13, 14, 15, 16}},
		{"add row vec 1d", Add, FromData([]float32{100, 200, 300}, 3), []float32{101, 202, 303, 104, 205, 306}},
		{"add row vec 2d", Add, FromData([]float32{100, 200, 300}, 1, 3), []float32{101, 202, 303, 104, 205, 306}},
		{"sub scalar", Sub, FromData([]float32{1}, 1), []float32{0, 1, 2, 3, 4, 5}},
		{"sub row vec", Sub, FromData([]float32{1, 2, 3}, 3), []float32{0, 0, 0, 3, 3, 3}},
		{"hadamard row vec", Hadamard, FromData([]float32{2, 3, 4}, 3), []float32{2, 6, 12, 8, 15, 24}},
		{"scalar minus tensor", Sub, FromData([]float32{10}, 1), nil}, // handled below
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := tt.b
			want := tt.want
			if want == nil { // scalar minus tensor case
				b = a
				want = []float32{9, 8, 7, 6, 5, 4}
			}
			got := tt.op(a, b)
			if tt.name == "scalar minus tensor" {
				got = tt.op(tt.b, a)
			}
			checkData(t, got, want, tol)
		})
	}
}

func TestElementwiseFuncs(t *testing.T) {
	x := FromData([]float32{0, 1, -1, 2}, 2, 2)

	s := Sigmoid(x)
	e := float32(math.E)
	checkData(t, s, []float32{0.5, e / (1 + e), 1 / (1 + e), 1 / (1 + float32(math.Exp(-2)))}, 1e-5)

	th := Tanh(x)
	checkData(t, th, []float32{0, float32(math.Tanh(1)), float32(math.Tanh(-1)), float32(math.Tanh(2))}, 1e-6)

	l := Log(Exp(x))
	checkData(t, l, x.Data, 1e-5)

	p := Pow(x, 2)
	checkData(t, p, []float32{0, 1, 1, 4}, tol)

	sp := Softplus(x)
	if sp.Data[0] != float32(math.Log(2)) {
		t.Fatalf("softplus(0) = %v, want ln2", sp.Data[0])
	}
	big := Softplus(FromData([]float32{100}, 1))
	if big.Data[0] != 100 {
		t.Fatalf("softplus(100) should be linear, got %v", big.Data[0])
	}

	c := Clip(x, -0.5, 1.5)
	checkData(t, c, []float32{0, 1, -0.5, 1.5}, 0)
}

func TestConcatColAndSliceCol(t *testing.T) {
	a := FromRows([]float32{1, 2}, []float32{3, 4})
	b := FromRows([]float32{5}, []float32{6})
	c := ConcatCol(a, b)
	checkShape(t, c, 2, 3)
	checkData(t, c, []float32{1, 2, 5, 3, 4, 6}, 0)

	s := SliceCol(c, 1, 3)
	checkShape(t, s, 2, 2)
	checkData(t, s, []float32{2, 5, 4, 6}, 0)
}

func TestReductions(t *testing.T) {
	a := FromRows([]float32{1, 2, 3}, []float32{4, 5, 6})

	if got := SumAll(a).Scalar(); got != 21 {
		t.Fatalf("SumAll = %v, want 21", got)
	}
	if got := MeanAll(a).Scalar(); got != 3.5 {
		t.Fatalf("MeanAll = %v, want 3.5", got)
	}
	sr := SumRows(a)
	checkShape(t, sr, 1, 3)
	checkData(t, sr, []float32{5, 7, 9}, 0)
	sc := SumCols(a)
	checkShape(t, sc, 2)
	checkData(t, sc, []float32{6, 15}, 0)
}

func TestSumToShape(t *testing.T) {
	grad := FromRows([]float32{1, 2, 3}, []float32{4, 5, 6})

	same := SumToShape(grad, []int{2, 3})
	checkData(t, same, grad.Data, 0)

	scalar := SumToShape(grad, []int{1})
	if scalar.Scalar() != 21 {
		t.Fatalf("scalar reduce = %v, want 21", scalar.Scalar())
	}

	row := SumToShape(grad, []int{3})
	checkShape(t, row, 3)
	checkData(t, row, []float32{5, 7, 9}, 0)

	row2 := SumToShape(grad, []int{1, 3})
	checkShape(t, row2, 1, 3)
	checkData(t, row2, []float32{5, 7, 9}, 0)
}

func TestSoftmaxRows(t *testing.T) {
	a := FromRows([]float32{1, 2, 3}, []float32{1000, 1001, 1002})
	s := SoftmaxRows(a)
	checkShape(t, s, 2, 3)
	for i := 0; i < 2; i++ {
		sum := s.Data[i*3] + s.Data[i*3+1] + s.Data[i*3+2]
		if !almostEq(sum, 1, 1e-5) {
			t.Fatalf("row %d sums to %v, want 1", i, sum)
		}
	}
	// softmax is shift-invariant: both rows must give the same distribution.
	for j := 0; j < 3; j++ {
		if !almostEq(s.Data[j], s.Data[3+j], 1e-5) {
			t.Fatalf("softmax not shift invariant at col %d: %v vs %v", j, s.Data[j], s.Data[3+j])
		}
	}
}

func TestStack(t *testing.T) {
	a := FromRows([]float32{1, 2}, []float32{3, 4})
	b := FromRows([]float32{5, 6}, []float32{7, 8})
	s := Stack(a, b)
	checkShape(t, s, 2, 2, 2)
	checkData(t, s, []float32{1, 2, 3, 4, 5, 6, 7, 8}, 0)
}

func TestRandom(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	u := Uniform(rng, -2, 3, 4, 5)
	for _, v := range u.Data {
		if v < -2 || v >= 3 {
			t.Fatalf("uniform value %v out of [-2, 3)", v)
		}
	}
	// Reproducibility with the same seed.
	rng2 := rand.New(rand.NewSource(42))
	u2 := Uniform(rng2, -2, 3, 4, 5)
	checkData(t, u, u2.Data, 0)

	n := Randn(rng, 1000)
	var mean float32
	for _, v := range n.Data {
		mean += v
	}
	mean /= float32(len(n.Data))
	if mean < -0.1 || mean > 0.1 {
		t.Fatalf("randn mean %v too far from 0", mean)
	}
}

func TestOuterBroadcast(t *testing.T) {
	col := FromData([]float32{1, 2}, 2, 1) // [2,1]
	row := FromData([]float32{10, 20, 30}, 3)

	got := Hadamard(col, row)
	checkShape(t, got, 2, 3)
	checkData(t, got, []float32{10, 20, 30, 20, 40, 60}, 0)

	// Symmetric order.
	got = Hadamard(row, col)
	checkShape(t, got, 2, 3)
	checkData(t, got, []float32{10, 20, 30, 20, 40, 60}, 0)

	// Full 2D broadcast against a column vector.
	full := FromRows([]float32{1, 2, 3}, []float32{4, 5, 6})
	got = Add(full, col)
	checkData(t, got, []float32{2, 3, 4, 6, 7, 8}, 0)
	got = Sub(full, col)
	checkData(t, got, []float32{0, 1, 2, 2, 3, 4}, 0)
}

func TestSumToShapeColVec(t *testing.T) {
	grad := FromRows([]float32{1, 2, 3}, []float32{4, 5, 6})
	got := SumToShape(grad, []int{2, 1})
	checkShape(t, got, 2, 1)
	checkData(t, got, []float32{6, 15}, 0)
}

func TestSliceRow(t *testing.T) {
	a := FromRows([]float32{1, 2, 3}, []float32{4, 5, 6})
	r := SliceRow(a, 1)
	checkShape(t, r, 1, 3)
	checkData(t, r, []float32{4, 5, 6}, 0)
	// Mutating the slice must not affect the source.
	r.Data[0] = 99
	if a.At(1, 0) != 4 {
		t.Fatal("SliceRow shares backing array")
	}
}

func TestLogSoftmaxRows(t *testing.T) {
	a := FromRows([]float32{1, 2, 3}, []float32{-1000, -1001, -1002})
	ls := LogSoftmaxRows(a)
	checkShape(t, ls, 2, 3)
	s := SoftmaxRows(a)
	for i, v := range ls.Data {
		want := float32(math.Log(float64(s.Data[i])))
		if !almostEq(v, want, 1e-4) {
			t.Fatalf("LogSoftmax[%d] = %v, want %v", i, v, want)
		}
	}
	// Shift invariance also for very negative logits (no NaN).
	for _, v := range ls.Data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("LogSoftmax produced non-finite value %v", v)
		}
	}
}
