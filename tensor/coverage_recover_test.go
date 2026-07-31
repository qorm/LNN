package tensor

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"testing"
)

// Coverage-recovery tests for tensor-level machinery the stage-7 autograd
// deep rewrite leans on but tensor's own suite never executed: the
// transpose-fused MatMul variants, the broadcast dispatch helpers, shape
// corner cases, the Randn u1 clamp, and the defensive panic contracts.
// Every test asserts values, shapes, bit patterns or exact panic messages.

// bitsSame reports whether a and b carry bit-identical float32 payloads
// (NaN payloads and ±0 distinguished).
func bitsSame(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}

// panicMsg runs f and returns the recovered panic message, or "" if f did
// not panic.
func panicMsg(f func()) string {
	var msg string
	func() {
		defer func() {
			if r := recover(); r != nil {
				msg = fmt.Sprint(r)
			}
		}()
		f()
	}()
	return msg
}

// TestRecoverMatMulZeroSkip exercises MatMul's zero-skipping inner branch:
// a fully zero row and an isolated zero entry must leave the numerical
// result untouched.
func TestRecoverMatMulZeroSkip(t *testing.T) {
	a := FromData([]float32{1, 0, 0, 0, 2, 0}, 3, 2) // sparse, row 1 all zero
	b := FromData([]float32{3, 4, 5, 6}, 2, 2)
	got := MatMul(a, b)
	want := []float32{3, 4, 0, 0, 6, 8}
	if !reflect.DeepEqual(got.Shape, []int{3, 2}) || !bitsSame(got.Data, want) {
		t.Fatalf("MatMul = shape %v data %v, want [3 2] %v", got.Shape, got.Data, want)
	}
}

// randT returns a [shape...] tensor of Uniform values in [-1, 1].
func randT(rng *rand.Rand, shape ...int) *Tensor {
	return Uniform(rng, -1, 1, shape...)
}

// TestRecoverMatMulTransVariantsBitExact proves MatMulTransA/MatMulTransB are
// not merely correct but bit-identical to MatMul over an explicit Transpose:
// the same products in the same accumulation order, zero-skipping included
// (zeros are injected so both skip branches run). The incompatible-shape
// panic messages are pinned as well.
func TestRecoverMatMulTransVariantsBitExact(t *testing.T) {
	rng := rand.New(rand.NewSource(7))

	// TransA: a is [m, k], b is [m, n] -> aᵀ·b of shape [k, n].
	a := randT(rng, 4, 3)
	b := randT(rng, 4, 5)
	a.Data[0] = 0 // isolated zero in the transposed read pattern
	a.Data[7] = 0
	got := MatMulTransA(a, b)
	want := MatMul(Transpose(a), b)
	if !reflect.DeepEqual(got.Shape, []int{3, 5}) {
		t.Fatalf("MatMulTransA shape = %v, want [3 5]", got.Shape)
	}
	if !bitsSame(got.Data, want.Data) {
		t.Fatalf("MatMulTransA diverges from MatMul(Transpose(a), b):\n%v\n%v", got.Data, want.Data)
	}

	// TransB: a is [m, k], b is [n, k] -> a·bᵀ of shape [m, n].
	c := randT(rng, 2, 3)
	d := randT(rng, 6, 3)
	c.Data[3] = 0
	d.Data[5] = 0
	gotB := MatMulTransB(c, d)
	wantB := MatMul(c, Transpose(d))
	if !reflect.DeepEqual(gotB.Shape, []int{2, 6}) {
		t.Fatalf("MatMulTransB shape = %v, want [2 6]", gotB.Shape)
	}
	if !bitsSame(gotB.Data, wantB.Data) {
		t.Fatalf("MatMulTransB diverges from MatMul(a, Transpose(b)):\n%v\n%v", gotB.Data, wantB.Data)
	}

	panics := []struct {
		name string
		f    func()
		want string
	}{
		{"TransA rank", func() { MatMulTransA(New(3), New(4, 5)) },
			"tensor.MatMulTransA: shapes [3] and [4 5] are incompatible"},
		{"TransA rows", func() { MatMulTransA(New(2, 2), New(3, 4)) },
			"tensor.MatMulTransA: shapes [2 2] and [3 4] are incompatible"},
		{"TransB rank", func() { MatMulTransB(New(3), New(4, 3)) },
			"tensor.MatMulTransB: shapes [3] and [4 3] are incompatible"},
		{"TransB cols", func() { MatMulTransB(New(2, 3), New(4, 2)) },
			"tensor.MatMulTransB: shapes [2 3] and [4 2] are incompatible"},
	}
	for _, p := range panics {
		if msg := panicMsg(p.f); msg != p.want {
			t.Fatalf("%s: panic %q, want %q", p.name, msg, p.want)
		}
	}
}

// TestRecoverBcastModeClassification pins the four broadcast participation
// modes and the defensive panic on shapes the library does not broadcast
// (a 3D tensor, unreachable through the public binary ops, which reject it
// one layer up — pinned white-box).
func TestRecoverBcastModeClassification(t *testing.T) {
	cases := []struct {
		t    *Tensor
		want int
	}{
		{FromData([]float32{1}, 1, 1), bcastScalar},
		{FromData([]float32{1}), bcastScalar}, // 0-dim, size 1
		{New(3, 1), bcastCol},
		{New(3, 4), bcastFull},
		{FromData([]float32{1, 2, 3}, 3), bcastRow},
		{New(1, 3), bcastRow},
	}
	for _, c := range cases {
		if got := bcastMode(c.t); got != c.want {
			t.Fatalf("bcastMode(shape %v) = %d, want %d", c.t.Shape, got, c.want)
		}
	}
	if msg := panicMsg(func() { bcastMode(New(2, 2, 2)) }); msg != "tensor: cannot broadcast shape [2 2 2]" {
		t.Fatalf("bcastMode 3D panic %q", msg)
	}
}

// TestRecoverBroadcastShapeWrapper covers the allocation-free wrapper: a
// row-vector operand resolves to the matrix shape, and the outer-product
// combination resolves to the fresh [m, n] shape.
func TestRecoverBroadcastShapeWrapper(t *testing.T) {
	if got := broadcastShape(New(2, 3), FromData([]float32{1, 2, 3}, 3)); !reflect.DeepEqual(got, []int{2, 3}) {
		t.Fatalf("broadcastShape(matrix, row-vec) = %v, want [2 3]", got)
	}
	if got := broadcastShape(New(2, 1), FromData([]float32{1, 2, 3}, 3)); !reflect.DeepEqual(got, []int{2, 3}) {
		t.Fatalf("broadcastShape(col, row-vec) = %v, want the outer shape [2 3]", got)
	}
}

// TestRecoverBroadcastBinaryCornerShapes pins broadcastBinary's shape corner
// cases: the historical 1D->[1, n] lift, the scalar-scalar (0-dim) landing
// at [1], the both-operands-constant fill branch, and the fresh-shape cases
// where the row vector or column vector is the left operand.
func TestRecoverBroadcastBinaryCornerShapes(t *testing.T) {
	p := Hadamard(FromData([]float32{1, 2, 3}, 3), FromData([]float32{4, 5, 6}, 3))
	if !reflect.DeepEqual(p.Shape, []int{1, 3}) || !bitsSame(p.Data, []float32{4, 10, 18}) {
		t.Fatalf("1D lift = shape %v data %v, want [1 3] [4 10 18]", p.Shape, p.Data)
	}

	s := Add(FromData([]float32{2}), FromData([]float32{3}))
	if !reflect.DeepEqual(s.Shape, []int{1}) || s.Data[0] != 5 {
		t.Fatalf("scalar x scalar = shape %v data %v, want [1] [5]", s.Shape, s.Data)
	}

	bc := Add(FromData([]float32{2}, 1), FromData([]float32{7}, 1, 1))
	if !reflect.DeepEqual(bc.Shape, []int{1, 1}) || bc.Data[0] != 9 {
		t.Fatalf("both-constant = shape %v data %v, want [1 1] [9]", bc.Shape, bc.Data)
	}

	lr := Add(FromData([]float32{1, 2, 3}, 3), FromData([]float32{10, 20, 30, 40, 50, 60}, 2, 3))
	if !reflect.DeepEqual(lr.Shape, []int{2, 3}) || !bitsSame(lr.Data, []float32{11, 22, 33, 41, 52, 63}) {
		t.Fatalf("row-vec on left = shape %v data %v", lr.Shape, lr.Data)
	}

	lc := Add(FromData([]float32{100, 200}, 2, 1), FromData([]float32{1, 2, 3, 4, 5, 6}, 2, 3))
	if !reflect.DeepEqual(lc.Shape, []int{2, 3}) || !bitsSame(lc.Data, []float32{101, 102, 103, 204, 205, 206}) {
		t.Fatalf("col-vec on left = shape %v data %v", lc.Shape, lc.Data)
	}
}

// TestRecoverScaleNegOnesLike covers the plain elementwise helpers. Neg is
// asserted at the bit level: it is a v*(-1) multiply, so a zero element
// becomes -0 (sign bit set) rather than staying +0 as a unary minus would.
func TestRecoverScaleNegOnesLike(t *testing.T) {
	in := FromData([]float32{1, -2, 3}, 3)
	s := Scale(in, 2.5)
	if !reflect.DeepEqual(s.Shape, []int{3}) || !bitsSame(s.Data, []float32{2.5, -5, 7.5}) {
		t.Fatalf("Scale = shape %v data %v", s.Shape, s.Data)
	}
	s.Data[0] = 99
	if in.Data[0] != 1 {
		t.Fatal("Scale result aliases the input buffer")
	}

	n := Neg(FromData([]float32{1, -2, 0}, 3))
	negZero := float32(math.Copysign(0, -1))
	if !bitsSame(n.Data, []float32{-1, 2, negZero}) {
		t.Fatalf("Neg = %v; the v*(-1) multiply must turn 0 into -0 (bit pattern)", n.Data)
	}

	o := New(2, 2).OnesLike()
	if !reflect.DeepEqual(o.Shape, []int{2, 2}) || !bitsSame(o.Data, []float32{1, 1, 1, 1}) {
		t.Fatalf("OnesLike = shape %v data %v", o.Shape, o.Data)
	}
	src := FromData([]float32{5, 6}, 2)
	o2 := src.OnesLike()
	o2.Data[0] = 0
	if src.Data[0] != 5 {
		t.Fatal("OnesLike aliases the source buffer")
	}
}

// TestRecoverStringRendering pins both rendering forms verbatim.
func TestRecoverStringRendering(t *testing.T) {
	small := FromData([]float32{1, 2, 3, 4}, 2, 2)
	if got, want := small.String(), "Tensor(shape=[2 2], data=[1 2 3 4])"; got != want {
		t.Fatalf("String small = %q, want %q", got, want)
	}
	big := New(65)
	for i := range big.Data {
		big.Data[i] = float32(i)
	}
	if got, want := big.String(), "Tensor(shape=[65], data=[0 ... 64])"; got != want {
		t.Fatalf("String big = %q, want %q", got, want)
	}
}

// TestRecoverTensorPanicContracts pins the defensive panic messages so a
// refactor cannot silently reword them. The SumToShape case drives the
// irreducible default arm (the owning SumToShapeTake variant moved into the
// autograd package in v0.4.0 as the unexported sumToShapeTake, whose identical
// panic contract is pinned by an autograd internal test); the must2D/offset
// cases drive the rank and arity guards.
func TestRecoverTensorPanicContracts(t *testing.T) {
	cases := []struct {
		name string
		f    func()
		want string
	}{
		{"ConcatCol empty", func() { ConcatCol() },
			"tensor.ConcatCol: no tensors"},
		{"ConcatCol 1D operand", func() { ConcatCol(New(2, 2), New(3)) },
			"tensor.ConcatCol: shape [3] incompatible with 2 rows"},
		{"ConcatCol row mismatch", func() { ConcatCol(New(2, 2), New(3, 2)) },
			"tensor.ConcatCol: shape [3 2] incompatible with 2 rows"},
		{"SliceRow beyond", func() { SliceRow(New(2, 2), 2) },
			"tensor.SliceRow: invalid row 2 for shape [2 2]"},
		{"SliceRow negative", func() { SliceRow(New(2, 2), -1) },
			"tensor.SliceRow: invalid row -1 for shape [2 2]"},
		{"SliceRow rank", func() { SliceRow(New(3), 0) },
			"tensor.SliceRow: invalid row 0 for shape [3]"},
		{"SumToShape irreducible", func() { SumToShape(New(2, 2), []int{3}) },
			"tensor.SumToShape: cannot reduce shape [2 2] to [3]"},
		{"must2D on 1D", func() { New(3).Rows() },
			"tensor: expected 2D tensor, got shape [3]"},
		{"must2D on 3D", func() { New(2, 2, 2).Cols() },
			"tensor: expected 2D tensor, got shape [2 2 2]"},
		{"offset arity low", func() { New(2, 2).At(0) },
			"tensor: 1 indices for shape [2 2]"},
		{"offset arity high", func() { New(2, 2).At(0, 0, 0) },
			"tensor: 3 indices for shape [2 2]"},
		{"Scalar non-scalar", func() { New(2).Scalar() },
			"tensor.Scalar: shape [2] is not scalar"},
	}
	for _, c := range cases {
		if msg := panicMsg(c.f); msg != c.want {
			t.Fatalf("%s: panic %q, want %q", c.name, msg, c.want)
		}
	}
}

// TestRecoverFromRowsEmpty pins the degenerate constructor result: no rows
// yields the empty [0, 0] tensor, not a panic.
func TestRecoverFromRowsEmpty(t *testing.T) {
	e := FromRows()
	if !reflect.DeepEqual(e.Shape, []int{0, 0}) || len(e.Data) != 0 {
		t.Fatalf("FromRows() = shape %v data-len %d, want [0 0] / 0", e.Shape, len(e.Data))
	}
}

// zeroSource is a degenerate rand.Source whose every draw is 0, so Randn's
// u1 arrives as 0.0 and the documented 1e-12 clamp must engage on both the
// pair loop and the odd-tail branch.
type zeroSource struct{}

func (zeroSource) Int63() int64 { return 0 }
func (zeroSource) Seed(int64)   {}

// TestRecoverRandnU1Clamp pins the clamp's observable contract: with u1
// forced to zero, every radius is exactly sqrt(-2 ln 1e-12) and the u2 = 0
// angle puts the full magnitude on the cos sample (sin sample +0). The final
// assertion pins the documented hard truncation: no sample from an ordinary
// rng ever exceeds the clamp radius.
func TestRecoverRandnU1Clamp(t *testing.T) {
	rng := rand.New(zeroSource{})
	radius := math.Sqrt(-2 * math.Log(1e-12))
	peak := float32(radius) // cos(0) = 1, sin(0) = +0

	pair := Randn(rng, 4)
	if !bitsSame(pair.Data, []float32{peak, 0, peak, 0}) {
		t.Fatalf("Randn(4) with clamped u1 = %v, want [%v 0 %v 0]", pair.Data, peak, peak)
	}
	odd := Randn(rng, 3)
	if !bitsSame(odd.Data, []float32{peak, 0, peak}) {
		t.Fatalf("Randn(3) with clamped u1 = %v, want [%v 0 %v] (odd tail uses cos)", odd.Data, peak, peak)
	}

	reg := Randn(rand.New(rand.NewSource(3)), 1000)
	for i, v := range reg.Data {
		if math.Abs(float64(v)) > float64(peak) {
			t.Fatalf("sample %d = %v exceeds the documented clamp ceiling %v", i, v, peak)
		}
	}
}
