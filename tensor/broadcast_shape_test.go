package tensor

import "testing"

// White-box tests for the broadcast metadata fast path (stage 19a C10):
// broadcastShapeFresh carries fresh output shapes in a fixed [2]int array
// instead of a heap slice, so every broadcastBinary arm lands at the
// allocation floor — the result struct plus its data buffer — and the
// operand-owned arms keep returning the operand's Shape for New to copy.

// TestBroadcastShapeFreshDispatch pins the classifier's contract arm by arm:
// which cases return an operand-owned shape (and which operand), which
// return a fresh rank-2 shape in the fixed array, and that the fresh flag
// and the nil shape on fresh arms stay in lockstep (broadcastBinary's
// scalar-scalar guard keys on exactly that pair).
func TestBroadcastShapeFreshDispatch(t *testing.T) {
	mat := New(2, 3)
	row := FromData([]float32{1, 2, 3}, 3)
	col := New(2, 1)
	sc := FromData([]float32{5}, 1)

	// Operand-owned arms: fresh must be false and the returned shape must
	// be the expected operand's very slice (no copy, no fresh storage).
	owned := []struct {
		name string
		a, b *Tensor
		want *Tensor // the operand whose Shape must come back
	}{
		{"sameShape", mat, New(2, 3), mat},
		{"aScalar", sc, mat, mat},
		{"bScalar", mat, sc, mat},
		{"matRowVec", mat, row, mat},
		{"rowVecMat", row, mat, mat},
		{"colMat", col, mat, mat},
		{"matCol", mat, col, mat},
	}
	for _, c := range owned {
		shape, _, fresh := broadcastShapeFresh(c.a, c.b)
		if fresh {
			t.Fatalf("%s: fresh = true for an operand-owned case", c.name)
		}
		if len(shape) != len(c.want.Shape) || &shape[0] != &c.want.Shape[0] {
			t.Fatalf("%s: shape %v does not alias %v as the contract requires", c.name, shape, c.want.Shape)
		}
	}

	// Fresh arms (outer products): shape nil, the fixed array carries [m, n].
	freshCases := []struct {
		name string
		a, b *Tensor
		want [2]int
	}{
		{"colRowVec", col, row, [2]int{2, 3}},
		{"rowVecCol", row, New(3, 1), [2]int{3, 3}},
	}
	for _, c := range freshCases {
		shape, fs, fresh := broadcastShapeFresh(c.a, c.b)
		if !fresh || shape != nil || fs != c.want {
			t.Fatalf("%s: (shape %v, freshShape %v, fresh %v), want (nil, %v, true)",
				c.name, shape, fs, fresh, c.want)
		}
	}
}

// TestBroadcastOutputShapeIndependence proves the output tensor never shares
// shape storage with an operand, on both the operand-owned path (New copies
// into the inline backing) and the fresh/adopting path (useShape copies the
// stack array): mutating the result's Shape in place must leave the
// operands' shapes untouched. White-box: the in-place write is exactly what
// the type doc forbids callers to do, done here to expose aliasing.
func TestBroadcastOutputShapeIndependence(t *testing.T) {
	a := New(2, 3)
	b := New(2, 3)
	out := Add(a, b) // operand-owned arm (same shape)
	out.Shape[0] = 99
	if a.Shape[0] != 2 || b.Shape[0] != 2 {
		t.Fatalf("same-shape result aliases an operand's shape: a %v, b %v", a.Shape, b.Shape)
	}

	col := New(2, 1)
	row := FromData([]float32{1, 2, 3}, 3)
	outer := Hadamard(col, row) // fresh arm (outer product)
	outer.Shape[0] = 99
	if col.Shape[0] != 2 {
		t.Fatalf("outer-product result aliases the column operand's shape: %v", col.Shape)
	}

	d1 := FromData([]float32{1, 2, 3}, 3)
	d2 := FromData([]float32{4, 5, 6}, 3)
	lifted := Add(d1, d2) // fresh arm (1D -> [1, n] lift)
	lifted.Shape[1] = 99
	if d1.Shape[0] != 3 || d2.Shape[0] != 3 {
		t.Fatalf("lifted result aliases an operand's shape: %v, %v", d1.Shape, d2.Shape)
	}
}

// TestBroadcastAllocationFloor pins every broadcastBinary shape arm at the
// two-allocation floor (the result struct and its data buffer): shape
// metadata must never add a heap allocation of its own. Exact-equality is
// deliberate — this is the regression gate for the fixed-array fresh-shape
// carrier, and the repo pins its toolchain in go.mod.
func TestBroadcastAllocationFloor(t *testing.T) {
	a := New(8, 16)
	b := New(8, 16)
	row := FromData(make([]float32, 16), 16)
	row2d := New(1, 16)
	sc := New(1)
	col := New(8, 1)
	d1 := New(16)
	d2 := New(16)
	cases := []struct {
		name string
		f    func() *Tensor
	}{
		{"sameShape2D", func() *Tensor { return Add(a, b) }},
		{"matRowVec1D", func() *Tensor { return Add(a, row) }},
		{"matRowVec2D", func() *Tensor { return Add(a, row2d) }},
		{"matScalar", func() *Tensor { return Add(a, sc) }},
		{"matCol", func() *Tensor { return Add(a, col) }},
		{"outerProduct", func() *Tensor { return Hadamard(col, row) }},
		{"sameShape1DLift", func() *Tensor { return Add(d1, d2) }},
	}
	for _, c := range cases {
		if n := testing.AllocsPerRun(100, func() { benchSink = c.f() }); n != 2 {
			t.Fatalf("%s: %.1f allocs/op, want the 2-allocation floor (result + data)", c.name, n)
		}
	}
}
