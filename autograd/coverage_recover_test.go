package autograd

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/qorm/LNN/tensor"
)

// Coverage-recovery tests for the stage-7 P-B autograd deep rewrite (opKind
// tag dispatch, fused unary backwards, hadamardReduce/negReduce fusion,
// broadcastLift/productCarriesGShape helpers) and the stage-8 SigmoidHadamard
// fusion. Every test asserts values, shapes, bit patterns or panic messages;
// the fallback-branch tests rebuild the legacy composition with tensor-level
// ops and demand bitwise equality, which is the contract those branches exist
// to honor.

// recoverMsg runs f and returns the recovered panic message, or "" if f did
// not panic.
func recoverMsg(f func()) string {
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

// TestRecoverUnaryIrregularSeedFallbacks exercises the gradMatchesElemwise
// fallback of every fused unary backward (Tanh/Sigmoid/Log/Pow/Abs/Relu):
// with a manually seeded Grad whose [1, 1] shape does not match the node's
// [2, 2] shape, each op must reproduce the legacy composition exactly. The
// expected gradient is rebuilt with the legacy tensor-level op chain —
// derivatives from the node output for Tanh/Sigmoid, from the parent's data
// for Log/Pow/Abs/Relu — and compared bit for bit. The Abs/Relu inputs
// include a zero so every arm of their sign/mask classification runs.
func TestRecoverUnaryIrregularSeedFallbacks(t *testing.T) {
	seed := func() *tensor.Tensor {
		return tensor.FromData([]float32{3}, 1, 1)
	}
	xGeneral := []float32{-0.7, 0.4, 1.2, -2} // Tanh/Sigmoid/Pow domain
	xPositive := []float32{0.3, 1.5, 2.25, 7} // Log domain
	xSigned := []float32{-2, 0, 0.5, 3}       // exercises the Abs zero arm and the Relu mask

	cases := []struct {
		name string
		op   func(*Variable) *Variable
		x    []float32
		// deriv rebuilds the legacy derivative: out is the node's forward
		// output (v.Data), parent is the operand's data (a.Data).
		deriv func(out, parent *tensor.Tensor) *tensor.Tensor
	}{
		{"Tanh", Tanh, xGeneral, func(out, parent *tensor.Tensor) *tensor.Tensor {
			one := out.OnesLike()
			return tensor.Sub(one, tensor.Hadamard(out, out))
		}},
		{"Sigmoid", Sigmoid, xGeneral, func(out, parent *tensor.Tensor) *tensor.Tensor {
			one := out.OnesLike()
			return tensor.Hadamard(out, tensor.Sub(one, out))
		}},
		{"Log", Log, xPositive, func(out, parent *tensor.Tensor) *tensor.Tensor {
			return tensor.Apply(parent, func(v float32) float32 { return 1 / v })
		}},
		{"Pow2", func(v *Variable) *Variable { return Pow(v, 2) }, xGeneral,
			func(out, parent *tensor.Tensor) *tensor.Tensor {
				return tensor.Scale(tensor.Pow(parent, 1), 2)
			}},
		{"Abs", Abs, xSigned, func(out, parent *tensor.Tensor) *tensor.Tensor {
			return tensor.Apply(parent, func(v float32) float32 {
				switch {
				case v > 0:
					return 1
				case v < 0:
					return -1
				default:
					return 0
				}
			})
		}},
		{"Relu", Relu, xSigned, func(out, parent *tensor.Tensor) *tensor.Tensor {
			return tensor.Apply(parent, func(v float32) float32 {
				if v > 0 {
					return 1
				}
				return 0
			})
		}},
	}

	for _, c := range cases {
		x := New(append([]float32(nil), c.x...), 2, 2)
		out := c.op(x)
		out.Grad = seed()
		out.Backward()

		want := tensor.Hadamard(seed(), c.deriv(out.Data, x.Data))
		if x.Grad == nil {
			t.Fatalf("%s: leaf gradient is nil", c.name)
		}
		if !bitsEqual(x.Grad, want) {
			t.Fatalf("%s: fallback grad = %v (shape %v), want %v (shape %v)",
				c.name, x.Grad.Data, x.Grad.Shape, want.Data, want.Shape)
		}
		// V-09: Backward must restore the caller's seed unmutated.
		if !bitsEqual(out.Grad, seed()) {
			t.Fatalf("%s: seeded grad not restored pristine: %v", c.name, out.Grad.Data)
		}
	}
}

// TestRecoverLogSoftmaxRowsIrregularSeed covers the opLogSoftmaxRows fallback
// under a non-matching [1, 1] seed: it must replicate the legacy
// Exp/SumCols/Hadamard/Sub composition element for element, including the
// rowsum reshape to [rows, 1].
func TestRecoverLogSoftmaxRowsIrregularSeed(t *testing.T) {
	x := New([]float32{0.5, -1, 2, 0.25, 1.5, -0.75}, 2, 3)
	out := LogSoftmaxRows(x)
	out.Grad = tensor.FromData([]float32{2}, 1, 1)
	out.Backward()

	seedT := tensor.FromData([]float32{2}, 1, 1)
	softmax := tensor.Exp(out.Data)
	rowsum := tensor.SumCols(seedT)
	rowsum.Reshape(rowsum.Size(), 1)
	want := tensor.Sub(seedT, tensor.Hadamard(softmax, rowsum))
	if x.Grad == nil || !bitsEqual(x.Grad, want) {
		t.Fatalf("fallback grad = %v (shape %v), want %v (shape %v)",
			x.Grad.Data, x.Grad.Shape, want.Data, want.Shape)
	}
}

// TestRecoverLogSoftmaxRows1DSeedPanicContract pins the documented panic
// replication: the legacy composition's SumCols reduction panics on a 1D
// seed, and the fused node must surface the very same panic rather than
// silently doing something new.
func TestRecoverLogSoftmaxRows1DSeedPanicContract(t *testing.T) {
	x := New([]float32{0.5, -1, 2, 0.25, 1.5, -0.75}, 2, 3)
	out := LogSoftmaxRows(x)
	out.Grad = tensor.FromData([]float32{1, 1, 1}, 3)
	msg := recoverMsg(func() { out.Backward() })
	if !strings.Contains(msg, "tensor: expected 2D tensor, got shape [3]") {
		t.Fatalf("panic %q does not reproduce the legacy SumCols 2D contract", msg)
	}
}

// TestRecoverSigmoidHadamard0DFallback covers the non-2D leg of the
// opSigmoidHadamard backward all the way into its own irregular-shape
// fallback: with 0-dim z and w (shape []), the hadamardReduce products are
// lifted to [1], which matches neither z's empty shape nor its
// elemwiseGradShape lift, so gradMatchesElemwise fails and the backward must
// run the legacy opSigmoid-style broadcast composition — whose final
// Hadamard lift leaves z's gradient shaped [1, 1]. Values are pinned bit for
// bit against hand-evaluated float32 arithmetic.
func TestRecoverSigmoidHadamard0DFallback(t *testing.T) {
	z := Var(tensor.FromData([]float32{0.5}))
	w := Var(tensor.FromData([]float32{2}))
	out := SigmoidHadamard(z, w)

	// Forward: sigmoid(0.5) computed exactly as tensor.sigmoid does it,
	// then the scalar-scalar Hadamard product with w.
	e := float32(math.Exp(-0.5))
	s := float32(1 / (1 + e))
	if math.Float32bits(out.Data.Data[0]) != math.Float32bits(s*2) {
		t.Fatalf("forward = %v, want %v", out.Data.Data[0], s*2)
	}

	out.Backward() // size-1 output: auto-seeded with ones

	// dz: gs = g*w = 2 (scalar-scalar product), deriv = s*(1-s) in float32
	// grouping, final product 2*deriv — the fallback's three tensor ops.
	wantDZ := float32(2) * (s * (1 - s))
	if z.Grad == nil || len(z.Grad.Shape) != 2 || z.Grad.Shape[0] != 1 || z.Grad.Shape[1] != 1 {
		t.Fatalf("z.Grad shape = %v, want [1 1] (legacy Hadamard 1D lift)", z.Grad.Shape)
	}
	if math.Float32bits(z.Grad.Data[0]) != math.Float32bits(wantDZ) {
		t.Fatalf("z.Grad = %v (bits %08x), want %v (bits %08x)",
			z.Grad.Data[0], math.Float32bits(z.Grad.Data[0]), wantDZ, math.Float32bits(wantDZ))
	}
	// dw: g*s reduced to w's shape, [1].
	wantDW := s
	if w.Grad == nil || len(w.Grad.Shape) != 1 || w.Grad.Shape[0] != 1 {
		t.Fatalf("w.Grad shape = %v, want [1]", w.Grad.Shape)
	}
	if math.Float32bits(w.Grad.Data[0]) != math.Float32bits(wantDW) {
		t.Fatalf("w.Grad = %v, want %v", w.Grad.Data[0], wantDW)
	}
}

// TestRecoverProductCarriesGShapeDispatch walks every case arm of the
// broadcast-shape classifier the fused hadamardReduce branches guard on,
// asserting the exact boolean each operand-shape combination must yield.
func TestRecoverProductCarriesGShapeDispatch(t *testing.T) {
	cases := []struct {
		name string
		g, x *tensor.Tensor
		want bool
	}{
		// g scalar: s = x.Shape; the empty shape lifts to [1], matching g.
		{"gScalar_x0D_liftMatches", tensor.FromData([]float32{1}, 1), tensor.FromData([]float32{2}), true},
		// g scalar against a 2D x lifts wider than [1].
		{"gScalar_x2D_liftTooWide", tensor.FromData([]float32{1}, 1), tensor.New(2, 2), false},
		// x scalar: s = g.Shape, the product carries g's shape.
		{"xScalar_carriesG", tensor.New(2, 2), tensor.FromData([]float32{3}, 1), true},
		// x 2D, g a 1D row vector: s = x.Shape [1, 3] vs g [3].
		{"x2D_gRowVec_liftMismatch", tensor.FromData([]float32{1, 2, 3}, 3), tensor.New(1, 3), false},
		// g a column vector, x full 2D: outer shape [2, 3] vs g [2, 1].
		{"gCol_xFull_outerShape", tensor.New(2, 1), tensor.New(2, 3), false},
		// g a column vector, x a row vector: fresh outer shape vs g.
		{"gCol_xRowVec_outer", tensor.New(2, 1), tensor.FromData([]float32{1, 2, 3}, 3), false},
		// x a column vector, g a row vector: mirror outer case.
		{"xCol_gRowVec_outer", tensor.FromData([]float32{1, 2, 3}, 3), tensor.New(2, 1), false},
		// Nothing matches: unbroadcastable x falls into the default false.
		{"unbroadcastable_default", tensor.New(2, 2), tensor.New(2, 2, 2), false},
	}
	for _, c := range cases {
		if got := productCarriesGShape(c.g, c.x); got != c.want {
			t.Fatalf("%s: productCarriesGShape(g %v, x %v) = %v, want %v",
				c.name, c.g.Shape, c.x.Shape, got, c.want)
		}
	}
}

// TestRecoverBroadcastLiftNormalization pins broadcastLift's three arms: the
// empty shape (a scalar-scalar product) lands at [1], a 1D shape lifts to
// [1, n], and 2D shapes pass through as the very same slice (no copy — the
// fused branches rely on pass-through identity).
func TestRecoverBroadcastLiftNormalization(t *testing.T) {
	if got := broadcastLift(nil); len(got) != 1 || got[0] != 1 {
		t.Fatalf("broadcastLift(empty) = %v, want [1]", got)
	}
	if got := broadcastLift([]int{5}); len(got) != 2 || got[0] != 1 || got[1] != 5 {
		t.Fatalf("broadcastLift([5]) = %v, want [1 5]", got)
	}
	in := []int{2, 3}
	if got := broadcastLift(in); len(got) != 2 || &got[0] != &in[0] {
		t.Fatalf("broadcastLift must pass 2D shapes through unaltered, got %v", got)
	}
}

// TestRecoverNegReduceLegacyFallback proves the opSub b-branch falls through
// to negReduce's default arm — the unfused sumToShapeTake(Neg(g), shape)
// composition — and therefore reproduces that composition's panic contract
// when the seeded shape can be neither reduced nor broadcast.
func TestRecoverNegReduceLegacyFallback(t *testing.T) {
	a := New([]float32{1}, 1, 1)
	b := New([]float32{1, 2, 3, 4}, 2, 2)
	out := Sub(a, b) // [1, 1] broadcasts against [2, 2]
	out.Grad = tensor.FromData([]float32{4}, 1)
	msg := recoverMsg(func() { out.Backward() })
	if !strings.Contains(msg, "tensor.SumToShape: cannot reduce shape [1] to [2 2]") {
		t.Fatalf("panic %q does not come from negReduce's legacy fallback", msg)
	}
}

// TestRecoverXRowAccessDispatchAndPanic asserts the four broadcast access
// modes (scalar / column vector / full matrix / row vector) return exactly
// the (base, stride) pairs the fused loops index with, and that the
// defensive default keeps its panic message contract. The default is
// unreachable through the public API (forward broadcasting rejects such
// operands before a node exists), so it is pinned white-box.
func TestRecoverXRowAccessDispatchAndPanic(t *testing.T) {
	if base, stride := xRowAccess(tensor.FromData([]float32{7}, 1, 1), 2, 4); base != 0 || stride != 0 {
		t.Fatalf("scalar access = (%d, %d), want (0, 0)", base, stride)
	}
	if base, stride := xRowAccess(tensor.New(3, 1), 2, 4); base != 2 || stride != 0 {
		t.Fatalf("column access = (%d, %d), want (2, 0)", base, stride)
	}
	if base, stride := xRowAccess(tensor.New(3, 4), 2, 4); base != 8 || stride != 1 {
		t.Fatalf("full access = (%d, %d), want (8, 1)", base, stride)
	}
	if base, stride := xRowAccess(tensor.FromData([]float32{1, 2, 3, 4}, 4), 2, 4); base != 0 || stride != 1 {
		t.Fatalf("row-vector access = (%d, %d), want (0, 1)", base, stride)
	}
	msg := recoverMsg(func() { xRowAccess(tensor.New(2, 2, 2), 0, 2) })
	if msg != "tensor: cannot broadcast shape [2 2 2]" {
		t.Fatalf("panic %q, want the broadcast-shape contract message", msg)
	}
}

// TestRecoverConstructorPanicContracts pins the validation panic messages of
// the graph constructors added or touched by the rewrite, so a future
// refactor cannot silently reword them. The upper-bound GatherRows case uses
// idx == cols exactly, distinguishing j >= n from the off-by-one j > n.
func TestRecoverConstructorPanicContracts(t *testing.T) {
	cases := []struct {
		name string
		f    func()
		want string
	}{
		{"ConcatCol no inputs", func() { ConcatCol() },
			"autograd.ConcatCol: no inputs"},
		{"GatherRows 1D operand", func() { GatherRows(New([]float32{1, 2, 3}, 3), nil) },
			"autograd.GatherRows: shape [3] vs 0 indices"},
		{"GatherRows index count", func() { GatherRows(New([]float32{1, 2, 3, 4}, 2, 2), []int{0}) },
			"autograd.GatherRows: shape [2 2] vs 1 indices"},
		{"GatherRows upper bound", func() { GatherRows(New([]float32{1, 2, 3, 4}, 2, 2), []int{0, 2}) },
			"autograd.GatherRows: index 2 out of bounds for 2 columns"},
		{"GatherRows negative", func() { GatherRows(New([]float32{1, 2, 3, 4}, 2, 2), []int{-1, 0}) },
			"autograd.GatherRows: index -1 out of bounds for 2 columns"},
	}
	for _, c := range cases {
		if msg := recoverMsg(c.f); msg != c.want {
			t.Fatalf("%s: panic %q, want %q", c.name, msg, c.want)
		}
	}
}
