package autograd

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/qorm/LNN/tensor"
)

// opFunc maps input variables to an output variable.
type opFunc func(inputs ...*Variable) *Variable

func evalScalar(f opFunc, inputs []*Variable) float32 {
	out := f(inputs...)
	if out.Data.IsScalar() {
		return out.Value()
	}
	return tensor.SumAll(out.Data).Scalar()
}

func absf(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// gradCheck compares analytic gradients from Backward against central
// finite differences for every element of every input.
func gradCheck(t *testing.T, name string, f opFunc, inputs ...*Variable) {
	t.Helper()
	for _, in := range inputs {
		in.ZeroGrad()
	}
	out := f(inputs...)
	loss := out
	if !loss.Data.IsScalar() {
		loss = SumAll(loss)
	}
	loss.Backward()
	analytic := make([][]float32, len(inputs))
	for i, in := range inputs {
		if in.Grad == nil {
			t.Fatalf("%s: no gradient for input %d", name, i)
		}
		analytic[i] = append([]float32(nil), in.Grad.Data...)
	}

	const eps = 1e-3
	for k, in := range inputs {
		for j := range in.Data.Data {
			orig := in.Data.Data[j]
			in.Data.Data[j] = orig + eps
			plus := evalScalar(f, inputs)
			in.Data.Data[j] = orig - eps
			minus := evalScalar(f, inputs)
			in.Data.Data[j] = orig

			num := (plus - minus) / (2 * eps)
			ana := analytic[k][j]
			denom := absf(num) + absf(ana) + 1e-4
			if rel := 2 * absf(num-ana) / denom; rel > 2e-2 {
				t.Errorf("%s: input %d elem %d: analytic %v vs numeric %v (rel err %v)",
					name, k, j, ana, num, rel)
			}
		}
	}
}

func randVar(rng *rand.Rand, lo, hi float32, shape ...int) *Variable {
	return Var(tensor.Uniform(rng, lo, hi, shape...))
}

// randVarNonZero returns a variable whose elements stay away from 0, for ops
// that are non-differentiable at the origin (Abs, Relu).
func randVarNonZero(rng *rand.Rand, shape ...int) *Variable {
	v := tensor.Uniform(rng, 0.2, 2, shape...)
	for i := range v.Data {
		if i%2 == 0 {
			v.Data[i] = -v.Data[i]
		}
	}
	return Var(v)
}

func TestOpGradients(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	rand23 := func() *Variable { return randVar(rng, -1, 1, 2, 3) }
	rand32 := func() *Variable { return randVar(rng, -1, 1, 3, 2) }

	tests := []struct {
		name   string
		f      opFunc
		inputs []*Variable
	}{
		{"MatMul", func(v ...*Variable) *Variable { return MatMul(v[0], v[1]) }, []*Variable{rand23(), rand32()}},
		{"Add same shape", func(v ...*Variable) *Variable { return Add(v[0], v[1]) }, []*Variable{rand23(), rand23()}},
		{"Add row broadcast", func(v ...*Variable) *Variable { return Add(v[0], v[1]) }, []*Variable{rand23(), randVar(rng, -1, 1, 3)}},
		{"Add col broadcast", func(v ...*Variable) *Variable { return Add(v[0], v[1]) }, []*Variable{rand23(), randVar(rng, -1, 1, 2, 1)}},
		{"Sub row broadcast", func(v ...*Variable) *Variable { return Sub(v[0], v[1]) }, []*Variable{rand23(), randVar(rng, -1, 1, 3)}},
		{"Sub col broadcast", func(v ...*Variable) *Variable { return Sub(v[0], v[1]) }, []*Variable{rand23(), randVar(rng, -1, 1, 2, 1)}},
		{"Hadamard outer", func(v ...*Variable) *Variable { return Hadamard(v[0], v[1]) }, []*Variable{randVar(rng, -1, 1, 2, 1), randVar(rng, -1, 1, 3)}},
		{"Hadamard same shape", func(v ...*Variable) *Variable { return Hadamard(v[0], v[1]) }, []*Variable{rand23(), rand23()}},
		{"Scale", func(v ...*Variable) *Variable { return Scale(v[0], -2.5) }, []*Variable{rand23()}},
		{"Neg", func(v ...*Variable) *Variable { return Neg(v[0]) }, []*Variable{rand23()}},
		{"Tanh", func(v ...*Variable) *Variable { return Tanh(v[0]) }, []*Variable{rand23()}},
		{"Sigmoid", func(v ...*Variable) *Variable { return Sigmoid(v[0]) }, []*Variable{rand23()}},
		{"Exp", func(v ...*Variable) *Variable { return Exp(v[0]) }, []*Variable{rand23()}},
		{"Log", func(v ...*Variable) *Variable { return Log(v[0]) }, []*Variable{randVar(rng, 0.3, 1.5, 2, 3)}},
		{"Pow2", func(v ...*Variable) *Variable { return Pow(v[0], 2) }, []*Variable{rand23()}},
		{"Pow3", func(v ...*Variable) *Variable { return Pow(v[0], 3) }, []*Variable{rand23()}},
		{"Softplus", func(v ...*Variable) *Variable { return Softplus(v[0]) }, []*Variable{rand23()}},
		{"Abs", func(v ...*Variable) *Variable { return Abs(v[0]) }, []*Variable{randVarNonZero(rng, 2, 3)}},
		{"Relu", func(v ...*Variable) *Variable { return Relu(v[0]) }, []*Variable{randVarNonZero(rng, 2, 3)}},
		{"Div", func(v ...*Variable) *Variable { return Div(v[0], v[1]) }, []*Variable{rand23(), randVar(rng, 0.5, 2, 3)}},
		{"ConcatCol", func(v ...*Variable) *Variable { return ConcatCol(v...) }, []*Variable{randVar(rng, -1, 1, 2, 2), randVar(rng, -1, 1, 2, 1), randVar(rng, -1, 1, 2, 3)}},
		{"SliceCol", func(v ...*Variable) *Variable { return SliceCol(v[0], 1, 3) }, []*Variable{rand23()}},
		{"Col", func(v ...*Variable) *Variable { return Col(v[0], 2) }, []*Variable{rand23()}},
		{"SliceRow", func(v ...*Variable) *Variable { return SliceRow(v[0], 1) }, []*Variable{randVar(rng, -1, 1, 3, 4)}},
		{"SumAll", func(v ...*Variable) *Variable { return SumAll(v[0]) }, []*Variable{rand23()}},
		{"MeanAll", func(v ...*Variable) *Variable { return MeanAll(v[0]) }, []*Variable{rand23()}},
		{"GatherRows", func(v ...*Variable) *Variable { return GatherRows(v[0], []int{0, 2}) }, []*Variable{rand23()}},
		{"LogSoftmaxRows", func(v ...*Variable) *Variable { return LogSoftmaxRows(v[0]) }, []*Variable{randVar(rng, -2, 2, 2, 3)}},
		{"shared input square", func(v ...*Variable) *Variable { return Hadamard(v[0], v[0]) }, []*Variable{rand23()}},
		{"chain add mul", func(v ...*Variable) *Variable {
			return Add(Hadamard(v[0], v[1]), Hadamard(v[0], v[0]))
		}, []*Variable{rand23(), rand23()}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gradCheck(t, tt.name, tt.f, tt.inputs...)
		})
	}
}

func TestCompositeGradient(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	x := randVar(rng, -1, 1, 2, 4)
	w := randVar(rng, -1, 1, 4, 3)
	b := randVar(rng, -1, 1, 3)

	f := func(v ...*Variable) *Variable {
		return SumAll(Tanh(Add(MatMul(v[0], v[1]), v[2])))
	}
	gradCheck(t, "linear+tanh", f, x, w, b)
}

func TestBackwardAccumulatesAndZeroGrad(t *testing.T) {
	a := New([]float32{1, 2}, 2)
	y := SumAll(Hadamard(a, a)) // dy/da = 2a
	y.Backward()
	if got := a.Grad.Data; got[0] != 2 || got[1] != 4 {
		t.Fatalf("grad = %v, want [2 4]", got)
	}
	// Second backward on a fresh graph accumulates further.
	y2 := SumAll(Hadamard(a, a))
	y2.Backward()
	if got := a.Grad.Data; got[0] != 4 || got[1] != 8 {
		t.Fatalf("accumulated grad = %v, want [4 8]", got)
	}
	a.ZeroGrad()
	if a.Grad != nil {
		t.Fatal("ZeroGrad should clear the gradient")
	}
}

func TestBackwardPanicsOnNonScalar(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on non-scalar Backward")
		}
	}()
	a := randVar(rand.New(rand.NewSource(1)), -1, 1, 2, 2)
	Tanh(a).Backward()
}

// TestAddGradPanicsOnShapeMismatch is a regression test for V-12: gradients
// with equal element counts but different shapes ([1, 6] vs [2, 3]) used to be
// added silently, producing wrong gradients without any diagnostic.
func TestAddGradPanicsOnShapeMismatch(t *testing.T) {
	v := New([]float32{1, 2, 3, 4, 5, 6}, 1, 6)
	v.Grad = tensor.New(1, 6)
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic on gradient shape mismatch")
		}
		msg := fmt.Sprint(r)
		if !strings.Contains(msg, "[1 6]") || !strings.Contains(msg, "[2 3]") {
			t.Fatalf("panic message %q should carry both shapes", msg)
		}
	}()
	v.addGrad(tensor.New(2, 3))
}

// TestGatherRowsIdxMutation is a regression test for V-08: the backward
// closure used to capture the caller's idx slice by reference, so mutating it
// between the forward pass and Backward silently corrupted the gradient (the
// red team measured [0 1 1 0] where [1 0 0 1] is correct).
func TestGatherRowsIdxMutation(t *testing.T) {
	a := New([]float32{1, 2, 3, 4}, 2, 2)
	idx := []int{0, 1}
	out := GatherRows(a, idx)

	// Mutate the caller's slice after the forward pass, before Backward.
	idx[0], idx[1] = 1, 0

	SumAll(out).Backward()
	want := []float32{1, 0, 0, 1} // row 0 col 0 and row 1 col 1 gathered
	for i, w := range want {
		if a.Grad.Data[i] != w {
			t.Fatalf("grad = %v, want %v (idx aliasing corrupted the gradient)", a.Grad.Data, want)
		}
	}
}

// TestBackwardTwiceAccumulatesLinearly is a regression test for V-09: calling
// Backward twice on the same graph used to accumulate super-linearly (the red
// team measured 3x) because stale intermediate gradients were re-propagated.
// Leaf gradients must instead grow linearly: two calls == twice one call.
func TestBackwardTwiceAccumulatesLinearly(t *testing.T) {
	a := New([]float32{1, 2, 3}, 3)
	y := SumAll(Hadamard(a, a)) // y = sum(a^2), dy/da = 2a

	y.Backward()
	single := append([]float32(nil), a.Grad.Data...)
	y.Backward() // same graph, second run

	for i := range single {
		if got, want := a.Grad.Data[i], 2*single[i]; got != want {
			t.Fatalf("elem %d: grad after two Backward calls = %v, want %v (2x single %v, not super-linear)",
				i, got, want, single[i])
		}
	}
	// Concrete values: single = [2 4 6], doubled = [4 8 12] (bug gave [6 12 18]).
	want := []float32{4, 8, 12}
	for i, w := range want {
		if a.Grad.Data[i] != w {
			t.Fatalf("grad = %v, want %v", a.Grad.Data, want)
		}
	}
}

// TestPowZeroExponentGradient is a regression test for V-11: the backward of
// Pow(x, 0) computed 0 * x^-1, which is 0*Inf = NaN at x == 0. The gradient of
// the constant function x^0 must be exactly 0 everywhere.
func TestPowZeroExponentGradient(t *testing.T) {
	x := New([]float32{0, 0.5, -2}, 3)
	SumAll(Pow(x, 0)).Backward()
	for i, g := range x.Grad.Data {
		if g != 0 || math.IsNaN(float64(g)) {
			t.Fatalf("grad[%d] = %v, want finite 0", i, g)
		}
	}
}

// TestBackwardWithSeededNonScalarGrad covers the documented contract that
// Backward on a non-scalar node is allowed when its Grad has been seeded
// manually (variable.go doc comment).
func TestBackwardWithSeededNonScalarGrad(t *testing.T) {
	x := New([]float32{1, 2, 3, 4}, 2, 2)
	y := Hadamard(x, x) // non-scalar output
	y.Grad = tensor.FromData([]float32{1, 10, 100, 1000}, 2, 2)

	y.Backward() // must not panic

	// dL/dx = y.Grad ⊙ 2x
	want := []float32{2, 40, 600, 8000}
	for i, w := range want {
		if !almostEq(x.Grad.Data[i], w, 1e-4) {
			t.Fatalf("grad[%d] = %v, want %v", i, x.Grad.Data[i], w)
		}
	}
}

// legacyDiv rebuilds the pre-cleanup Div composition so tests can pin the
// closed-form implementation to the exact historical behavior.
func legacyDiv(a, b *Variable) *Variable { return Hadamard(a, Pow(b, -1)) }

// TestDivRandomGradCheck exercises the closed-form Div quotient gradients on
// random inputs of both signs, plus the row/column broadcast paths and the
// shared-variable case. Denominators stay away from zero so the finite
// differences remain well conditioned.
func TestDivRandomGradCheck(t *testing.T) {
	rng := rand.New(rand.NewSource(101))
	signedDen := func(shape ...int) *Variable {
		v := randVar(rng, 0.5, 2.5, shape...)
		for i := range v.Data.Data {
			if i%2 == 0 {
				v.Data.Data[i] = -v.Data.Data[i]
			}
		}
		return v
	}
	div := func(v ...*Variable) *Variable { return Div(v[0], v[1]) }

	gradCheck(t, "Div random", div, randVar(rng, -2, 2, 2, 3), signedDen(2, 3))
	gradCheck(t, "Div random row broadcast", div, randVar(rng, -2, 2, 2, 3), signedDen(3))
	gradCheck(t, "Div random col broadcast", div, randVar(rng, -2, 2, 2, 3), signedDen(2, 1))
	gradCheck(t, "Div random outer product", div, randVar(rng, -2, 2, 2, 1), signedDen(3))
	gradCheck(t, "Div shared variable", func(v ...*Variable) *Variable { return Div(v[0], v[0]) },
		signedDen(2, 3))
}

// TestDivMatchesLegacyComposition pins debt-#2's zero-behavior-change claim:
// across every broadcast combination the library supports, the closed-form
// Div must produce the same output shape and bit-identical values as the old
// Hadamard(a, Pow(b, -1)) composition, and its leaf gradients must match the
// legacy chain bit-for-bit as well.
func TestDivMatchesLegacyComposition(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	num := func(shape ...int) *tensor.Tensor { return tensor.Uniform(rng, -2, 2, shape...) }
	den := func(shape ...int) *tensor.Tensor {
		d := tensor.Uniform(rng, 0.5, 2.5, shape...)
		for i := range d.Data {
			if i%2 == 0 {
				d.Data[i] = -d.Data[i]
			}
		}
		return d
	}
	cases := []struct {
		name      string
		a, b      *tensor.Tensor
		wantShape []int
	}{
		{"same shape", num(2, 3), den(2, 3), []int{2, 3}},
		{"scalar numerator", num(1), den(2, 3), []int{2, 3}},
		{"scalar denominator", num(2, 3), den(1), []int{2, 3}},
		{"1D row denominator", num(2, 3), den(3), []int{2, 3}},
		{"2D row denominator", num(2, 3), den(1, 3), []int{2, 3}},
		{"1D row numerator", num(3), den(2, 3), []int{2, 3}},
		{"col denominator", num(2, 3), den(2, 1), []int{2, 3}},
		{"col numerator", num(2, 1), den(2, 3), []int{2, 3}},
		{"outer product col/row", num(2, 1), den(3), []int{2, 3}},
		{"outer product row/col", num(3), den(2, 1), []int{2, 3}},
		{"1D over 1D", num(3), den(3), []int{1, 3}},
		{"scalar over scalar", num(1), den(1), []int{1, 1}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := Div(Var(tc.a.Clone()), Var(tc.b.Clone()))
			legacy := legacyDiv(Var(tc.a.Clone()), Var(tc.b.Clone()))

			if !tensor.SameShape(d.Data, legacy.Data) {
				t.Fatalf("shape drift: Div %v vs legacy %v", d.Data.Shape, legacy.Data.Shape)
			}
			want := &tensor.Tensor{Shape: tc.wantShape}
			if !tensor.SameShape(d.Data, want) {
				t.Fatalf("shape %v, want %v", d.Data.Shape, tc.wantShape)
			}
			for i := range d.Data.Data {
				if d.Data.Data[i] != legacy.Data.Data[i] {
					t.Fatalf("elem %d: Div %v vs legacy %v", i, d.Data.Data[i], legacy.Data.Data[i])
				}
			}

			// Gradients must be bit-identical too.
			a1, b1 := Var(tc.a.Clone()), Var(tc.b.Clone())
			a2, b2 := Var(tc.a.Clone()), Var(tc.b.Clone())
			SumAll(Div(a1, b1)).Backward()
			SumAll(legacyDiv(a2, b2)).Backward()
			for _, pair := range []struct {
				name      string
				got, want *tensor.Tensor
			}{{"da", a1.Grad, a2.Grad}, {"db", b1.Grad, b2.Grad}} {
				if !tensor.SameShape(pair.got, pair.want) {
					t.Fatalf("%s shape drift: %v vs legacy %v", pair.name, pair.got.Shape, pair.want.Shape)
				}
				for i := range pair.got.Data {
					if pair.got.Data[i] != pair.want.Data[i] {
						t.Fatalf("%s elem %d: %v vs legacy %v", pair.name, i, pair.got.Data[i], pair.want.Data[i])
					}
				}
			}
		})
	}
}

// TestDivSingleNodeGraph verifies the debt-#2 cleanup itself: Div records one
// closed-form op node wired directly to a and b, where the old composition
// recorded two op nodes (Hadamard over a Pow) that Backward had to traverse.
func TestDivSingleNodeGraph(t *testing.T) {
	countOps := func(root *Variable) int {
		n := 0
		seen := map[*Variable]bool{}
		var walk func(v *Variable)
		walk = func(v *Variable) {
			if seen[v] {
				return
			}
			seen[v] = true
			if len(v.parents) > 0 {
				n++
			}
			for _, p := range v.parents {
				walk(p)
			}
		}
		walk(root)
		return n
	}
	a := New([]float32{1, 2, 3, 4}, 2, 2)
	b := New([]float32{4, 3, 2, 1}, 2, 2)

	d := Div(a, b)
	if len(d.parents) != 2 || d.parents[0] != a || d.parents[1] != b {
		t.Fatalf("Div must wire a and b directly, got %d parents", len(d.parents))
	}
	if got := countOps(d); got != 1 {
		t.Fatalf("Div graph has %d op nodes, want 1", got)
	}
	if got := countOps(legacyDiv(a, b)); got != 2 {
		t.Fatalf("legacy composition has %d op nodes, want 2", got)
	}
}

// TestDivZeroDivisor pins the b == 0 contract: the forward yields +/-Inf (and
// NaN for 0/0) exactly as the legacy composition did, and Backward must not
// panic (gradients may be non-finite; b == 0 is documented as out of contract).
func TestDivZeroDivisor(t *testing.T) {
	a := New([]float32{1, -1, 0}, 1, 3)
	b := New([]float32{0, 0, 0}, 1, 3)

	d := Div(a, b)
	legacy := legacyDiv(a, b)
	for i := range d.Data.Data {
		x, y := float64(d.Data.Data[i]), float64(legacy.Data.Data[i])
		if d.Data.Data[i] != legacy.Data.Data[i] && !(math.IsNaN(x) && math.IsNaN(y)) {
			t.Fatalf("elem %d: Div %v vs legacy %v", i, d.Data.Data[i], legacy.Data.Data[i])
		}
	}
	got := d.Data.Data
	if !math.IsInf(float64(got[0]), 1) || !math.IsInf(float64(got[1]), -1) || !math.IsNaN(float64(got[2])) {
		t.Fatalf("Div at b=0 = %v, want [+Inf -Inf NaN]", got)
	}
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Backward panicked at b=0: %v", r)
		}
	}()
	SumAll(d).Backward()
}

func almostEq(a, b, eps float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= eps
}

func ExampleVariable() {
	// Minimize (x - 3)^2 by gradient descent.
	x := New([]float32{0}, 1)
	for i := 0; i < 200; i++ {
		d := Sub(x, Const(tensor.FromData([]float32{3}, 1)))
		loss := Pow(d, 2)
		x.ZeroGrad()
		loss.Backward()
		x.Data.Data[0] -= 0.1 * x.Grad.Data[0]
	}
	fmt.Printf("x = %.2f", x.Data.Data[0])
	// Output: x = 3.00
}
