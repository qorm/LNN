package autograd

import (
	"fmt"
	"math/rand"
	"testing"

	"lnn/tensor"
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
