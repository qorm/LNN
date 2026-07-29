// Package autograd implements a small dynamic computation-graph engine.
// Every operation records a backward closure on its output Variable, and
// Backward walks the graph in reverse topological order accumulating
// gradients into leaf Variables (typically model parameters).
package autograd

import (
	"fmt"

	"lnn/tensor"
)

// Variable is a node in the computation graph: a tensor value plus its
// accumulated gradient and the backward closure that propagates it.
type Variable struct {
	Data *tensor.Tensor
	Grad *tensor.Tensor

	parents  []*Variable
	backward func()
}

// Var wraps a tensor as a graph leaf (e.g. a parameter or an input).
func Var(t *tensor.Tensor) *Variable {
	return &Variable{Data: t}
}

// New builds a leaf Variable from raw data and a shape.
func New(data []float32, shape ...int) *Variable {
	return Var(tensor.FromData(data, shape...))
}

// Const is an alias for Var, used at call sites to clarify that the value is
// a constant input rather than a trainable parameter. Gradients still flow
// into it if it sits inside the graph; simply ignore them.
func Const(t *tensor.Tensor) *Variable { return Var(t) }

// newOp creates the output Variable of an operation and records its parents
// and backward closure.
func newOp(data *tensor.Tensor, parents []*Variable, backward func()) *Variable {
	return &Variable{Data: data, parents: parents, backward: backward}
}

// addGrad accumulates g into the variable's gradient buffer.
func (v *Variable) addGrad(g *tensor.Tensor) {
	if v.Grad == nil {
		v.Grad = g.Clone()
		return
	}
	for i := range v.Grad.Data {
		v.Grad.Data[i] += g.Data[i]
	}
}

// ZeroGrad clears the accumulated gradient.
func (v *Variable) ZeroGrad() { v.Grad = nil }

// Backward runs reverse-mode differentiation from v. v must be a scalar (the
// usual case for a loss) unless its Grad has been seeded manually.
func (v *Variable) Backward() {
	if v.Grad == nil {
		if !v.Data.IsScalar() {
			panic(fmt.Sprintf("autograd.Backward: non-scalar output of shape %v needs a seeded Grad", v.Data.Shape))
		}
		v.Grad = v.Data.OnesLike()
	}
	topo := make([]*Variable, 0, 16)
	visited := make(map[*Variable]bool)
	var build func(n *Variable)
	build = func(n *Variable) {
		if visited[n] {
			return
		}
		visited[n] = true
		for _, p := range n.parents {
			build(p)
		}
		topo = append(topo, n)
	}
	build(v)
	for i := len(topo) - 1; i >= 0; i-- {
		if topo[i].backward != nil {
			topo[i].backward()
		}
	}
}

// Value returns the scalar value of a size-1 variable.
func (v *Variable) Value() float32 { return v.Data.Scalar() }
