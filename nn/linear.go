package nn

import (
	"math"
	"math/rand"

	"lnn/autograd"
	"lnn/tensor"
)

// Linear is a fully connected layer: y = x @ W + b.
type Linear struct {
	W *autograd.Variable // [in, out]
	B *autograd.Variable // [out]
}

// NewLinear creates a layer with Xavier-uniform weights and zero bias.
func NewLinear(in, out int, rng *rand.Rand) *Linear {
	limit := float32(math.Sqrt(6.0 / float64(in+out)))
	return &Linear{
		W: autograd.Var(tensor.Uniform(rng, -limit, limit, in, out)),
		B: autograd.Var(tensor.New(out)),
	}
}

// Forward applies the layer to a [batch, in] variable.
func (l *Linear) Forward(x *autograd.Variable) *autograd.Variable {
	return autograd.Add(autograd.MatMul(x, l.W), l.B)
}

// Parameters returns the trainable variables (W, B).
func (l *Linear) Parameters() []*autograd.Variable {
	return []*autograd.Variable{l.W, l.B}
}
