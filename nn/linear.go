package nn

import (
	"math"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/tensor"
)

// Linear is a fully connected layer: y = x @ W + b. Both fields are
// trainable graph leaves and both are captured by the persistence
// functions SaveLinear/LoadLinear.
type Linear struct {
	W *autograd.Variable // weight matrix [in, out]
	B *autograd.Variable // bias vector [out], broadcast over the batch
}

// NewLinear creates an [in, out] layer with Xavier-uniform weights
// (U(-sqrt(6/(in+out)), sqrt(6/(in+out)))) and zero bias. rng must be
// non-nil. Panics if in or out is negative (via tensor.New).
func NewLinear(in, out int, rng *rand.Rand) *Linear {
	limit := float32(math.Sqrt(6.0 / float64(in+out)))
	return &Linear{
		W: autograd.Var(tensor.Uniform(rng, -limit, limit, in, out)),
		B: autograd.Var(tensor.New(out)),
	}
}

// Forward applies the layer to a [batch, in] variable, returning a
// [batch, out] variable. Panics if x is not 2D or its column count is
// not l.W's row count (the tensor.MatMul contract).
func (l *Linear) Forward(x *autograd.Variable) *autograd.Variable {
	return autograd.Add(autograd.MatMul(x, l.W), l.B)
}

// Parameters returns the trainable variables in the fixed order W, B.
// The order is the layer's stream order for SaveLinear and part of the
// positional contract of serialize.WriteParameters / optimizer.SaveState.
func (l *Linear) Parameters() []*autograd.Variable {
	return []*autograd.Variable{l.W, l.B}
}
