// Package nn provides neural network building blocks for the LNN library:
// plain Linear layers, wiring topologies, the LTC and CfC liquid cells, and
// a sequence-level RNN wrapper that unfolds cells over time.
package nn

import "lnn/autograd"

// Module is anything that owns trainable parameters.
type Module interface {
	Parameters() []*autograd.Variable
}

// ParametersOf flattens the parameter lists of several modules, convenient
// for handing a whole model to an optimizer.
func ParametersOf(modules ...Module) []*autograd.Variable {
	var out []*autograd.Variable
	for _, m := range modules {
		out = append(out, m.Parameters()...)
	}
	return out
}
