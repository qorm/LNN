// Package nn provides neural network building blocks for the LNN library:
// the Linear layer, Wiring synapse topologies, the LTC liquid cell, and the
// Cell/Unroll abstractions for driving recurrent cells over sequences.
//
// Two update styles are supported: the hand-rolled loop over Parameters()
// (plain Go over p.Data/p.Grad — the basis for understanding the engine)
// and the optimizer package (SGD/Momentum/Adam), which packages that same
// loop and is the recommended form for production training. See
// examples/ltc-sequence for an end-to-end training loop and doc/training.md
// for both styles.
//
// Concurrency contract: lnn is single-threaded by design. Backward mutates
// the Grad buffers of a graph's leaf variables without synchronization, so
// running it concurrently on variables that share parameters is a data race
// that loses or corrupts gradients (verified under go test -race). Never
// share a Variable, Tensor, or computation graph across goroutines.
package nn

import "github.com/qorm/LNN/autograd"

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
