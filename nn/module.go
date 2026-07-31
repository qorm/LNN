package nn

import "github.com/qorm/LNN/autograd"

// Module is anything that owns trainable parameters.
type Module interface {
	Parameters() []*autograd.Variable
}

// ParametersOf flattens the parameter lists of several modules,
// convenient for handing a whole model to an optimizer. The order is
// deterministic — modules in the order given, each in its own
// Parameters() order — and significant: serialize.WriteParameters /
// LoadParameters and optimizer.SaveState / LoadState all key values by
// position, so the same order must be reproduced when loading a
// checkpoint. With no modules it returns nil.
func ParametersOf(modules ...Module) []*autograd.Variable {
	var out []*autograd.Variable
	for _, m := range modules {
		out = append(out, m.Parameters()...)
	}
	return out
}
