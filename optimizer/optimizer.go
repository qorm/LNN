package optimizer

import "lnn/autograd"

// Optimizer applies an in-place update to params using each variable's
// accumulated Grad. Callers own the training loop: build a fresh graph,
// ZeroGrad every parameter, loss.Backward(), then Step — the four-phase
// discipline of doc/training.md, with the hand-rolled update loop
// replaced by one method call.
//
// Step never calls ZeroGrad. Leaf gradients accumulate across Backward
// calls by design (see autograd.Variable), so resetting them is the
// caller's contract: zeroing before every Backward gives the usual
// per-iteration gradient, while zeroing every N iterations gives
// gradient accumulation for free. An optimizer that zeroed on the
// caller's behalf would silently break that pattern.
//
// Step skips parameters whose Grad is nil — a parameter that did not
// take part in the graph just built (e.g. an unused module handed over
// by nn.ParametersOf) keeps its Data and any accumulated optimizer
// state. It assumes p.Grad has the same shape as p.Data, which
// autograd's addGrad guarantees.
type Optimizer interface {
	Step(params []*autograd.Variable)
}
