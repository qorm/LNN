// Package optimizer provides small, explicit parameter-update rules over
// github.com/qorm/LNN/autograd: plain SGD, classical heavy-ball momentum, and Adam. Each
// optimizer packages exactly the hand-rolled update loop that
// doc/training.md builds by hand — same float32 arithmetic, same in-place
// writes to p.Data — so a training loop's update phase becomes one
// auditable method call instead of five lines per parameter.
//
// # The contract
//
// Callers still own the loop; every iteration is the four phases of
// doc/training.md with Step replacing phase 4:
//
//	params := nn.ParametersOf(model...)
//	opt := optimizer.NewSGD(0.05)
//	for ... {
//		for _, p := range params {
//			p.ZeroGrad() // grads accumulate; always reset
//		}
//		loss := ...      // build a fresh graph
//		loss.Backward()  // one Backward per graph
//		opt.Step(params) // update in place
//	}
//
// Step never calls ZeroGrad: leaf gradients accumulate across Backward
// calls by design (see autograd.Variable), and when to reset them is the
// caller's contract — every iteration for plain training, every N
// iterations for gradient accumulation. Step skips parameters whose Grad
// is nil (a parameter unused in the last graph keeps its Data and, for
// stateful optimizers, its state untouched) and assumes p.Grad has the
// same shape as p.Data, which autograd's addGrad guarantees.
//
// # Hyperparameters and state
//
// Every hyperparameter is an exported field; read and write it directly —
// adjusting the learning rate mid-training is a supported pattern.
// Constructors validate their arguments and panic with the offending
// value; Step trusts field values as written, just as a hand-rolled loop
// trusts its lr constant.
//
// Momentum and Adam keep per-parameter state keyed by *autograd.Variable
// pointer identity: the same variable stepped repeatedly accumulates
// velocity or moments, and distinct variables never share state. The
// state maps pin every variable they have ever seen (map keys are strong
// references), so construct a fresh optimizer when a model is discarded.
// Re-pointing a variable's Data at a new tensor of the same size keeps
// its state (documented and tested); resizing a parameter in place
// panics rather than silently corrupting the step.
//
// Aliased variables couple. State is keyed by pointer identity, not by
// the underlying storage: two Variables constructed over the same Tensor
// (sharing one Data slice) are distinct map keys but one buffer, so
// stepping both applies each update to the shared storage — with SGD at
// LR=0.1 and unit gradients, a single aliased pair steps the value by
// 2*0.1, not 0.1. Treat aliased variables as one parameter and step it
// once.
//
// # Numerics
//
// float32 everywhere, per the library convention: updates and all
// optimizer state (velocities, Adam's moments) are float32. Adam's
// update is self-normalizing — mhat/sqrt(vhat) stays bounded near ±1 —
// so no wide-magnitude accumulation ever forms, and the float64 trick
// used for the global gradient norm in doc/training.md's clipping
// section does not apply here.
//
// # Concurrency
//
// lnn is single-threaded by design: like Backward, Step mutates
// parameter Data (and optimizer state) without synchronization. Never
// share an optimizer or its parameters across goroutines.
package optimizer
