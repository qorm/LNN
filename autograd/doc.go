// Package autograd implements a small dynamic computation-graph engine with
// reverse-mode automatic differentiation. It sits on top of tensor (every
// value is a *tensor.Tensor) and underneath nn (models are graphs of
// Variables).
//
// # The graph model
//
// A Variable is a graph node: a tensor value (Data), an accumulated
// gradient (Grad), plus unexported links to its parents and a backward
// closure that knows how to push gradient into them. Every operation (Add,
// MatMul, Tanh, ...) runs the forward computation eagerly and records its
// closure on the output node; there is no tape or session object. Leaf
// nodes — created with Var, New or Const — have no parents and typically
// are parameters or inputs. Var and Const are the same function; Const is
// an alias that documents intent at the call site (gradients still flow
// into it if it sits inside the graph — just ignore them).
//
// Backward walks the graph from its receiver in reverse topological order
// and runs each closure, accumulating gradients into the leaves. The whole
// graph stays resident until Backward runs: every intermediate tensor is
// kept alive by its node, so peak memory grows with the number of ops in a
// step.
//
// # Gradient semantics
//
//   - Gradients accumulate into leaf variables across Backward calls — and
//     across distinct graphs that share a leaf. Call ZeroGrad on each
//     parameter before the backward pass of a new iteration: the standard
//     loop is zero-grad, forward, backward, update.
//   - Convention: one Backward per graph; build a fresh graph each
//     iteration. Calling Backward twice on the same graph is defined but
//     almost never intended: leaves receive exactly one more full backward
//     pass, so two calls yield precisely twice the gradient (three calls,
//     three times, and so on). Intermediate (non-leaf) gradients are
//     transient: when a traversal completes, Backward clears the Grad of
//     every non-leaf node except the receiver itself, which is what keeps
//     reruns linear rather than super-linear.
//   - Backward requires a scalar receiver (the usual loss). To
//     differentiate from a non-scalar node, seed its Grad manually before
//     calling Backward; the seeded tensor then plays the role of
//     dL/d(output). Calling Backward on an unseeded non-scalar panics.
//   - Gradient accumulation is shape-strict: an incoming gradient whose
//     shape differs from the accumulated one panics with both shapes in
//     the message, even at equal element counts (e.g. [1, 6] vs [2, 3]),
//     so upstream shape bugs surface immediately instead of silently
//     adding misaligned buffers.
//
// # Broadcasting backwards
//
// Add, Sub and Hadamard broadcast in the forward direction (see the tensor
// package for the enumerated rules). Their backward passes reduce the
// output gradient back to each operand's shape with tensor.SumToShape:
// identity for equal shapes, total sum for scalar targets, column sums for
// row-vector targets, row sums for column-vector targets.
//
// # Concurrency
//
// LNN is single-threaded by design. Backward mutates the leaf Grad buffers
// without synchronization; running it concurrently on variables that share
// parameters is a data race that loses or corrupts gradients (verified
// under go test -race). Never share a Variable, Tensor, or computation
// graph across goroutines — give each goroutine its own instances.
//
// Minimizing (x-3)^2 with plain gradient descent:
//
//	x := autograd.New([]float32{0}, 1)
//	for i := 0; i < 200; i++ {
//		d := autograd.Sub(x, autograd.Const(tensor.FromData([]float32{3}, 1)))
//		loss := autograd.Pow(d, 2)
//		x.ZeroGrad()
//		loss.Backward()
//		x.Data.Data[0] -= 0.1 * x.Grad.Data[0]
//	}
//	// x.Data.Data[0] is now 3 within float32 precision
package autograd
