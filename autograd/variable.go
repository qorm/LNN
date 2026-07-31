package autograd

import (
	"fmt"

	"github.com/qorm/LNN/tensor"
)

// Variable is a node in the computation graph: a tensor value plus its
// accumulated gradient and the backward step that propagates it.
//
// Backward steps are dispatched on the op kind tag instead of a per-node
// closure: a closure capture allocated a heap object per graph node (one of
// the largest allocation sources in deep unrolled graphs), while the tag
// adds a few bytes to this struct. The payload fields (scalar, from/to,
// aux, idx) carry exactly the constants the former closures captured.
type Variable struct {
	// Data is the node's current value. For leaves it is the tensor the
	// node was constructed over (not copied by Var/Const — see Var);
	// for op nodes it is the eagerly computed forward result.
	Data *tensor.Tensor
	// Grad accumulates the gradient of a differentiated output with
	// respect to this node. It is nil until the first Backward (or a
	// manual seed) contributes to it, and Backward keeps it on leaf
	// nodes only — see Backward for the accumulation and clearing
	// semantics. ZeroGrad resets it to nil.
	Grad *tensor.Tensor

	parents  []*Variable
	kind     opKind
	scalar   float32        // Scale factor / Pow exponent
	from, to int            // SliceCol column range / SliceRow row index
	aux      *tensor.Tensor // Div's captured inverse of the denominator
	idx      []int          // GatherRows indices (copied at construction)
}

// Var wraps a tensor as a graph leaf (e.g. a parameter or an input). It
// does not copy t: the leaf aliases the caller's tensor, so later writes
// to t.Data change the leaf's value (the standard pattern for in-place
// parameter updates). Grad starts out nil.
func Var(t *tensor.Tensor) *Variable {
	return &Variable{Data: t}
}

// New builds a leaf Variable from raw data and a shape, copying data
// (via tensor.FromData): the returned leaf aliases nothing. Panics if
// the shape does not match len(data), if any dimension is negative, or
// if the element count overflows int64 — exactly the tensor.FromData
// contract.
func New(data []float32, shape ...int) *Variable {
	return Var(tensor.FromData(data, shape...))
}

// Const is an alias for Var, used at call sites to clarify that the value is
// a constant input rather than a trainable parameter. Gradients still flow
// into it if it sits inside the graph; simply ignore them.
func Const(t *tensor.Tensor) *Variable { return Var(t) }

// newOp creates the output Variable of an operation and records its parents
// and op kind; runBackward dispatches the gradient propagation on the kind.
func newOp(data *tensor.Tensor, parents []*Variable, kind opKind) *Variable {
	return &Variable{Data: data, parents: parents, kind: kind}
}

// addGrad accumulates g into the variable's gradient buffer. It panics if the
// incoming gradient's shape differs from the accumulated one: two tensors can
// have the same element count yet incompatible layouts (e.g. [1, 6] vs [2,
// 3]), and silently adding them elementwise would hide upstream shape bugs.
//
// On the first contribution addGrad takes ownership of g without cloning:
// every backward closure passes either a freshly allocated tensor or a buffer
// it dedicates to this call (see the per-op audit in ops.go), so no other
// reference can observe later accumulation into it. Backward protects the
// root seed the same way, handing the traversal a private buffer.
func (v *Variable) addGrad(g *tensor.Tensor) {
	if v.Grad == nil {
		v.Grad = g
		return
	}
	if !tensor.SameShape(v.Grad, g) {
		panic(fmt.Sprintf("autograd: gradient shape mismatch: accumulated %v vs incoming %v", v.Grad.Shape, g.Shape))
	}
	for i := range v.Grad.Data {
		v.Grad.Data[i] += g.Data[i]
	}
}

// ZeroGrad clears the accumulated gradient.
func (v *Variable) ZeroGrad() { v.Grad = nil }

// Backward runs reverse-mode differentiation from v. v must be a scalar (the
// usual case for a loss) unless its Grad has been seeded manually.
//
// Gradients accumulate into leaf variables across Backward calls (use
// ZeroGrad to reset them). Intermediate (non-leaf) gradients, in contrast,
// are transient: after the traversal completes, the Grad of every non-leaf
// node other than v itself is cleared. This makes repeated Backward calls on
// the same graph accumulate linearly into leaves; leaving stale intermediate
// gradients in place would make each rerun propagate through already-seeded
// buffers and accumulate super-linearly.
func (v *Variable) Backward() {
	seed := v.Grad
	if seed == nil {
		if !v.Data.IsScalar() {
			panic(fmt.Sprintf("autograd.Backward: non-scalar output of shape %v needs a seeded Grad", v.Data.Shape))
		}
		v.Grad = v.Data.OnesLike()
	} else {
		// Propagate from a private copy. Backward closures may transfer
		// ownership of the root's gradient buffer to a leaf (addGrad's
		// clone-free first contribution), which would alias — and then
		// corrupt — the caller-visible seed during later accumulation.
		v.Grad = seed.Clone()
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
		topo[i].runBackward()
	}
	for _, n := range topo {
		if n != v && len(n.parents) > 0 {
			n.Grad = nil
		}
	}
	if seed == nil {
		// The traversal may have handed the auto-seeded buffer to a leaf.
		// Leave a fresh pristine seed behind so repeated Backward calls
		// accumulate linearly (each run propagates ones again) and v.Grad
		// stays inspectable, exactly as when addGrad cloned on arrival.
		v.Grad = v.Data.OnesLike()
	} else {
		// Restore the caller's (never-propagated, never-mutated) seed.
		v.Grad = seed
	}
}

// Value returns the scalar value of a size-1 variable (the usual way to
// read a loss). Panics if the variable's Data does not hold exactly one
// element.
func (v *Variable) Value() float32 { return v.Data.Scalar() }
