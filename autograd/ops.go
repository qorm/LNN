package autograd

import (
	"fmt"

	"github.com/qorm/LNN/tensor"
)

// MatMul differentiably multiplies two 2D variables: a of shape [m, k]
// times b of shape [k, n] yields a node of shape [m, n]. Panics exactly
// when tensor.MatMul does — if either operand is not 2D or the inner
// dimensions disagree. The backward propagates gradients to both
// operands without materializing transposes.
func MatMul(a, b *Variable) *Variable {
	return newOp(tensor.MatMul(a.Data, b.Data), []*Variable{a, b}, opMatMul)
}

// Add differentiably computes a + b with the tensor package's
// enumerated broadcasting rules: the forward panics on a
// non-broadcastable pair ("not broadcastable"), and two 1D operands
// yield a [1, n] node (the 1D output promotion). The backward reduces
// the output gradient to each operand's own shape with
// tensor.SumToShape, so leaf gradients match leaf shapes even after
// broadcasting.
func Add(a, b *Variable) *Variable {
	return newOp(tensor.Add(a.Data, b.Data), []*Variable{a, b}, opAdd)
}

// Sub differentiably computes a - b with the tensor package's
// enumerated broadcasting rules (forward panics on a non-broadcastable
// pair; two 1D operands yield [1, n]). The backward reduces the output
// gradient to each operand's shape, negated for b, via tensor.SumToShape.
func Sub(a, b *Variable) *Variable {
	return newOp(tensor.Sub(a.Data, b.Data), []*Variable{a, b}, opSub)
}

// Hadamard differentiably computes elementwise a * b with the tensor
// package's enumerated broadcasting rules (forward panics on a
// non-broadcastable pair; [m, 1] against a row vector is the outer
// product; two 1D operands yield [1, n]). The backward multiplies the
// output gradient by the other operand, then reduces it to each
// operand's shape via tensor.SumToShape.
func Hadamard(a, b *Variable) *Variable {
	return newOp(tensor.Hadamard(a.Data, b.Data), []*Variable{a, b}, opHadamard)
}

// Scale multiplies every element of a by the constant s, any shape.
// The backward scales the incoming gradient by the same s.
func Scale(a *Variable, s float32) *Variable {
	out := newOp(tensor.Scale(a.Data, s), []*Variable{a}, opScale)
	out.scalar = s
	return out
}

// Neg negates every element of a, any shape. It is Scale(a, -1).
func Neg(a *Variable) *Variable { return Scale(a, -1) }

// Tanh applies tanh elementwise, any shape. The backward is the fused
// g ⊙ (1 − tanh²(x)).
func Tanh(a *Variable) *Variable {
	return newOp(tensor.Tanh(a.Data), []*Variable{a}, opTanh)
}

// Sigmoid applies the logistic sigmoid elementwise, any shape. The
// backward is the fused g ⊙ σ(x) ⊙ (1 − σ(x)).
func Sigmoid(a *Variable) *Variable {
	return newOp(tensor.Sigmoid(a.Data), []*Variable{a}, opSigmoid)
}

// SigmoidHadamard differentiably computes Hadamard(Sigmoid(z), w) as a single
// fused node. The forward runs the very same two tensor operations the
// composition ran (sigmoid, then the elementwise product), so shapes,
// broadcasting and values are bit-identical; the sigmoid buffer is kept on
// the node (aux) so the backward reuses it instead of recomputing. The
// backward propagates dz = g⊙w⊙s⊙(1−s) in one fused loop and dw = g⊙s
// through the same fused reduction the Hadamard backward used, where the
// composition recorded two op nodes and ran Sigmoid's backward on top of
// Hadamard's (materializing the intermediate g⊙s gradient buffer the fusion
// avoids). Broadcasting and shape semantics follow Hadamard exactly:
// the forward panics when sigmoid(z) and w are not broadcastable. The
// node exists to keep the LTC/CfC inner loop cheap (one node, one fused
// backward, the sigmoid buffer reused); plain code can compose Sigmoid
// and Hadamard instead.
func SigmoidHadamard(z, w *Variable) *Variable {
	s := tensor.Sigmoid(z.Data)
	out := newOp(tensor.Hadamard(s, w.Data), []*Variable{z, w}, opSigmoidHadamard)
	out.aux = s
	return out
}

// Exp applies exp elementwise, any shape. The backward multiplies the
// incoming gradient by the (stored) forward output.
func Exp(a *Variable) *Variable {
	return newOp(tensor.Exp(a.Data), []*Variable{a}, opExp)
}

// Log applies natural log elementwise, any shape; the domain is not
// checked (non-positive elements give NaN/-Inf forward and backward,
// as float32 arithmetic dictates). The backward is g ⊙ (1/x).
func Log(a *Variable) *Variable {
	return newOp(tensor.Log(a.Data), []*Variable{a}, opLog)
}

// Pow raises every element of a to the constant power p, any shape.
// The backward is g ⊙ p·x^(p−1); for p == 0 it is exactly zero
// everywhere (evaluated directly, not as 0·x^-1, which would be NaN at
// x == 0).
func Pow(a *Variable, p float32) *Variable {
	out := newOp(tensor.Pow(a.Data, p), []*Variable{a}, opPow)
	out.scalar = p
	return out
}

// Softplus applies log(1 + e^x) elementwise, any shape, stably (see
// tensor.Softplus). The backward is g ⊙ σ(x).
func Softplus(a *Variable) *Variable {
	return newOp(tensor.Softplus(a.Data), []*Variable{a}, opSoftplus)
}

// Abs applies |x| elementwise, any shape. It is not differentiable at
// 0; the backward takes the gradient there as 0 (g ⊙ sign(x) with
// sign(0) = 0).
func Abs(a *Variable) *Variable {
	return newOp(tensor.Apply(a.Data, func(x float32) float32 {
		if x < 0 {
			return -x
		}
		return x
	}), []*Variable{a}, opAbs)
}

// Relu applies max(0, x) elementwise. The gradient at 0 is taken as 0.
func Relu(a *Variable) *Variable {
	return newOp(tensor.Apply(a.Data, func(x float32) float32 {
		if x > 0 {
			return x
		}
		return 0
	}), []*Variable{a}, opRelu)
}

// Div differentiably computes a / b elementwise with broadcasting; b must be
// nonzero (b == 0 yields +/-Inf in the forward pass, as float32 division does).
//
// Div is a single closed-form graph node. The previous implementation
// composed Hadamard(a, Pow(b, -1)), which recorded two op nodes and ran
// Pow's backward on top of Hadamard's. The forward deliberately reuses the
// exact tensor-level computation of that composition (a ⊙ pow(b, -1)), so
// shapes, broadcasting and values are bit-identical. The backward follows
// the quotient rule, da = g/b and db = -g·a/b²; because b⁻² is constant
// along every axis b was broadcast over, db may reduce g·a to b's shape
// first and only then scale by -b⁻² — which equals the closed form and also
// keeps gradients bit-identical to the legacy two-node chain.
func Div(a, b *Variable) *Variable {
	inv := tensor.Pow(b.Data, -1)
	out := newOp(tensor.Hadamard(a.Data, inv), []*Variable{a, b}, opDiv)
	out.aux = inv
	return out
}

// ConcatCol concatenates 2D variables along the column axis: inputs of
// shapes [m, n1], [m, n2], ... yield a node of shape [m, n1+n2+...].
// Panics if called with no inputs, or per tensor.ConcatCol if any input
// is not 2D or the row counts differ. The backward slices each input's
// gradient back out of the output gradient.
func ConcatCol(vs ...*Variable) *Variable {
	if len(vs) == 0 {
		panic("autograd.ConcatCol: no inputs")
	}
	ts := make([]*tensor.Tensor, len(vs))
	for i, v := range vs {
		ts[i] = v.Data
	}
	return newOp(tensor.ConcatCol(ts...), vs, opConcatCol)
}

// SliceCol differentiably extracts columns [from, to) of a 2D variable
// as a node of shape [m, to-from]. Panics per tensor.SliceCol if a is
// not 2D or the range is invalid or empty. The backward writes the
// gradient back into the corresponding columns of a zero-padded buffer
// the size of a.
func SliceCol(a *Variable, from, to int) *Variable {
	out := newOp(tensor.SliceCol(a.Data, from, to), []*Variable{a}, opSliceCol)
	out.from, out.to = from, to
	return out
}

// Col differentiably extracts column i of a 2D variable as a node of
// shape [m, 1]. It is SliceCol(a, i, i+1) and panics under the same
// conditions (a not 2D, i out of range).
func Col(a *Variable, i int) *Variable { return SliceCol(a, i, i+1) }

// SliceRow differentiably extracts row i of a 2D variable as a node of
// shape [1, n]. Panics per tensor.SliceRow if a is not 2D or i is
// outside [0, a.Data.Rows()). The backward writes the gradient back
// into row i of a zero-padded buffer the size of a.
func SliceRow(a *Variable, i int) *Variable {
	out := newOp(tensor.SliceRow(a.Data, i), []*Variable{a}, opSliceRow)
	out.from = i
	return out
}

// SumAll sums every element of a (any shape) into a scalar node of
// shape [1]. The backward broadcasts the scalar gradient to a's shape.
func SumAll(a *Variable) *Variable {
	return newOp(tensor.SumAll(a.Data), []*Variable{a}, opSumAll)
}

// MeanAll averages every element of a (any shape) into a scalar node
// of shape [1]. Panics on an empty tensor, like tensor.MeanAll. The
// backward broadcasts g/size to a's shape.
func MeanAll(a *Variable) *Variable {
	return newOp(tensor.MeanAll(a.Data), []*Variable{a}, opMeanAll)
}

// GatherRows picks one element per row: out[i] = a[i, idx[i]]. a must be 2D
// and len(idx) must equal its row count. The output is 1D with shape [rows].
// The idx slice is copied on entry, so the caller may freely reuse or mutate
// it between the forward pass and Backward without corrupting gradients.
func GatherRows(a *Variable, idx []int) *Variable {
	if a.Data.Dims() != 2 || len(idx) != a.Data.Rows() {
		panic(fmt.Sprintf("autograd.GatherRows: shape %v vs %d indices", a.Data.Shape, len(idx)))
	}
	idx = append([]int(nil), idx...)
	m, n := a.Data.Rows(), a.Data.Cols()
	data := make([]float32, m)
	for i, j := range idx {
		if j < 0 || j >= n {
			panic(fmt.Sprintf("autograd.GatherRows: index %d out of bounds for %d columns", j, n))
		}
		data[i] = a.Data.Data[i*n+j]
	}
	out := newOp(tensor.FromData(data, m), []*Variable{a}, opGatherRows)
	out.idx = idx
	return out
}

// FusedOp creates an op node with a caller-supplied backward step: the
// forward value is data (computed by the caller, exactly as for the fixed
// op constructors), and backward is invoked by runBackward with the node
// after its Grad has fully accumulated. The backward is responsible for
// every addGrad call to the parents — nothing is dispatched automatically.
//
// This is the integration point for hand-written fused kernels (the LTC's
// ODE unfold loop, nn/ltc_fused.go): one node replaces a whole subgraph,
// and the closure must replicate the replaced subgraph's backward
// accumulation order contribution for contribution, bit for bit. It exists
// because a fused kernel's backward cannot be expressed by composing the
// fixed ops without re-materializing the very graph the fusion removes.
// Unlike the fixed ops, the closure allocates one heap object per node —
// negligible at the intended rate of one node per RNN step (against the
// hundreds of nodes the fused subgraph used to record per step). Panics
// if backward is nil.
func FusedOp(data *tensor.Tensor, parents []*Variable, backward func(v *Variable)) *Variable {
	if backward == nil {
		panic("autograd.FusedOp: nil backward")
	}
	out := newOp(data, parents, opFused)
	out.fused = backward
	return out
}

// LogSoftmaxRows applies the numerically stable log-softmax to each
// row of a 2D variable, yielding a node of the same shape [m, n].
// Panics per tensor.LogSoftmaxRows if a is not 2D. The backward is the
// fused g − softmax(row) ⊙ rowsum(g); a manually seeded Grad whose
// shape differs from the node's takes the legacy (shape-strict)
// reduction path and may panic on 1D seeds, exactly as the pre-fusion
// composition did.
func LogSoftmaxRows(a *Variable) *Variable {
	return newOp(tensor.LogSoftmaxRows(a.Data), []*Variable{a}, opLogSoftmaxRows)
}
