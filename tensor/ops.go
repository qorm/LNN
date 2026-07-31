package tensor

import (
	"fmt"
	"math"
)

// MatMul multiplies two 2D tensors: a of shape [m, k] times b of shape
// [k, n] yields a fresh [m, n] tensor. It is matrix multiplication only —
// no batched or vector products. Panics if either operand is not 2D, or
// if the inner dimensions disagree (a.Cols() != b.Rows()).
func MatMul(a, b *Tensor) *Tensor {
	if a.Dims() != 2 || b.Dims() != 2 || a.Cols() != b.Rows() {
		panic(fmt.Sprintf("tensor.MatMul: shapes %v and %v are incompatible", a.Shape, b.Shape))
	}
	m, k, n := a.Rows(), a.Cols(), b.Cols()
	out := New(m, n)
	for i := 0; i < m; i++ {
		arow := a.Data[i*k : (i+1)*k]
		orow := out.Data[i*n : (i+1)*n]
		for kk := 0; kk < k; kk++ {
			av := arow[kk]
			if av == 0 {
				continue
			}
			brow := b.Data[kk*n : (kk+1)*n]
			for j := 0; j < n; j++ {
				orow[j] += av * brow[j]
			}
		}
	}
	return out
}

// Transpose returns the transpose of a 2D tensor: [m, n] -> [n, m], in a
// fresh buffer. Panics if a is not 2D.
func Transpose(a *Tensor) *Tensor {
	m, n := a.Rows(), a.Cols()
	out := New(n, m)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			out.Data[j*m+i] = a.Data[i*n+j]
		}
	}
	return out
}

// MatMulTransA multiplies the transpose of a 2D tensor by another: a is
// [m, k] and b is [m, n], and the result is aᵀ·b with shape [k, n]. It
// evaluates exactly the same products and accumulation order as
// MatMul(Transpose(a), b) — the transposed entries are read in place in the
// same zero-skipping loop — without allocating the transpose. Used by the
// autograd backward of MatMul. Panics if either operand is not 2D, or if
// the row counts disagree (a.Rows() != b.Rows()).
func MatMulTransA(a, b *Tensor) *Tensor {
	if a.Dims() != 2 || b.Dims() != 2 || a.Rows() != b.Rows() {
		panic(fmt.Sprintf("tensor.MatMulTransA: shapes %v and %v are incompatible", a.Shape, b.Shape))
	}
	m, k, n := a.Rows(), a.Cols(), b.Cols()
	out := New(k, n)
	for i := 0; i < k; i++ {
		orow := out.Data[i*n : (i+1)*n]
		for kk := 0; kk < m; kk++ {
			av := a.Data[kk*k+i]
			if av == 0 {
				continue
			}
			brow := b.Data[kk*n : (kk+1)*n]
			for j := 0; j < n; j++ {
				orow[j] += av * brow[j]
			}
		}
	}
	return out
}

// MatMulTransB multiplies a 2D tensor by the transpose of another: a is
// [m, k] and b is [n, k], and the result is a·bᵀ with shape [m, n]. It
// evaluates exactly the same products and accumulation order as
// MatMul(a, Transpose(b)) — the transposed entries are read in place in the
// same zero-skipping loop — without allocating the transpose. Used by the
// autograd backward of MatMul. Panics if either operand is not 2D, or if
// the column counts disagree (a.Cols() != b.Cols()).
func MatMulTransB(a, b *Tensor) *Tensor {
	if a.Dims() != 2 || b.Dims() != 2 || a.Cols() != b.Cols() {
		panic(fmt.Sprintf("tensor.MatMulTransB: shapes %v and %v are incompatible", a.Shape, b.Shape))
	}
	m, k, n := a.Rows(), a.Cols(), b.Rows()
	out := New(m, n)
	for i := 0; i < m; i++ {
		arow := a.Data[i*k : (i+1)*k]
		orow := out.Data[i*n : (i+1)*n]
		for kk := 0; kk < k; kk++ {
			av := arow[kk]
			if av == 0 {
				continue
			}
			for j := 0; j < n; j++ {
				orow[j] += av * b.Data[j*k+kk]
			}
		}
	}
	return out
}

// Broadcast participation modes. For output row i, an operand's element at
// column j lives at Data[base+stride*j] with (base, stride) determined by its
// mode: scalar (0, 0), column vector (i, 0), full matrix (i*cols, 1), row
// vector (0, 1). The stride description replaces the former per-call closure
// pair — pure data movement, so evaluation order and values are unchanged.
// The classification order mirrors the historical dispatch exactly.
const (
	bcastScalar = iota
	bcastCol
	bcastFull
	bcastRow
)

func bcastMode(t *Tensor) int {
	switch {
	case t.IsScalar():
		return bcastScalar
	case t.Dims() == 2 && t.Shape[1] == 1 && t.Shape[0] > 1: // column vector [m, 1]
		return bcastCol
	case t.Dims() == 2 && t.Shape[0] > 1: // full 2D, same shape as output
		return bcastFull
	case t.IsRowVec(): // [n] or [1, n]
		return bcastRow
	default:
		panic(fmt.Sprintf("tensor: cannot broadcast shape %v", t.Shape))
	}
}

// broadcastShape resolves the output shape of a binary op, following the
// library's limited broadcasting: same shape, scalar, row-vector against a
// 2D tensor, or size-1 dimensions (a [m, 1] column vector broadcasts against
// [m, n], and [m, 1] x [1, n] produces the outer product [m, n]).
func broadcastShape(a, b *Tensor) []int {
	s, _ := broadcastShapeFresh(a, b)
	return s
}

// broadcastShapeFresh is broadcastShape with a freshness flag: fresh is true
// exactly when the returned slice was newly allocated (the outer-product
// cases) and may therefore be adopted as a result tensor's Shape without a
// defensive copy. The other cases return an operand's Shape, which must
// never be shared with the output.
func broadcastShapeFresh(a, b *Tensor) (shape []int, fresh bool) {
	switch {
	case SameShape(a, b):
		return a.Shape, false
	case a.IsScalar():
		return b.Shape, false
	case b.IsScalar():
		return a.Shape, false
	case a.Dims() == 2 && b.IsRowVec() && a.Cols() == b.Size():
		return a.Shape, false
	case b.Dims() == 2 && a.IsRowVec() && b.Cols() == a.Size():
		return b.Shape, false
	case a.Dims() == 2 && a.Cols() == 1 && b.Dims() == 2 && b.Shape[0] == a.Shape[0]:
		return b.Shape, false
	case b.Dims() == 2 && b.Cols() == 1 && a.Dims() == 2 && a.Shape[0] == b.Shape[0]:
		return a.Shape, false
	case a.Dims() == 2 && a.Cols() == 1 && b.IsRowVec():
		return []int{a.Shape[0], b.Size()}, true
	case b.Dims() == 2 && b.Cols() == 1 && a.IsRowVec():
		return []int{b.Shape[0], a.Size()}, true
	default:
		panic(fmt.Sprintf("tensor: shapes %v and %v are not broadcastable", a.Shape, b.Shape))
	}
}

// newAdopting builds a zero-filled tensor from a freshly allocated shape slice
// that no other tensor references (see broadcastShapeFresh). The shape is
// copied inline into the struct via useShape: since #12's inline backing this
// is cheaper than the former heap adoption, which needed a separate shape
// slice allocation that the inline path now avoids entirely.
func newAdopting(shape []int) *Tensor {
	t := &Tensor{}
	t.useShape(shape)
	t.Data = make([]float32, t.Size())
	return t
}

func broadcastBinary(a, b *Tensor, f func(x, y float32) float32) *Tensor {
	shape, fresh := broadcastShapeFresh(a, b)
	if len(shape) == 1 {
		shape, fresh = []int{1, shape[0]}, true
	}
	if len(shape) == 0 { // scalar x scalar
		return FromData([]float32{f(a.Data[0], b.Data[0])}, 1)
	}
	// A fresh shape slice is adopted directly as the output's Shape; an
	// operand-owned shape goes through New, which copies it.
	var out *Tensor
	if fresh {
		out = newAdopting(shape)
	} else {
		out = New(shape...)
	}
	rows, cols := shape[0], shape[1]
	ad, bd, od := a.Data, b.Data, out.Data
	if SameShape(a, b) && a.Dims() == 2 {
		// Fast path: both operands already have the output layout. The
		// general loop below would index i*cols+j in exactly this order,
		// so the flat walk evaluates f on the identical pair sequence.
		for i := range od {
			od[i] = f(ad[i], bd[i])
		}
		return out
	}
	ma, mb := bcastMode(a), bcastMode(b)
	for i := 0; i < rows; i++ {
		orow := od[i*cols : (i+1)*cols]
		ab, as := bcastRowAccess(ma, i, cols)
		bb, bs := bcastRowAccess(mb, i, cols)
		switch {
		case as == 1 && bs == 1:
			for j := range orow {
				orow[j] = f(ad[ab+j], bd[bb+j])
			}
		case as == 1: // b constant across the row
			bv := bd[bb]
			for j := range orow {
				orow[j] = f(ad[ab+j], bv)
			}
		case bs == 1: // a constant across the row
			av := ad[ab]
			for j := range orow {
				orow[j] = f(av, bd[bb+j])
			}
		default: // both constant across the row
			// cols == 1 always holds in this arm: reaching it requires both
			// operands to broadcast with stride zero (bcastScalar or bcastCol
			// modes), and under those modes every broadcastShapeFresh arm
			// resolves to a single-column shape — an operand shape [m, 1], a
			// size-1 shape (every dimension 1, at any rank), or the [1, 1]
			// lift of a 1D scalar pair; [1, 1] x [1, 1] itself is intercepted
			// by the SameShape fast path above, and any rank-3+ non-scalar
			// operand panics in bcastMode before the loop. The j >= 1 fill
			// that once stood here could never execute and was removed.
			orow[0] = f(ad[ab], bd[bb])
		}
	}
	return out
}

// bcastRowAccess returns the (base, stride) describing where output row i
// reads its operand values: Data[base+stride*j] for column j.
func bcastRowAccess(mode, i, cols int) (base, stride int) {
	switch mode {
	case bcastScalar:
		return 0, 0
	case bcastCol:
		return i, 0
	case bcastFull:
		return i * cols, 1
	default: // bcastRow
		return 0, 1
	}
}

// Add computes a + b with broadcasting, returning a fresh tensor. The
// operand shapes must match one of the enumerated broadcasting rules in
// the package doc (and doc/shapes-and-broadcasting.md): identical shapes,
// scalar against anything, row/column vector against a matrix, or the
// [m, 1] x row-vector outer product. Panics on any other combination
// ("not broadcastable"). Two 1D operands yield a [1, n] result (the 1D
// output promotion); in particular [1] + [1] yields [1, 1], and only
// rank-0 ([], from tensor.New()) operands produce a [1] result.
func Add(a, b *Tensor) *Tensor {
	return broadcastBinary(a, b, func(x, y float32) float32 { return x + y })
}

// Sub computes a - b with broadcasting, returning a fresh tensor. The
// operands follow the same enumerated broadcasting rules as Add (package
// doc, doc/shapes-and-broadcasting.md) and panic on any other combination
// ("not broadcastable"), with the same [1, n] promotion for two 1D
// operands.
func Sub(a, b *Tensor) *Tensor {
	return broadcastBinary(a, b, func(x, y float32) float32 { return x - y })
}

// Hadamard computes elementwise a * b with broadcasting, returning a
// fresh tensor. The operands follow the same enumerated broadcasting
// rules as Add (package doc, doc/shapes-and-broadcasting.md) and panic on
// any other combination ("not broadcastable"); in particular [m, 1]
// against a row vector produces the outer product [m, n], and two 1D
// operands yield a [1, n] result.
func Hadamard(a, b *Tensor) *Tensor {
	return broadcastBinary(a, b, func(x, y float32) float32 { return x * y })
}

// Scale multiplies every element of a by the constant s, returning a
// fresh tensor of the same shape (any rank). It does not modify a.
func Scale(a *Tensor, s float32) *Tensor {
	out := a.ZerosLike()
	for i, v := range a.Data {
		out.Data[i] = v * s
	}
	return out
}

// Neg negates every element, returning a fresh tensor of the same shape.
// It is Scale(a, -1).
func Neg(a *Tensor) *Tensor { return Scale(a, -1) }

// Apply maps f over every element of a in flat row-major order,
// returning a fresh tensor of the same shape (any rank). It does not
// modify a; f must not retain or mutate the tensor.
func Apply(a *Tensor, f func(float32) float32) *Tensor {
	out := a.ZerosLike()
	for i, v := range a.Data {
		out.Data[i] = f(v)
	}
	return out
}

func sigmoid(x float32) float32 {
	// Numerically stable logistic sigmoid.
	if x >= 0 {
		return 1 / (1 + float32(math.Exp(float64(-x))))
	}
	e := float32(math.Exp(float64(x)))
	return e / (1 + e)
}

// Tanh applies tanh elementwise, returning a fresh tensor of the same
// shape (any rank).
func Tanh(a *Tensor) *Tensor {
	return Apply(a, func(x float32) float32 { return float32(math.Tanh(float64(x))) })
}

// Sigmoid applies the logistic sigmoid 1/(1+e^-x) elementwise, in a
// numerically stable form, returning a fresh tensor of the same shape
// (any rank).
func Sigmoid(a *Tensor) *Tensor { return Apply(a, sigmoid) }

// Exp applies exp elementwise, returning a fresh tensor of the same
// shape (any rank). Large inputs overflow to +Inf like plain float32
// arithmetic (no domain checking, per the package doc).
func Exp(a *Tensor) *Tensor {
	return Apply(a, func(x float32) float32 { return float32(math.Exp(float64(x))) })
}

// Log applies natural log elementwise, returning a fresh tensor of the
// same shape (any rank). The domain is not checked: log of a negative
// element is NaN and log of zero is -Inf, exactly as float32 arithmetic
// dictates (per the package doc).
func Log(a *Tensor) *Tensor {
	return Apply(a, func(x float32) float32 { return float32(math.Log(float64(x))) })
}

// Pow raises every element of a to the constant power p, returning a
// fresh tensor of the same shape (any rank). The domain is not checked:
// a negative element with a non-integer p yields NaN, as float32
// arithmetic dictates (per the package doc).
func Pow(a *Tensor, p float32) *Tensor {
	return Apply(a, func(x float32) float32 { return float32(math.Pow(float64(x), float64(p))) })
}

// Softplus applies log(1 + e^x) elementwise, returning a fresh tensor of
// the same shape (any rank). It is numerically stable: elements above 20
// return x itself, where log(1 + e^x) rounds to x in float32 anyway, so
// large inputs never overflow through exp.
func Softplus(a *Tensor) *Tensor {
	return Apply(a, func(x float32) float32 {
		if x > 20 {
			return x
		}
		return float32(math.Log1p(math.Exp(float64(x))))
	})
}

// Clip clamps every element of a to [lo, hi], returning a fresh tensor
// of the same shape (any rank). It expects lo <= hi; with lo > hi every
// element maps to one of the two bounds (elements below lo to lo, all
// others to hi), which is almost never what a caller wants.
func Clip(a *Tensor, lo, hi float32) *Tensor {
	return Apply(a, func(x float32) float32 {
		if x < lo {
			return lo
		}
		if x > hi {
			return hi
		}
		return x
	})
}

// ConcatCol concatenates 2D tensors along the column axis: inputs of
// shapes [m, n1], [m, n2], ... yield a fresh [m, n1+n2+...] tensor,
// copies of the inputs laid side by side. Panics if called with no
// tensors, if any input is not 2D, or if the inputs have differing row
// counts.
func ConcatCol(ts ...*Tensor) *Tensor {
	if len(ts) == 0 {
		panic("tensor.ConcatCol: no tensors")
	}
	rows := ts[0].Rows()
	cols := 0
	for _, t := range ts {
		if t.Dims() != 2 || t.Rows() != rows {
			panic(fmt.Sprintf("tensor.ConcatCol: shape %v incompatible with %d rows", t.Shape, rows))
		}
		cols += t.Cols()
	}
	out := New(rows, cols)
	off := 0
	for _, t := range ts {
		for i := 0; i < rows; i++ {
			copy(out.Data[i*cols+off:i*cols+off+t.Cols()], t.Data[i*t.Cols():(i+1)*t.Cols()])
		}
		off += t.Cols()
	}
	return out
}

// SliceCol returns columns [from, to) of a 2D tensor as a fresh [m,
// to-from] copy (no storage is shared with a). Panics if a is not 2D,
// or if the range is invalid or empty: from < 0, to > a.Cols(), or
// from >= to.
func SliceCol(a *Tensor, from, to int) *Tensor {
	if a.Dims() != 2 || from < 0 || to > a.Cols() || from >= to {
		panic(fmt.Sprintf("tensor.SliceCol: invalid range [%d, %d) for shape %v", from, to, a.Shape))
	}
	rows, cols := a.Rows(), to-from
	out := New(rows, cols)
	for i := 0; i < rows; i++ {
		copy(out.Data[i*cols:(i+1)*cols], a.Data[i*a.Cols()+from:i*a.Cols()+to])
	}
	return out
}

// SliceRow returns row i of a 2D tensor as a fresh [1, n] copy (no
// storage is shared with a). Panics if a is not 2D, or if i is outside
// [0, a.Rows()).
func SliceRow(a *Tensor, i int) *Tensor {
	if a.Dims() != 2 || i < 0 || i >= a.Rows() {
		panic(fmt.Sprintf("tensor.SliceRow: invalid row %d for shape %v", i, a.Shape))
	}
	n := a.Cols()
	out := New(1, n)
	copy(out.Data, a.Data[i*n:(i+1)*n])
	return out
}

// SumAll returns the sum of all elements of a (any rank) as a scalar
// tensor of shape [1]. The sum of an empty tensor is 0.
func SumAll(a *Tensor) *Tensor {
	var s float32
	for _, v := range a.Data {
		s += v
	}
	return FromData([]float32{s}, 1)
}

// MeanAll returns the mean of all elements of a (any rank) as a scalar
// tensor of shape [1]. Panics on an empty tensor: the mean of zero
// elements is undefined, and dividing by Size()==0 would silently
// produce NaN.
func MeanAll(a *Tensor) *Tensor {
	if a.Size() == 0 {
		panic(fmt.Sprintf("tensor.MeanAll: mean of empty tensor (shape %v) is undefined", a.Shape))
	}
	return FromData([]float32{SumAll(a).Data[0] / float32(a.Size())}, 1)
}

// SumRows sums over axis 0 of a 2D tensor [m, n], returning the column
// sums as a [1, n] matrix (deliberately kept 2D so the result
// re-broadcasts against [m, n]). Panics if a is not 2D. The reduction
// output shapes are asymmetric (SumRows keeps a [1, n] matrix while
// SumCols drops to 1D [m]); the convention was frozen as of v0.4.0 —
// see doc/shapes-and-broadcasting.md for the rationale and the full
// reduction table.
func SumRows(a *Tensor) *Tensor {
	m, n := a.Rows(), a.Cols()
	out := New(1, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			out.Data[j] += a.Data[i*n+j]
		}
	}
	return out
}

// SumCols sums over axis 1 of a 2D tensor [m, n], returning the row
// sums as a 1D [m] vector. Panics if a is not 2D. The reduction output
// shapes are asymmetric (SumCols drops to 1D [m] while SumRows keeps a
// [1, n] matrix); the convention was frozen as of v0.4.0 — see
// doc/shapes-and-broadcasting.md for the rationale and the full
// reduction table.
func SumCols(a *Tensor) *Tensor {
	m, n := a.Rows(), a.Cols()
	out := New(m)
	for i := 0; i < m; i++ {
		var s float32
		for j := 0; j < n; j++ {
			s += a.Data[i*n+j]
		}
		out.Data[i] = s
	}
	return out
}

// SumToShape reduces a broadcast-produced gradient back to a target
// shape, following the table in doc/shapes-and-broadcasting.md
// ("Backward reductions"). For a gradient of shape [m, n]:
//
//   - target [m, n]: identity (a clone of grad);
//   - a scalar target (any single-element shape — [1], [1, 1], []): the
//     total sum, returned with shape [1];
//   - target [n] or [1, n]: column sums, returned with the target's own
//     layout ([n] or [1, n]);
//   - target [m, 1]: row sums, returned with shape [m, 1];
//   - anything else: panic ("cannot reduce").
//
// The result never aliases grad: the matching-shape arm returns a Clone
// and every reduction arm (SumAll/SumRows/SumCols) allocates a fresh
// buffer. This is what makes autograd leaf gradients always match leaf
// shapes even when the forward pass broadcast them. The owning variant
// that once lived here (SumToShapeTake, which returned grad itself on a
// shape match) moved into the autograd package in v0.4.0 as the
// unexported sumToShapeTake — its only callers were the backward path,
// and keeping an ownership footgun on the public surface bought nothing
// external.
func SumToShape(grad *Tensor, shape []int) *Tensor {
	target := &Tensor{Shape: shape}
	switch {
	case SameShape(grad, target):
		return grad.Clone()
	case target.IsScalar():
		return SumAll(grad)
	case grad.Dims() == 2 && target.IsRowVec() && grad.Cols() == target.Size():
		s := SumRows(grad)
		if len(shape) == 1 {
			s.Reshape(shape[0])
		}
		return s
	case grad.Dims() == 2 && target.Dims() == 2 && target.Shape[1] == 1 && grad.Shape[0] == target.Shape[0]:
		s := SumCols(grad)
		s.Reshape(shape[0], 1)
		return s
	default:
		panic(fmt.Sprintf("tensor.SumToShape: cannot reduce shape %v to %v", grad.Shape, shape))
	}
}

// LogSoftmaxRows applies log-softmax to each row of a 2D tensor [m, n],
// returning a fresh [m, n] tensor, computed in the numerically stable
// max-subtracted form. A tensor with zero columns has no elements per
// row and yields an empty result of the same shape. Panics if a is not
// 2D.
func LogSoftmaxRows(a *Tensor) *Tensor {
	m, n := a.Rows(), a.Cols()
	out := New(m, n)
	if n == 0 {
		return out
	}
	for i := 0; i < m; i++ {
		row := a.Data[i*n : (i+1)*n]
		max := row[0]
		for _, v := range row {
			if v > max {
				max = v
			}
		}
		var sum float64
		for _, v := range row {
			sum += math.Exp(float64(v - max))
		}
		logsum := max + float32(math.Log(sum))
		orow := out.Data[i*n : (i+1)*n]
		for j, v := range row {
			orow[j] = v - logsum
		}
	}
	return out
}

// SoftmaxRows applies softmax to each row of a 2D tensor [m, n],
// returning a fresh [m, n] tensor in the numerically stable
// max-subtracted form. A tensor with zero columns has no elements per
// row and yields an empty result of the same shape. Panics if a is not
// 2D.
func SoftmaxRows(a *Tensor) *Tensor {
	m, n := a.Rows(), a.Cols()
	out := New(m, n)
	if n == 0 {
		return out
	}
	for i := 0; i < m; i++ {
		row := a.Data[i*n : (i+1)*n]
		max := row[0]
		for _, v := range row {
			if v > max {
				max = v
			}
		}
		var sum float32
		orow := out.Data[i*n : (i+1)*n]
		for j, v := range row {
			e := float32(math.Exp(float64(v - max)))
			orow[j] = e
			sum += e
		}
		for j := range orow {
			orow[j] /= sum
		}
	}
	return out
}
