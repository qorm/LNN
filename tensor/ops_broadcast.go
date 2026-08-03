package tensor

import (
	"fmt"
)

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
	s, fs, fresh := broadcastShapeFresh(a, b)
	if fresh {
		return fs[:]
	}
	return s
}

// broadcastShapeFresh is broadcastShape with a freshness flag: fresh is true
// exactly in the outer-product cases, where no operand carries the output
// shape and it may therefore be adopted as a result tensor's Shape without a
// defensive copy. The fresh shape is always rank 2 and comes back in the
// fixed array freshShape (shape is nil then) rather than as a slice: the
// operand-owned cases go through New in broadcastBinary, and New's
// validation panics make their shape argument escape — a freshly allocated
// slice variable sharing that flow would escape to the heap with it (one
// allocation per broadcast op), while the fixed array is a value that stays
// on the caller's stack along the adopting path. The other cases return an
// operand's Shape, which must never be shared with the output.
func broadcastShapeFresh(a, b *Tensor) (shape []int, freshShape [2]int, fresh bool) {
	switch {
	case SameShape(a, b):
		return a.Shape, [2]int{}, false
	case a.IsScalar():
		return b.Shape, [2]int{}, false
	case b.IsScalar():
		return a.Shape, [2]int{}, false
	case a.Dims() == 2 && b.IsRowVec() && a.Cols() == b.Size():
		return a.Shape, [2]int{}, false
	case b.Dims() == 2 && a.IsRowVec() && b.Cols() == a.Size():
		return b.Shape, [2]int{}, false
	case a.Dims() == 2 && a.Cols() == 1 && b.Dims() == 2 && b.Shape[0] == a.Shape[0]:
		return b.Shape, [2]int{}, false
	case b.Dims() == 2 && b.Cols() == 1 && a.Dims() == 2 && a.Shape[0] == b.Shape[0]:
		return a.Shape, [2]int{}, false
	case a.Dims() == 2 && a.Cols() == 1 && b.IsRowVec():
		return nil, [2]int{a.Shape[0], b.Size()}, true
	case b.Dims() == 2 && b.Cols() == 1 && a.IsRowVec():
		return nil, [2]int{b.Shape[0], a.Size()}, true
	default:
		panic(fmt.Sprintf("tensor: shapes %v and %v are not broadcastable", a.Shape, b.Shape))
	}
}

// newAdopting builds a zero-filled tensor from a shape slice no other tensor
// references (see broadcastShapeFresh — today a view over the caller's stack
// array, retained nowhere). The shape is copied inline into the struct via
// useShape: since #12's inline backing this is cheaper than the former heap
// adoption, which needed a separate shape slice allocation that the inline
// path now avoids entirely.
func newAdopting(shape []int) *Tensor {
	t := &Tensor{}
	t.useShape(shape)
	t.Data = make([]float32, t.Size())
	return t
}

func broadcastBinary(a, b *Tensor, f func(x, y float32) float32) *Tensor {
	shape, fs, fresh := broadcastShapeFresh(a, b)
	if len(shape) == 1 {
		// The 1D output promotion lifts the result to [1, n]: the lifted
		// shape is fresh (no operand carries it), so it takes the adopting
		// path, carried in the fixed array like the outer-product cases.
		fs, fresh = [2]int{1, shape[0]}, true
		shape = nil
	}
	if len(shape) == 0 && !fresh { // scalar x scalar
		return FromData([]float32{f(a.Data[0], b.Data[0])}, 1)
	}
	// A fresh shape is adopted directly as the output's Shape (copied into
	// the inline backing); an operand-owned shape goes through New, which
	// copies it. The two shapes deliberately live in separate variables:
	// New's validation flow makes its argument escape, and the fresh array
	// must not share that fate (see broadcastShapeFresh).
	var out *Tensor
	var rows, cols int
	if fresh {
		out = newAdopting(fs[:])
		rows, cols = fs[0], fs[1]
	} else {
		out = New(shape...)
		rows, cols = shape[0], shape[1]
	}
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
