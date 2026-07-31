// Package tensor provides a minimal dense float32 n-dimensional tensor type
// with row-major layout, plus the small set of numeric operations the LNN
// library needs. It is the numeric kernel of the lnn module: autograd builds
// on it, and nn builds on autograd.
//
// # Memory layout
//
// A Tensor is just two exported fields:
//
//	type Tensor struct {
//		Shape []int
//		Data  []float32
//	}
//
// Data is a flat, row-major buffer: element (i, j) of an [m, n] tensor lives
// at Data[i*n+j]. There are no strides and no views — every operation
// allocates a fresh buffer and copies, so two tensors never alias unless you
// deliberately share their Data slices. Constructors copy their inputs
// (FromData, FromRows), and slice-like operations (SliceRow, SliceCol)
// return copies as well.
//
// # Shape conventions
//
// The library is intentionally 1D/2D-focused. MatMul, Transpose, Rows, Cols,
// ConcatCol, SliceCol, SliceRow, SoftmaxRows and LogSoftmaxRows are defined
// for matrices only and panic on other ranks. Elementwise operations (Add,
// Sub, Hadamard, the unary math family, Apply) work on any shape, but the
// binary ops promote a 1D-vs-1D result to shape [1, n], and the axis
// reductions are asymmetric: SumRows returns [1, n] while SumCols returns
// [m]. See doc/shapes-and-broadcasting.md for the full table and rationale.
//
// # Broadcasting
//
// Binary elementwise ops (Add, Sub, Hadamard) implement an explicit,
// enumerated subset of broadcasting — not general NumPy-style rules:
//
//   - identical shapes;
//   - a scalar (any tensor with exactly one element) against anything;
//   - a row vector ([n] or [1, n]) against a [m, n] matrix;
//   - a column vector ([m, 1]) against a [m, n] matrix;
//   - [m, 1] with [n] or [1, n], producing the outer product [m, n].
//
// Any other combination panics with a "not broadcastable" message.
//
// # Error handling: panic contracts
//
// The package reports misuse with panics rather than errors, so mistakes
// fail loudly instead of producing silently wrong numbers:
//
//   - construction: negative dimensions (New), shape/length mismatch
//     (FromData), ragged rows (FromRows), element-count overflow of int64
//     (Size);
//   - indexing: wrong index count or out-of-bounds indices (At, Set),
//     Scalar on a non-scalar, Rows/Cols on a non-2D tensor;
//   - operations: MatMul shape mismatch, non-broadcastable operands,
//     invalid slice ranges (SliceCol, SliceRow);
//   - reductions: MeanAll of an empty tensor panics because the mean of
//     zero elements is undefined (SumAll of an empty tensor is 0).
//
// Numeric domain errors (Log of a negative value, division by zero) are NOT
// checked and yield Inf/NaN exactly like plain float32 arithmetic.
//
// # Concurrency
//
// Tensors carry no synchronization and expose their storage directly. lnn
// is single-threaded by design: do not read or write one tensor from
// multiple goroutines. Give each goroutine its own tensors (see the nn
// package documentation for the full contract).
//
// A typical session:
//
//	a := tensor.FromRows(
//		[]float32{1, 2, 3},
//		[]float32{4, 5, 6},
//	)                                    // shape [2 3], row-major
//	b := tensor.FromData([]float32{10, 20, 30}, 3) // row vector
//	c := tensor.Add(a, b)                // broadcasts [3] across rows -> [2 3]
//	fmt.Println(c.At(1, 2))              // 36
package tensor
