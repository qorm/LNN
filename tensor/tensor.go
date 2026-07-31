// Package tensor provides a minimal dense float32 n-dimensional tensor type
// with row-major layout, plus the small set of numeric operations the LNN
// library needs. It is intentionally 1D/2D-focused: MatMul is defined for
// matrices only, while elementwise operations work on any shape.
package tensor

import (
	"fmt"
	"math"
	"math/bits"
)

// maxInlineRank is the rank up to which a tensor's shape is stored inline in
// the struct (shapeBuf) rather than in a separate heap allocation.
//
// Coverage: every operator in this library is 1D/2D-focused — MatMul is
// matrices only and the broadcasting binary ops, reductions and softmax
// families all reject Dims() != 2. Stack was the sole rank-3 producer and it
// is deleted in v0.4.0 (API hygiene, zero in-library callers), so live library
// tensors top out at rank 2; maxInlineRank=4 leaves two ranks of headroom for
// direct external construction. The serialize package carries its own
// rank+dims wire stream (up to rank 8) and constructs tensors directly without
// going through useShape, so any tensor whose rank exceeds maxInlineRank falls
// back to a copied heap slice (still correct — Shape remains the single source
// of truth — just without the allocation saving).
const maxInlineRank = 4

// Tensor is a dense, row-major float32 tensor.
//
// Shape stays the single source of truth as a []int, but for rank <=
// maxInlineRank it points into the embedded shapeBuf instead of a separately
// heap-allocated slice, removing one allocation per tensor construction (API
// hardening #12, option ②). The exported field type and all read paths are
// unchanged; shapeBuf is an implementation detail callers never touch.
type Tensor struct {
	Shape    []int
	Data     []float32
	shapeBuf [maxInlineRank]int
}

// useShape points t.Shape at the embedded shapeBuf (zero heap allocation) when
// the rank fits, otherwise falls back to a copied heap slice. It is the single
// internal choke point for (re)assigning a tensor's shape; New, Clone and
// Reshape all route through it so no code hand-writes the Shape slice.
func (t *Tensor) useShape(dims []int) {
	if len(dims) <= maxInlineRank {
		copy(t.shapeBuf[:], dims)
		t.Shape = t.shapeBuf[:len(dims)]
		return
	}
	// rank > maxInlineRank: heap fallback (e.g. serialize's rank-8 tensors).
	t.Shape = append([]int(nil), dims...)
}

// Reshape sets the tensor's shape to dims, storing ranks up to maxInlineRank
// inline in the struct and falling back to a heap copy beyond that. It is the
// exported replacement for the former direct `t.Shape = []int{...}` writes:
// it re-points Shape without reallocating Data, so the caller must pass a
// shape whose element count matches the existing buffer. Every dimension must
// be non-negative; a negative dimension panics.
func (t *Tensor) Reshape(dims ...int) {
	for _, d := range dims {
		if d < 0 {
			panic(fmt.Sprintf("tensor.Reshape: negative dimension %d in shape %v", d, dims))
		}
	}
	t.useShape(dims)
}

// New returns a zero-filled tensor with the given shape. Every dimension must
// be non-negative; a negative dimension panics.
func New(shape ...int) *Tensor {
	for _, d := range shape {
		if d < 0 {
			panic(fmt.Sprintf("tensor.New: negative dimension %d in shape %v", d, shape))
		}
	}
	t := &Tensor{}
	t.useShape(shape)
	t.Data = make([]float32, t.Size())
	return t
}

// FromData returns a tensor wrapping a copy of data with the given shape.
// It panics if the shape does not match len(data), if any dimension is
// negative, or if the shape's element count overflows int64.
func FromData(data []float32, shape ...int) *Tensor {
	t := New(shape...)
	if t.Size() != len(data) {
		panic(fmt.Sprintf("tensor.FromData: shape %v has size %d, got %d elements", shape, t.Size(), len(data)))
	}
	copy(t.Data, data)
	return t
}

// FromRows builds a 2D tensor from per-row slices.
func FromRows(rows ...[]float32) *Tensor {
	if len(rows) == 0 {
		return New(0, 0)
	}
	cols := len(rows[0])
	t := New(len(rows), cols)
	for i, r := range rows {
		if len(r) != cols {
			panic(fmt.Sprintf("tensor.FromRows: row %d has %d elements, want %d", i, len(r), cols))
		}
		copy(t.Data[i*cols:(i+1)*cols], r)
	}
	return t
}

// Size returns the total number of elements. The product of the dimensions is
// computed with overflow checks: a shape whose element count exceeds int64
// (e.g. {1 << 62, 4}) panics instead of silently wrapping around and
// describing a tensor whose Data buffer cannot possibly hold its shape.
func (t *Tensor) Size() int {
	var n uint64 = 1
	for _, d := range t.Shape {
		hi, lo := bits.Mul64(n, uint64(d))
		if hi != 0 || lo > math.MaxInt64 {
			panic(fmt.Sprintf("tensor: shape %v overflows: too many elements", t.Shape))
		}
		n = lo
	}
	return int(n)
}

// Dims returns the number of dimensions.
func (t *Tensor) Dims() int { return len(t.Shape) }

// Rows returns the first dimension of a 2D tensor.
func (t *Tensor) Rows() int { return t.must2D()[0] }

// Cols returns the second dimension of a 2D tensor.
func (t *Tensor) Cols() int { return t.must2D()[1] }

func (t *Tensor) must2D() []int {
	if t.Dims() != 2 {
		panic(fmt.Sprintf("tensor: expected 2D tensor, got shape %v", t.Shape))
	}
	return t.Shape
}

// At returns the element at the given indices.
func (t *Tensor) At(idx ...int) float32 { return t.Data[t.offset(idx)] }

// Set sets the element at the given indices.
func (t *Tensor) Set(v float32, idx ...int) { t.Data[t.offset(idx)] = v }

func (t *Tensor) offset(idx []int) int {
	if len(idx) != len(t.Shape) {
		panic(fmt.Sprintf("tensor: %d indices for shape %v", len(idx), t.Shape))
	}
	off := 0
	for i, d := range t.Shape {
		if idx[i] < 0 || idx[i] >= d {
			panic(fmt.Sprintf("tensor: index %v out of bounds for shape %v", idx, t.Shape))
		}
		off = off*d + idx[i]
	}
	return off
}

// Clone returns a deep copy.
func (t *Tensor) Clone() *Tensor {
	out := &Tensor{}
	out.useShape(t.Shape)
	out.Data = append([]float32(nil), t.Data...)
	return out
}

// ZerosLike returns a zero-filled tensor with the same shape.
func (t *Tensor) ZerosLike() *Tensor { return New(t.Shape...) }

// OnesLike returns a one-filled tensor with the same shape.
func (t *Tensor) OnesLike() *Tensor {
	out := New(t.Shape...)
	for i := range out.Data {
		out.Data[i] = 1
	}
	return out
}

// SameShape reports whether a and b have identical shapes.
func SameShape(a, b *Tensor) bool {
	if len(a.Shape) != len(b.Shape) {
		return false
	}
	for i := range a.Shape {
		if a.Shape[i] != b.Shape[i] {
			return false
		}
	}
	return true
}

// IsScalar reports whether the tensor holds exactly one element.
func (t *Tensor) IsScalar() bool { return t.Size() == 1 }

// Scalar returns the single element of a size-1 tensor.
func (t *Tensor) Scalar() float32 {
	if !t.IsScalar() {
		panic(fmt.Sprintf("tensor.Scalar: shape %v is not scalar", t.Shape))
	}
	return t.Data[0]
}

// IsRowVec reports whether the tensor can act as a broadcastable row vector
// for 2D operations: shape [n] or [1, n].
func (t *Tensor) IsRowVec() bool {
	if t.Dims() == 1 {
		return true
	}
	return t.Dims() == 2 && t.Shape[0] == 1
}

// String renders small tensors for debugging.
func (t *Tensor) String() string {
	if t.Size() > 64 {
		return fmt.Sprintf("Tensor(shape=%v, data=[%v ... %v])", t.Shape, t.Data[0], t.Data[len(t.Data)-1])
	}
	return fmt.Sprintf("Tensor(shape=%v, data=%v)", t.Shape, t.Data)
}
