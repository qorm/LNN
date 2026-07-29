// Package tensor provides a minimal dense float32 n-dimensional tensor type
// with row-major layout, plus the small set of numeric operations the LNN
// library needs. It is intentionally 1D/2D-focused: MatMul is defined for
// matrices only, while elementwise operations work on any shape.
package tensor

import "fmt"

// Tensor is a dense, row-major float32 tensor.
type Tensor struct {
	Shape []int
	Data  []float32
}

// New returns a zero-filled tensor with the given shape.
func New(shape ...int) *Tensor {
	t := &Tensor{Shape: append([]int(nil), shape...)}
	t.Data = make([]float32, t.Size())
	return t
}

// FromData returns a tensor wrapping a copy of data with the given shape.
// It panics if the shape does not match len(data).
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

// Size returns the total number of elements.
func (t *Tensor) Size() int {
	n := 1
	for _, d := range t.Shape {
		n *= d
	}
	return n
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
	return &Tensor{Shape: append([]int(nil), t.Shape...), Data: append([]float32(nil), t.Data...)}
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

// Stack stacks tensors along a new leading dimension. All inputs must share
// one shape; the result has shape [len(ts), shape...].
func Stack(ts ...*Tensor) *Tensor {
	if len(ts) == 0 {
		panic("tensor.Stack: no tensors")
	}
	shape := append([]int{len(ts)}, ts[0].Shape...)
	out := New(shape...)
	n := ts[0].Size()
	for i, t := range ts {
		if !SameShape(t, ts[0]) {
			panic(fmt.Sprintf("tensor.Stack: shape %v does not match %v", t.Shape, ts[0].Shape))
		}
		copy(out.Data[i*n:(i+1)*n], t.Data)
	}
	return out
}

// String renders small tensors for debugging.
func (t *Tensor) String() string {
	if t.Size() > 64 {
		return fmt.Sprintf("Tensor(shape=%v, data=[%v ... %v])", t.Shape, t.Data[0], t.Data[len(t.Data)-1])
	}
	return fmt.Sprintf("Tensor(shape=%v, data=%v)", t.Shape, t.Data)
}
