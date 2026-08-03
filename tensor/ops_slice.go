package tensor

import (
	"fmt"
)

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
