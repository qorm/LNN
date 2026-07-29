package tensor

import (
	"fmt"
	"math"
)

// MatMul multiplies two 2D tensors: [m, k] x [k, n] -> [m, n].
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

// Transpose returns the transpose of a 2D tensor.
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

// elemGetter resolves broadcasting: a tensor provides its value for output
// position (i, j) of a 2D result with the given column count.
type elemGetter func(i, j int) float32

func broadcastGetter(t *Tensor, outCols int) elemGetter {
	switch {
	case t.IsScalar():
		return func(i, j int) float32 { return t.Data[0] }
	case t.Dims() == 2 && t.Shape[1] == 1 && t.Shape[0] > 1: // column vector [m, 1]
		return func(i, j int) float32 { return t.Data[i] }
	case t.Dims() == 2 && t.Shape[0] > 1: // full 2D, same shape as output
		return func(i, j int) float32 { return t.Data[i*outCols+j] }
	case t.IsRowVec(): // [n] or [1, n]
		return func(i, j int) float32 { return t.Data[j] }
	default:
		panic(fmt.Sprintf("tensor: cannot broadcast shape %v", t.Shape))
	}
}

// broadcastShape resolves the output shape of a binary op, following the
// library's limited broadcasting: same shape, scalar, row-vector against a
// 2D tensor, or size-1 dimensions (a [m, 1] column vector broadcasts against
// [m, n], and [m, 1] x [1, n] produces the outer product [m, n]).
func broadcastShape(a, b *Tensor) []int {
	switch {
	case SameShape(a, b):
		return a.Shape
	case a.IsScalar():
		return b.Shape
	case b.IsScalar():
		return a.Shape
	case a.Dims() == 2 && b.IsRowVec() && a.Cols() == b.Size():
		return a.Shape
	case b.Dims() == 2 && a.IsRowVec() && b.Cols() == a.Size():
		return b.Shape
	case a.Dims() == 2 && a.Cols() == 1 && b.Dims() == 2 && b.Shape[0] == a.Shape[0]:
		return b.Shape
	case b.Dims() == 2 && b.Cols() == 1 && a.Dims() == 2 && a.Shape[0] == b.Shape[0]:
		return a.Shape
	case a.Dims() == 2 && a.Cols() == 1 && b.IsRowVec():
		return []int{a.Shape[0], b.Size()}
	case b.Dims() == 2 && b.Cols() == 1 && a.IsRowVec():
		return []int{b.Shape[0], a.Size()}
	default:
		panic(fmt.Sprintf("tensor: shapes %v and %v are not broadcastable", a.Shape, b.Shape))
	}
}

func broadcastBinary(a, b *Tensor, f func(x, y float32) float32) *Tensor {
	shape := broadcastShape(a, b)
	if len(shape) == 1 {
		shape = []int{1, shape[0]}
	}
	if len(shape) == 0 { // scalar x scalar
		return FromData([]float32{f(a.Data[0], b.Data[0])}, 1)
	}
	out := New(shape...)
	cols := shape[1]
	ga, gb := broadcastGetter(a, cols), broadcastGetter(b, cols)
	for i := 0; i < shape[0]; i++ {
		for j := 0; j < cols; j++ {
			out.Data[i*cols+j] = f(ga(i, j), gb(i, j))
		}
	}
	return out
}

// Add computes a + b with broadcasting.
func Add(a, b *Tensor) *Tensor { return broadcastBinary(a, b, func(x, y float32) float32 { return x + y }) }

// Sub computes a - b with broadcasting.
func Sub(a, b *Tensor) *Tensor { return broadcastBinary(a, b, func(x, y float32) float32 { return x - y }) }

// Hadamard computes elementwise a * b with broadcasting.
func Hadamard(a, b *Tensor) *Tensor {
	return broadcastBinary(a, b, func(x, y float32) float32 { return x * y })
}

// Scale multiplies every element by s.
func Scale(a *Tensor, s float32) *Tensor {
	out := a.ZerosLike()
	for i, v := range a.Data {
		out.Data[i] = v * s
	}
	return out
}

// Neg negates every element.
func Neg(a *Tensor) *Tensor { return Scale(a, -1) }

// Apply maps f over every element.
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

// Tanh applies tanh elementwise.
func Tanh(a *Tensor) *Tensor { return Apply(a, func(x float32) float32 { return float32(math.Tanh(float64(x))) }) }

// Sigmoid applies the logistic sigmoid elementwise.
func Sigmoid(a *Tensor) *Tensor { return Apply(a, sigmoid) }

// Exp applies exp elementwise.
func Exp(a *Tensor) *Tensor { return Apply(a, func(x float32) float32 { return float32(math.Exp(float64(x))) }) }

// Log applies natural log elementwise.
func Log(a *Tensor) *Tensor { return Apply(a, func(x float32) float32 { return float32(math.Log(float64(x))) }) }

// Pow raises every element to p.
func Pow(a *Tensor, p float32) *Tensor {
	return Apply(a, func(x float32) float32 { return float32(math.Pow(float64(x), float64(p))) })
}

// Softplus applies log(1 + e^x) elementwise, stably.
func Softplus(a *Tensor) *Tensor {
	return Apply(a, func(x float32) float32 {
		if x > 20 {
			return x
		}
		return float32(math.Log1p(math.Exp(float64(x))))
	})
}

// Clip clamps every element to [lo, hi].
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

// ConcatCol concatenates 2D tensors along the column axis.
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

// SliceCol returns columns [from, to) of a 2D tensor.
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

// SliceRow returns row i of a 2D tensor with shape [1, n].
func SliceRow(a *Tensor, i int) *Tensor {
	if a.Dims() != 2 || i < 0 || i >= a.Rows() {
		panic(fmt.Sprintf("tensor.SliceRow: invalid row %d for shape %v", i, a.Shape))
	}
	n := a.Cols()
	out := New(1, n)
	copy(out.Data, a.Data[i*n:(i+1)*n])
	return out
}

// SumAll returns the sum of all elements as a scalar tensor.
func SumAll(a *Tensor) *Tensor {
	var s float32
	for _, v := range a.Data {
		s += v
	}
	return FromData([]float32{s}, 1)
}

// MeanAll returns the mean of all elements as a scalar tensor.
func MeanAll(a *Tensor) *Tensor {
	return FromData([]float32{SumAll(a).Data[0] / float32(a.Size())}, 1)
}

// SumRows sums over axis 0 of a 2D tensor, returning shape [1, n].
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

// SumCols sums over axis 1 of a 2D tensor, returning 1D shape [m].
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

// SumToShape reduces a broadcast-produced gradient back to a target shape:
// identical shape passes through, scalars get the total sum, row vectors get
// column sums, column vectors get row sums. It panics on any other
// combination.
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
			s.Shape = []int{shape[0]}
		}
		return s
	case grad.Dims() == 2 && target.Dims() == 2 && target.Shape[1] == 1 && grad.Shape[0] == target.Shape[0]:
		s := SumCols(grad)
		s.Shape = []int{shape[0], 1}
		return s
	default:
		panic(fmt.Sprintf("tensor.SumToShape: cannot reduce shape %v to %v", grad.Shape, shape))
	}
}

// LogSoftmaxRows applies log-softmax to each row of a 2D tensor, computed in
// the numerically stable max-subtracted form.
func LogSoftmaxRows(a *Tensor) *Tensor {
	m, n := a.Rows(), a.Cols()
	out := New(m, n)
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

// SoftmaxRows applies softmax to each row of a 2D tensor.
func SoftmaxRows(a *Tensor) *Tensor {
	m, n := a.Rows(), a.Cols()
	out := New(m, n)
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
