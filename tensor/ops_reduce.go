package tensor

import (
	"fmt"
	"math"
)

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
