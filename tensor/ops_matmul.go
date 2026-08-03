package tensor

import (
	"fmt"
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
