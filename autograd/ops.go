package autograd

import (
	"fmt"

	"lnn/tensor"
)

// MatMul differentiably multiplies two 2D variables.
func MatMul(a, b *Variable) *Variable {
	out := newOp(tensor.MatMul(a.Data, b.Data), []*Variable{a, b}, nil)
	out.backward = func() {
		a.addGrad(tensor.MatMul(out.Grad, tensor.Transpose(b.Data)))
		b.addGrad(tensor.MatMul(tensor.Transpose(a.Data), out.Grad))
	}
	return out
}

// Add differentiably computes a + b with broadcasting.
func Add(a, b *Variable) *Variable {
	out := newOp(tensor.Add(a.Data, b.Data), []*Variable{a, b}, nil)
	out.backward = func() {
		a.addGrad(tensor.SumToShape(out.Grad, a.Data.Shape))
		b.addGrad(tensor.SumToShape(out.Grad, b.Data.Shape))
	}
	return out
}

// Sub differentiably computes a - b with broadcasting.
func Sub(a, b *Variable) *Variable {
	out := newOp(tensor.Sub(a.Data, b.Data), []*Variable{a, b}, nil)
	out.backward = func() {
		a.addGrad(tensor.SumToShape(out.Grad, a.Data.Shape))
		b.addGrad(tensor.SumToShape(tensor.Neg(out.Grad), b.Data.Shape))
	}
	return out
}

// Hadamard differentiably computes elementwise a * b with broadcasting.
func Hadamard(a, b *Variable) *Variable {
	out := newOp(tensor.Hadamard(a.Data, b.Data), []*Variable{a, b}, nil)
	out.backward = func() {
		a.addGrad(tensor.SumToShape(tensor.Hadamard(out.Grad, b.Data), a.Data.Shape))
		b.addGrad(tensor.SumToShape(tensor.Hadamard(out.Grad, a.Data), b.Data.Shape))
	}
	return out
}

// Scale multiplies every element of a by the constant s.
func Scale(a *Variable, s float32) *Variable {
	out := newOp(tensor.Scale(a.Data, s), []*Variable{a}, nil)
	out.backward = func() {
		a.addGrad(tensor.Scale(out.Grad, s))
	}
	return out
}

// Neg negates every element of a.
func Neg(a *Variable) *Variable { return Scale(a, -1) }

// Tanh applies tanh elementwise.
func Tanh(a *Variable) *Variable {
	out := newOp(tensor.Tanh(a.Data), []*Variable{a}, nil)
	out.backward = func() {
		one := out.Data.OnesLike()
		deriv := tensor.Sub(one, tensor.Hadamard(out.Data, out.Data))
		a.addGrad(tensor.Hadamard(out.Grad, deriv))
	}
	return out
}

// Sigmoid applies the logistic sigmoid elementwise.
func Sigmoid(a *Variable) *Variable {
	out := newOp(tensor.Sigmoid(a.Data), []*Variable{a}, nil)
	out.backward = func() {
		one := out.Data.OnesLike()
		deriv := tensor.Hadamard(out.Data, tensor.Sub(one, out.Data))
		a.addGrad(tensor.Hadamard(out.Grad, deriv))
	}
	return out
}

// Exp applies exp elementwise.
func Exp(a *Variable) *Variable {
	out := newOp(tensor.Exp(a.Data), []*Variable{a}, nil)
	out.backward = func() {
		a.addGrad(tensor.Hadamard(out.Grad, out.Data))
	}
	return out
}

// Log applies natural log elementwise.
func Log(a *Variable) *Variable {
	out := newOp(tensor.Log(a.Data), []*Variable{a}, nil)
	out.backward = func() {
		inv := tensor.Apply(a.Data, func(x float32) float32 { return 1 / x })
		a.addGrad(tensor.Hadamard(out.Grad, inv))
	}
	return out
}

// Pow raises every element of a to the constant power p.
func Pow(a *Variable, p float32) *Variable {
	out := newOp(tensor.Pow(a.Data, p), []*Variable{a}, nil)
	out.backward = func() {
		if p == 0 {
			// d/dx x^0 == 0 everywhere. Computing p*x^(p-1) directly would
			// evaluate 0 * x^-1, which is 0*Inf = NaN at x == 0.
			a.addGrad(tensor.New(a.Data.Shape...))
			return
		}
		deriv := tensor.Scale(tensor.Pow(a.Data, p-1), p)
		a.addGrad(tensor.Hadamard(out.Grad, deriv))
	}
	return out
}

// Softplus applies log(1 + e^x) elementwise.
func Softplus(a *Variable) *Variable {
	out := newOp(tensor.Softplus(a.Data), []*Variable{a}, nil)
	out.backward = func() {
		a.addGrad(tensor.Hadamard(out.Grad, tensor.Sigmoid(a.Data)))
	}
	return out
}

// Abs applies |x| elementwise. It is not differentiable at 0.
func Abs(a *Variable) *Variable {
	out := newOp(tensor.Apply(a.Data, func(x float32) float32 {
		if x < 0 {
			return -x
		}
		return x
	}), []*Variable{a}, nil)
	out.backward = func() {
		sign := tensor.Apply(a.Data, func(x float32) float32 {
			switch {
			case x > 0:
				return 1
			case x < 0:
				return -1
			default:
				return 0
			}
		})
		a.addGrad(tensor.Hadamard(out.Grad, sign))
	}
	return out
}

// Relu applies max(0, x) elementwise. The gradient at 0 is taken as 0.
func Relu(a *Variable) *Variable {
	out := newOp(tensor.Apply(a.Data, func(x float32) float32 {
		if x > 0 {
			return x
		}
		return 0
	}), []*Variable{a}, nil)
	out.backward = func() {
		mask := tensor.Apply(a.Data, func(x float32) float32 {
			if x > 0 {
				return 1
			}
			return 0
		})
		a.addGrad(tensor.Hadamard(out.Grad, mask))
	}
	return out
}

// Div computes a / b elementwise with broadcasting; b must be nonzero.
func Div(a, b *Variable) *Variable { return Hadamard(a, Pow(b, -1)) }

// ConcatCol concatenates 2D variables along the column axis.
func ConcatCol(vs ...*Variable) *Variable {
	if len(vs) == 0 {
		panic("autograd.ConcatCol: no inputs")
	}
	ts := make([]*tensor.Tensor, len(vs))
	for i, v := range vs {
		ts[i] = v.Data
	}
	out := newOp(tensor.ConcatCol(ts...), vs, nil)
	out.backward = func() {
		off := 0
		for _, v := range vs {
			v.addGrad(tensor.SliceCol(out.Grad, off, off+v.Data.Cols()))
			off += v.Data.Cols()
		}
	}
	return out
}

// SliceCol differentiably extracts columns [from, to) of a 2D variable.
func SliceCol(a *Variable, from, to int) *Variable {
	out := newOp(tensor.SliceCol(a.Data, from, to), []*Variable{a}, nil)
	out.backward = func() {
		g := tensor.New(a.Data.Shape...)
		rows, cols := a.Data.Rows(), to-from
		for i := 0; i < rows; i++ {
			copy(g.Data[i*a.Data.Cols()+from:i*a.Data.Cols()+to], out.Grad.Data[i*cols:(i+1)*cols])
		}
		a.addGrad(g)
	}
	return out
}

// Col differentiably extracts column i of a 2D variable with shape [m, 1].
func Col(a *Variable, i int) *Variable { return SliceCol(a, i, i+1) }

// SliceRow differentiably extracts row i of a 2D variable with shape [1, n].
func SliceRow(a *Variable, i int) *Variable {
	out := newOp(tensor.SliceRow(a.Data, i), []*Variable{a}, nil)
	out.backward = func() {
		g := tensor.New(a.Data.Shape...)
		n := a.Data.Cols()
		copy(g.Data[i*n:(i+1)*n], out.Grad.Data)
		a.addGrad(g)
	}
	return out
}

// SumAll sums every element into a scalar variable.
func SumAll(a *Variable) *Variable {
	out := newOp(tensor.SumAll(a.Data), []*Variable{a}, nil)
	out.backward = func() {
		a.addGrad(tensor.Scale(a.Data.OnesLike(), out.Grad.Scalar()))
	}
	return out
}

// MeanAll averages every element into a scalar variable.
func MeanAll(a *Variable) *Variable {
	out := newOp(tensor.MeanAll(a.Data), []*Variable{a}, nil)
	out.backward = func() {
		a.addGrad(tensor.Scale(a.Data.OnesLike(), out.Grad.Scalar()/float32(a.Data.Size())))
	}
	return out
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
	out := newOp(tensor.FromData(data, m), []*Variable{a}, nil)
	out.backward = func() {
		g := tensor.New(a.Data.Shape...)
		for i, j := range idx {
			g.Data[i*n+j] += out.Grad.Data[i]
		}
		a.addGrad(g)
	}
	return out
}

// LogSoftmaxRows applies the numerically stable log-softmax to each row.
func LogSoftmaxRows(a *Variable) *Variable {
	out := newOp(tensor.LogSoftmaxRows(a.Data), []*Variable{a}, nil)
	out.backward = func() {
		softmax := tensor.Exp(out.Data)
		rowsum := tensor.SumCols(out.Grad)
		rowsum.Shape = []int{rowsum.Size(), 1}
		a.addGrad(tensor.Sub(out.Grad, tensor.Hadamard(softmax, rowsum)))
	}
	return out
}
