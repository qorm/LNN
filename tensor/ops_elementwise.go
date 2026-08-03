package tensor

import (
	"math"
)

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
