package autograd

import (
	"fmt"
	"math"

	"lnn/tensor"
)

// opKind tags a node's operation so runBackward can dispatch the gradient
// propagation without a per-node closure (see Variable). The zero value
// marks leaves, which have no backward step.
type opKind uint8

const (
	opLeaf opKind = iota
	opMatMul
	opAdd
	opSub
	opHadamard
	opScale
	opTanh
	opSigmoid
	opExp
	opLog
	opPow
	opSoftplus
	opAbs
	opRelu
	opDiv
	opConcatCol
	opSliceCol
	opSliceRow
	opSumAll
	opMeanAll
	opGatherRows
	opLogSoftmaxRows
)

// runBackward propagates v's accumulated gradient to its parents. The case
// bodies are exactly the gradient formulas the per-op closures used to
// carry; operands are read from v.parents and the captured constants from
// v's payload fields.
func (v *Variable) runBackward() {
	switch v.kind {
	case opLeaf:
		// Leaves have no backward step.
	case opMatMul:
		// MatMulTransB/MatMulTransA read the transposed entries in place —
		// the identical products and accumulation order as MatMul over an
		// explicit Transpose, minus the two transpose buffers.
		a, b := v.parents[0], v.parents[1]
		a.addGrad(tensor.MatMulTransB(v.Grad, b.Data))
		b.addGrad(tensor.MatMulTransA(a.Data, v.Grad))
	case opAdd:
		// Ownership: the a-branch hands v.Grad to a via SumToShapeTake,
		// which returns v.Grad itself when a's shape matches (no
		// reduction). That is safe because reverse-topological order
		// guarantees every consumer of v has finished contributing before
		// this step runs, and nothing reads v.Grad afterwards except the
		// b-branch below, which only reads it to build its own tensor. The
		// b-branch therefore uses the cloning SumToShape: when both
		// operands share v's shape it must hand b a distinct buffer, or
		// a.Grad and b.Grad would alias and later accumulation into either
		// would corrupt the other (e.g. Add(x, y) with both leaves reused
		// downstream).
		a, b := v.parents[0], v.parents[1]
		a.addGrad(tensor.SumToShapeTake(v.Grad, a.Data.Shape))
		b.addGrad(tensor.SumToShape(v.Grad, b.Data.Shape))
	case opSub:
		// The a-branch may take v.Grad directly (see opAdd). The b-branch
		// goes through negReduce, whose result is always a fresh buffer.
		a, b := v.parents[0], v.parents[1]
		a.addGrad(tensor.SumToShapeTake(v.Grad, a.Data.Shape))
		b.addGrad(negReduce(v.Grad, b.Data.Shape))
	case opHadamard:
		// Each branch produces a fresh buffer through hadamardReduce — the
		// full product when the shape matches, or the reduced gradient
		// directly (two distinct buffers even when a == b). v.Grad is only
		// read, never handed off, so addGrad may take either result.
		a, b := v.parents[0], v.parents[1]
		a.addGrad(hadamardReduce(v.Grad, b.Data, a.Data.Shape))
		b.addGrad(hadamardReduce(v.Grad, a.Data, b.Data.Shape))
	case opScale:
		v.parents[0].addGrad(tensor.Scale(v.Grad, v.scalar))
	case opTanh:
		// Fused g ⊙ (1 − tanh²): the per-element operation sequence is
		// unchanged from the former composition x*x, 1−r, g⊙r.
		a := v.parents[0]
		gd, x := v.Grad.Data, v.Data.Data
		r := tensor.New(elemwiseGradShape(a.Data.Shape)...)
		for i := range r.Data {
			// mul32 rounds x*x exactly as the former Hadamard did before
			// Sub computed 1−r; without the barrier the compiler could
			// fold 1−x*x into a fused negative-multiply-subtract with a
			// single rounding and drift ~1 ULP.
			sq := mul32(x[i], x[i])
			r.Data[i] = mul32(gd[i], 1-sq)
		}
		a.addGrad(r)
	case opSigmoid:
		// Fused g ⊙ σ ⊙ (1−σ), evaluating x*(1−x) per element exactly as
		// the former Sub-then-Hadamard composition did.
		a := v.parents[0]
		gd, x := v.Grad.Data, v.Data.Data
		r := tensor.New(elemwiseGradShape(a.Data.Shape)...)
		for i := range r.Data {
			r.Data[i] = mul32(gd[i], mul32(x[i], 1-x[i]))
		}
		a.addGrad(r)
	case opExp:
		a := v.parents[0]
		a.addGrad(tensor.Hadamard(v.Grad, v.Data))
	case opLog:
		// Fused g ⊙ (1/x): the reciprocal is computed with the same
		// float32 division as the former Apply, then multiplied.
		a := v.parents[0]
		gd, x := v.Grad.Data, a.Data.Data
		r := tensor.New(elemwiseGradShape(a.Data.Shape)...)
		for i := range r.Data {
			r.Data[i] = mul32(gd[i], 1/x[i])
		}
		a.addGrad(r)
	case opPow:
		a := v.parents[0]
		p := v.scalar
		if p == 0 {
			// d/dx x^0 == 0 everywhere. Computing p*x^(p-1) directly would
			// evaluate 0 * x^-1, which is 0*Inf = NaN at x == 0.
			a.addGrad(tensor.New(a.Data.Shape...))
			return
		}
		// Fused g ⊙ p·x^(p−1): the power goes through the identical
		// math.Pow(float64) call and float32 conversion as tensor.Pow, and
		// p·r is the same product (commuted) as the former Scale's r·p.
		pm1 := p - 1
		gd, x := v.Grad.Data, a.Data.Data
		r := tensor.New(elemwiseGradShape(a.Data.Shape)...)
		for i := range r.Data {
			pw := float32(math.Pow(float64(x[i]), float64(pm1)))
			r.Data[i] = mul32(gd[i], mul32(p, pw))
		}
		a.addGrad(r)
	case opSoftplus:
		a := v.parents[0]
		a.addGrad(tensor.Hadamard(v.Grad, tensor.Sigmoid(a.Data)))
	case opAbs:
		// Fused g ⊙ sign(x): the same sign classification and the same
		// g*mask multiplication as the former Apply-then-Hadamard pair.
		a := v.parents[0]
		gd, x := v.Grad.Data, a.Data.Data
		r := tensor.New(elemwiseGradShape(a.Data.Shape)...)
		for i := range r.Data {
			var mask float32
			switch {
			case x[i] > 0:
				mask = 1
			case x[i] < 0:
				mask = -1
			}
			r.Data[i] = mul32(gd[i], mask)
		}
		a.addGrad(r)
	case opRelu:
		// Fused g ⊙ [x > 0]: the same mask and the same g*mask
		// multiplication as the former Apply-then-Hadamard pair.
		a := v.parents[0]
		gd, x := v.Grad.Data, a.Data.Data
		r := tensor.New(elemwiseGradShape(a.Data.Shape)...)
		for i := range r.Data {
			var mask float32
			if x[i] > 0 {
				mask = 1
			}
			r.Data[i] = mul32(gd[i], mask)
		}
		a.addGrad(r)
	case opDiv:
		// da = g / b, fused product-or-reduced-product (see hadamardReduce);
		// the result is a fresh buffer dedicated to a.
		// db = -g·a/b², reduced first, then scaled (see Div's doc comment).
		// The final negated scaling is fused into one buffer.
		a, b := v.parents[0], v.parents[1]
		inv := v.aux
		a.addGrad(hadamardReduce(v.Grad, inv, a.Data.Shape))
		ga := hadamardReduce(v.Grad, a.Data, b.Data.Shape)
		b.addGrad(negHadamardPow2(ga, b.Data))
	case opConcatCol:
		off := 0
		for _, p := range v.parents {
			p.addGrad(tensor.SliceCol(v.Grad, off, off+p.Data.Cols()))
			off += p.Data.Cols()
		}
	case opSliceCol:
		a := v.parents[0]
		from, to := v.from, v.to
		g := tensor.New(a.Data.Shape...)
		rows, cols := a.Data.Rows(), to-from
		for i := 0; i < rows; i++ {
			copy(g.Data[i*a.Data.Cols()+from:i*a.Data.Cols()+to], v.Grad.Data[i*cols:(i+1)*cols])
		}
		a.addGrad(g)
	case opSliceRow:
		a := v.parents[0]
		g := tensor.New(a.Data.Shape...)
		n := a.Data.Cols()
		i := v.from
		copy(g.Data[i*n:(i+1)*n], v.Grad.Data)
		a.addGrad(g)
	case opSumAll:
		// Broadcast the scalar gradient into a fresh buffer directly. The
		// former OnesLike-then-Scale form multiplied each one by the
		// scalar; 1*s == s is exact for every float32.
		a := v.parents[0]
		s := v.Grad.Scalar()
		r := tensor.New(a.Data.Shape...)
		for i := range r.Data {
			r.Data[i] = s
		}
		a.addGrad(r)
	case opMeanAll:
		// Broadcast g/size directly; see opSumAll for why dropping the
		// multiply-by-one is bit-identical.
		a := v.parents[0]
		s := v.Grad.Scalar() / float32(a.Data.Size())
		r := tensor.New(a.Data.Shape...)
		for i := range r.Data {
			r.Data[i] = s
		}
		a.addGrad(r)
	case opGatherRows:
		a := v.parents[0]
		g := tensor.New(a.Data.Shape...)
		cols := a.Data.Cols()
		for i, j := range v.idx {
			g.Data[i*cols+j] += v.Grad.Data[i]
		}
		a.addGrad(g)
	case opLogSoftmaxRows:
		// Fused g − softmax⊙rowsum: per element the same product sm*rs is
		// subtracted from g, in the same row-major order.
		a := v.parents[0]
		softmax := tensor.Exp(v.Data)
		rowsum := tensor.SumCols(v.Grad)
		rowsum.Shape = []int{rowsum.Size(), 1}
		gd, sm, rs := v.Grad.Data, softmax.Data, rowsum.Data
		n := v.Data.Cols()
		r := tensor.New(v.Data.Shape...)
		for i := 0; i < v.Data.Rows(); i++ {
			row, srow, rsv := r.Data[i*n:(i+1)*n], sm[i*n:(i+1)*n], rs[i]
			for j := range row {
				// rounded product, then subtract: no FMA fusion
				row[j] = gd[i*n+j] - mul32(srow[j], rsv)
			}
		}
		a.addGrad(r)
	}
}

// hadamardReduce evaluates SumToShapeTake(Hadamard(g, x), shape) — the
// gradient contribution of one operand of an elementwise product — without
// materializing the full-size product when a reduction is due. When the
// target shape matches g's, the product itself is the gradient buffer (one
// allocation, handed to addGrad). When the target is a scalar, row vector or
// column vector, the reduction accumulates the elementwise products directly
// into the reduced buffer: same multiplicands, same summation order, so the
// values are bit-identical to the two-step form. x may be any shape that
// broadcasts against g (accessed through xRowAccess). Anything else —
// unreachable for broadcast-compatible operands — falls back to the unfused
// composition, preserving the historical panic contract exactly.
func hadamardReduce(g, x *tensor.Tensor, shape []int) *tensor.Tensor {
	if sameShape(g.Shape, shape) {
		return tensor.Hadamard(g, x)
	}
	switch {
	case shapeSize(shape) == 1:
		// The target being scalar means the other operand determined the
		// output shape, so x shares g's layout and flat indexing is safe.
		r := tensor.New(1)
		gd, xd := g.Data, x.Data
		var s float32
		for k := range gd {
			s += mul32(gd[k], xd[k]) // rounded product, then add: no FMA
		}
		r.Data[0] = s
		return r
	case len(g.Shape) == 2 && targetIsRowVec(shape) && g.Shape[1] == shapeSize(shape):
		n := shapeSize(shape)
		r := tensor.New(1, n)
		for i := 0; i < g.Shape[0]; i++ {
			xb, xs := xRowAccess(x, i, n)
			grow := g.Data[i*n : (i+1)*n]
			for j := 0; j < n; j++ {
				// mul32 rounds the product exactly as the former
				// Hadamard-then-SumRows pair did (product stored to a
				// tensor element, then accumulated), and its conversion
				// node bars arm64 FMA fusion that would collapse the two
				// roundings into one and drift ~1 ULP.
				r.Data[j] += mul32(grow[j], x.Data[xb+xs*j])
			}
		}
		if len(shape) == 1 {
			r.Shape = []int{shape[0]}
		}
		return r
	case len(g.Shape) == 2 && len(shape) == 2 && shape[1] == 1 && g.Shape[0] == shape[0]:
		n := g.Shape[1]
		r := tensor.New(shape[0], 1)
		for i := 0; i < shape[0]; i++ {
			xb, xs := xRowAccess(x, i, n)
			grow := g.Data[i*n : (i+1)*n]
			var s float32
			for j := 0; j < n; j++ {
				s += mul32(grow[j], x.Data[xb+xs*j]) // see note above: no FMA
			}
			r.Data[i] = s
		}
		return r
	default:
		return tensor.SumToShapeTake(tensor.Hadamard(g, x), shape)
	}
}

// negReduce evaluates SumToShapeTake(Neg(g), shape) for the subtracted
// operand's gradient, fusing the sign flip into the reduction. Negation is an
// exact sign flip whether written -v or v*(-1) (as tensor.Neg does), so the
// fused addends are bit-identical to the two-step form.
func negReduce(g *tensor.Tensor, shape []int) *tensor.Tensor {
	if sameShape(g.Shape, shape) {
		return tensor.Neg(g)
	}
	switch {
	case shapeSize(shape) == 1:
		r := tensor.New(1)
		var s float32
		for _, v := range g.Data {
			s += -v
		}
		r.Data[0] = s
		return r
	case len(g.Shape) == 2 && targetIsRowVec(shape) && g.Shape[1] == shapeSize(shape):
		n := shapeSize(shape)
		r := tensor.New(1, n)
		for i := 0; i < g.Shape[0]; i++ {
			grow := g.Data[i*n : (i+1)*n]
			for j := 0; j < n; j++ {
				r.Data[j] += -grow[j]
			}
		}
		if len(shape) == 1 {
			r.Shape = []int{shape[0]}
		}
		return r
	case len(g.Shape) == 2 && len(shape) == 2 && shape[1] == 1 && g.Shape[0] == shape[0]:
		n := g.Shape[1]
		r := tensor.New(shape[0], 1)
		for i := 0; i < shape[0]; i++ {
			grow := g.Data[i*n : (i+1)*n]
			var s float32
			for j := 0; j < n; j++ {
				s += -grow[j]
			}
			r.Data[i] = s
		}
		return r
	default:
		return tensor.SumToShapeTake(tensor.Neg(g), shape)
	}
}

// negHadamardPow2 evaluates Neg(Hadamard(ga, Pow(b, -2))) in one pass: the
// per-element power goes through the identical math.Pow(float64) call and
// float32 conversion as tensor.Pow, and x*(-1) is the same exact sign flip
// as -x, so the result is bit-identical to the three-tensor composition.
func negHadamardPow2(ga, b *tensor.Tensor) *tensor.Tensor {
	// The legacy chain ended in a Hadamard, whose 1D-lift the result shape
	// must replicate (see elemwiseGradShape).
	r := tensor.New(elemwiseGradShape(ga.Shape)...)
	for i := range r.Data {
		pb2 := float32(math.Pow(float64(b.Data[i]), -2))
		m := ga.Data[i] * pb2 // rounded before the sign flip, as in the old chain
		r.Data[i] = -m
	}
	return r
}

// xRowAccess classifies how x broadcasts against a row of the full-size
// gradient: for output row i, element j is x.Data[base+stride*j]. The case
// order mirrors tensor's broadcast dispatch exactly.
func xRowAccess(x *tensor.Tensor, i, outCols int) (base, stride int) {
	switch {
	case x.Size() == 1:
		return 0, 0
	case x.Dims() == 2 && x.Shape[1] == 1 && x.Shape[0] > 1:
		return i, 0
	case x.Dims() == 2 && x.Shape[0] > 1:
		return i * outCols, 1
	case x.IsRowVec():
		return 0, 1
	default:
		panic(fmt.Sprintf("tensor: cannot broadcast shape %v", x.Shape))
	}
}

func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func shapeSize(s []int) int {
	n := 1
	for _, d := range s {
		n *= d
	}
	return n
}

func targetIsRowVec(shape []int) bool {
	return len(shape) == 1 || (len(shape) == 2 && shape[0] == 1)
}

// elemwiseGradShape returns the buffer shape a fused elementwise backward
// must allocate to stay bit-compatible with the legacy composition, which
// ran the final gradient product through tensor.Hadamard: broadcastBinary
// lifts 1D results to [1, n], so historical leaf gradients of a [n] operand
// carry shape [1, n] (a documented library quirk — replicated faithfully
// here rather than "fixed", since shapes are part of the backward contract).
func elemwiseGradShape(shape []int) []int {
	if len(shape) == 1 {
		return []int{1, shape[0]}
	}
	return shape
}

// mul32 computes the float32 product a*b with the exact single rounding of a
// native float32 multiply (the exact product fits a 48-bit mantissa, which
// float64 represents precisely, so rounding it to float32 is bit-identical
// to the hardware float32 operation). Writing the product through this
// conversion keeps an explicit rounding node between multiply and a
// following add/sub in the SSA graph, which bars arm64 FMA/FNMS fusion
// across fused-loop statements — fusion would collapse the historical
// two-step rounding (product stored to a tensor, then accumulated) into one
// and drift ~1 ULP.
func mul32(a, b float32) float32 { return float32(float64(a) * float64(b)) }

// MatMul differentiably multiplies two 2D variables.
func MatMul(a, b *Variable) *Variable {
	return newOp(tensor.MatMul(a.Data, b.Data), []*Variable{a, b}, opMatMul)
}

// Add differentiably computes a + b with broadcasting.
func Add(a, b *Variable) *Variable {
	return newOp(tensor.Add(a.Data, b.Data), []*Variable{a, b}, opAdd)
}

// Sub differentiably computes a - b with broadcasting.
func Sub(a, b *Variable) *Variable {
	return newOp(tensor.Sub(a.Data, b.Data), []*Variable{a, b}, opSub)
}

// Hadamard differentiably computes elementwise a * b with broadcasting.
func Hadamard(a, b *Variable) *Variable {
	return newOp(tensor.Hadamard(a.Data, b.Data), []*Variable{a, b}, opHadamard)
}

// Scale multiplies every element of a by the constant s.
func Scale(a *Variable, s float32) *Variable {
	out := newOp(tensor.Scale(a.Data, s), []*Variable{a}, opScale)
	out.scalar = s
	return out
}

// Neg negates every element of a.
func Neg(a *Variable) *Variable { return Scale(a, -1) }

// Tanh applies tanh elementwise.
func Tanh(a *Variable) *Variable {
	return newOp(tensor.Tanh(a.Data), []*Variable{a}, opTanh)
}

// Sigmoid applies the logistic sigmoid elementwise.
func Sigmoid(a *Variable) *Variable {
	return newOp(tensor.Sigmoid(a.Data), []*Variable{a}, opSigmoid)
}

// Exp applies exp elementwise.
func Exp(a *Variable) *Variable {
	return newOp(tensor.Exp(a.Data), []*Variable{a}, opExp)
}

// Log applies natural log elementwise.
func Log(a *Variable) *Variable {
	return newOp(tensor.Log(a.Data), []*Variable{a}, opLog)
}

// Pow raises every element of a to the constant power p.
func Pow(a *Variable, p float32) *Variable {
	out := newOp(tensor.Pow(a.Data, p), []*Variable{a}, opPow)
	out.scalar = p
	return out
}

// Softplus applies log(1 + e^x) elementwise.
func Softplus(a *Variable) *Variable {
	return newOp(tensor.Softplus(a.Data), []*Variable{a}, opSoftplus)
}

// Abs applies |x| elementwise. It is not differentiable at 0.
func Abs(a *Variable) *Variable {
	return newOp(tensor.Apply(a.Data, func(x float32) float32 {
		if x < 0 {
			return -x
		}
		return x
	}), []*Variable{a}, opAbs)
}

// Relu applies max(0, x) elementwise. The gradient at 0 is taken as 0.
func Relu(a *Variable) *Variable {
	return newOp(tensor.Apply(a.Data, func(x float32) float32 {
		if x > 0 {
			return x
		}
		return 0
	}), []*Variable{a}, opRelu)
}

// Div differentiably computes a / b elementwise with broadcasting; b must be
// nonzero (b == 0 yields +/-Inf in the forward pass, as float32 division does).
//
// Div is a single closed-form graph node. The previous implementation
// composed Hadamard(a, Pow(b, -1)), which recorded two op nodes and ran
// Pow's backward on top of Hadamard's. The forward deliberately reuses the
// exact tensor-level computation of that composition (a ⊙ pow(b, -1)), so
// shapes, broadcasting and values are bit-identical. The backward follows
// the quotient rule, da = g/b and db = -g·a/b²; because b⁻² is constant
// along every axis b was broadcast over, db may reduce g·a to b's shape
// first and only then scale by -b⁻² — which equals the closed form and also
// keeps gradients bit-identical to the legacy two-node chain.
func Div(a, b *Variable) *Variable {
	inv := tensor.Pow(b.Data, -1)
	out := newOp(tensor.Hadamard(a.Data, inv), []*Variable{a, b}, opDiv)
	out.aux = inv
	return out
}

// ConcatCol concatenates 2D variables along the column axis.
func ConcatCol(vs ...*Variable) *Variable {
	if len(vs) == 0 {
		panic("autograd.ConcatCol: no inputs")
	}
	ts := make([]*tensor.Tensor, len(vs))
	for i, v := range vs {
		ts[i] = v.Data
	}
	return newOp(tensor.ConcatCol(ts...), vs, opConcatCol)
}

// SliceCol differentiably extracts columns [from, to) of a 2D variable.
func SliceCol(a *Variable, from, to int) *Variable {
	out := newOp(tensor.SliceCol(a.Data, from, to), []*Variable{a}, opSliceCol)
	out.from, out.to = from, to
	return out
}

// Col differentiably extracts column i of a 2D variable with shape [m, 1].
func Col(a *Variable, i int) *Variable { return SliceCol(a, i, i+1) }

// SliceRow differentiably extracts row i of a 2D variable with shape [1, n].
func SliceRow(a *Variable, i int) *Variable {
	out := newOp(tensor.SliceRow(a.Data, i), []*Variable{a}, opSliceRow)
	out.from = i
	return out
}

// SumAll sums every element into a scalar variable.
func SumAll(a *Variable) *Variable {
	return newOp(tensor.SumAll(a.Data), []*Variable{a}, opSumAll)
}

// MeanAll averages every element into a scalar variable.
func MeanAll(a *Variable) *Variable {
	return newOp(tensor.MeanAll(a.Data), []*Variable{a}, opMeanAll)
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
	out := newOp(tensor.FromData(data, m), []*Variable{a}, opGatherRows)
	out.idx = idx
	return out
}

// LogSoftmaxRows applies the numerically stable log-softmax to each row.
func LogSoftmaxRows(a *Variable) *Variable {
	return newOp(tensor.LogSoftmaxRows(a.Data), []*Variable{a}, opLogSoftmaxRows)
}
