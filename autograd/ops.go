package autograd

import (
	"fmt"
	"math"

	"github.com/qorm/LNN/tensor"
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
	opSigmoidHadamard
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
		// Ownership: the a-branch hands v.Grad to a via sumToShapeTake,
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
		a.addGrad(sumToShapeTake(v.Grad, a.Data.Shape))
		b.addGrad(tensor.SumToShape(v.Grad, b.Data.Shape))
	case opSub:
		// The a-branch may take v.Grad directly (see opAdd). The b-branch
		// goes through negReduce, whose result is always a fresh buffer.
		a, b := v.parents[0], v.parents[1]
		a.addGrad(sumToShapeTake(v.Grad, a.Data.Shape))
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
		if !gradMatchesElemwise(v.Grad.Shape, a.Data.Shape) {
			// Irregular seeded gradient: broadcast exactly as the legacy
			// composition did (see gradMatchesElemwise).
			one := v.Data.OnesLike()
			deriv := tensor.Sub(one, tensor.Hadamard(v.Data, v.Data))
			a.addGrad(tensor.Hadamard(v.Grad, deriv))
			return
		}
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
		if !gradMatchesElemwise(v.Grad.Shape, a.Data.Shape) {
			one := v.Data.OnesLike()
			deriv := tensor.Hadamard(v.Data, tensor.Sub(one, v.Data))
			a.addGrad(tensor.Hadamard(v.Grad, deriv))
			return
		}
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
		if !gradMatchesElemwise(v.Grad.Shape, a.Data.Shape) {
			inv := tensor.Apply(a.Data, func(x float32) float32 { return 1 / x })
			a.addGrad(tensor.Hadamard(v.Grad, inv))
			return
		}
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
		if !gradMatchesElemwise(v.Grad.Shape, a.Data.Shape) {
			deriv := tensor.Scale(tensor.Pow(a.Data, p-1), p)
			a.addGrad(tensor.Hadamard(v.Grad, deriv))
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
		if !gradMatchesElemwise(v.Grad.Shape, a.Data.Shape) {
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
			a.addGrad(tensor.Hadamard(v.Grad, sign))
			return
		}
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
		if !gradMatchesElemwise(v.Grad.Shape, a.Data.Shape) {
			mask := tensor.Apply(a.Data, func(x float32) float32 {
				if x > 0 {
					return 1
				}
				return 0
			})
			a.addGrad(tensor.Hadamard(v.Grad, mask))
			return
		}
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
		if !sameShape(v.Grad.Shape, v.Data.Shape) {
			// Irregular seeded gradient: the legacy composition's SumCols
			// reduction broadcasts differently (and panics on 1D seeds) —
			// replicate it exactly (see gradMatchesElemwise).
			softmax := tensor.Exp(v.Data)
			rowsum := tensor.SumCols(v.Grad)
			rowsum.Reshape(rowsum.Size(), 1)
			a.addGrad(tensor.Sub(v.Grad, tensor.Hadamard(softmax, rowsum)))
			return
		}
		softmax := tensor.Exp(v.Data)
		rowsum := tensor.SumCols(v.Grad)
		rowsum.Reshape(rowsum.Size(), 1)
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
	case opSigmoidHadamard:
		// Fused backward of Hadamard(Sigmoid(z), w): dz = g⊙w⊙s⊙(1−s) and
		// dw = g⊙s reduced to w's shape, where s = sigmoid(z) is the auxiliary
		// tensor the forward recorded (allocated once, never recomputed). The
		// two branches reproduce exactly what the opHadamard+opSigmoid pair
		// used to run: Hadamard's backward handed the sigmoid node the rounded
		// product g⊙w (hadamardReduce's sameShape fast path), and opSigmoid's
		// fused loop then evaluated mul32(g⊙w, mul32(s, 1−s)) per element. The
		// loop below rounds the g⊙w product through mul32 at the very same
		// spot, so the regular path is bit-identical to the legacy chain.
		z, w := v.parents[0], v.parents[1]
		s := v.aux
		if s.Dims() == 2 && sameShape(v.Grad.Shape, s.Shape) {
			// Regular 2D path (the LTC hot path): w broadcasts across s's rows
			// (xRowAccess classifies every mode forward's Hadamard accepted).
			gd, sd, wd := v.Grad.Data, s.Data, w.Data.Data
			rows, cols := s.Shape[0], s.Shape[1]
			r := tensor.New(elemwiseGradShape(z.Data.Shape)...)
			for i := 0; i < rows; i++ {
				wb, ws := xRowAccess(w.Data, i, cols)
				for j := 0; j < cols; j++ {
					k := i*cols + j
					// rounded product g⊙w, then the opSigmoid grouping: no FMA
					gw := mul32(gd[k], wd[wb+ws*j])
					r.Data[k] = mul32(gw, mul32(sd[k], 1-sd[k]))
				}
			}
			z.addGrad(r)
			// dw: the same fused product-or-reduction opHadamard's b-branch ran.
			w.addGrad(hadamardReduce(v.Grad, s, w.Data.Shape))
			return
		}
		// Non-2D operands or an irregular manually seeded gradient: reproduce
		// the legacy opHadamard+opSigmoid pair verbatim — both hadamardReduce
		// branches, then opSigmoid's own regular/fallback dispatch — so the
		// values, shapes, 1D-lift quirk and panic contract are all preserved.
		gs := hadamardReduce(v.Grad, w.Data, s.Shape)
		w.addGrad(hadamardReduce(v.Grad, s, w.Data.Shape))
		if !gradMatchesElemwise(gs.Shape, z.Data.Shape) {
			one := s.OnesLike()
			deriv := tensor.Hadamard(s, tensor.Sub(one, s))
			z.addGrad(tensor.Hadamard(gs, deriv))
			return
		}
		gsd, sd := gs.Data, s.Data
		r := tensor.New(elemwiseGradShape(z.Data.Shape)...)
		for i := range r.Data {
			r.Data[i] = mul32(gsd[i], mul32(sd[i], 1-sd[i]))
		}
		z.addGrad(r)
	}
}

// sumToShapeTake reduces grad to shape for the backward path, taking
// ownership of grad: when the shapes already match, grad itself is returned
// without a defensive clone, and the caller must not use grad afterwards
// (reverse-topological order guarantees no later consumer reads it). Every
// reduction arm still allocates a fresh buffer.
//
// Originally tensor.SumToShapeTake; moved into autograd in v0.4.0 and made
// unexported to narrow the public API surface — its only callers were the
// five backward sites in this file, and an ownership footgun on the exported
// tensor surface bought nothing external. The cloning alias-free variant
// (tensor.SumToShape) stays public for everyone else. Implemented purely on
// exported tensor primitives (SameShape/IsScalar/Dims/IsRowVec/Cols/Size/
// SumRows/SumCols/SumAll); the semantics and ownership contract, including
// the irreducible panic text, are preserved verbatim.
func sumToShapeTake(grad *tensor.Tensor, shape []int) *tensor.Tensor {
	target := &tensor.Tensor{Shape: shape}
	switch {
	case tensor.SameShape(grad, target):
		return grad
	case target.IsScalar():
		return tensor.SumAll(grad)
	case grad.Dims() == 2 && target.IsRowVec() && grad.Cols() == target.Size():
		s := tensor.SumRows(grad)
		if len(shape) == 1 {
			s.Reshape(shape[0])
		}
		return s
	case grad.Dims() == 2 && target.Dims() == 2 && target.Shape[1] == 1 && grad.Shape[0] == target.Shape[0]:
		s := tensor.SumCols(grad)
		s.Reshape(shape[0], 1)
		return s
	default:
		panic(fmt.Sprintf("tensor.SumToShape: cannot reduce shape %v to %v", grad.Shape, shape))
	}
}

// hadamardReduce evaluates sumToShapeTake(Hadamard(g, x), shape) — the
// gradient contribution of one operand of an elementwise product — without
// materializing the full-size product when a reduction is due. When the
// target shape matches g's AND the product's own broadcast shape, the
// product itself is the gradient buffer (one allocation, handed to addGrad).
// The product-shape check matters: broadcastBinary lifts 1D results to
// [1, n] (and a scalar-scalar product lands at [1, 1]), so the product can
// be shaped [1, n] even when both g and the target are [n] — the legacy
// chain's SumToShape flattened that lift away, and skipping it broke the
// gradient shape contract for 1D leaves (and panicked when one leaf then
// received both [n] and [1, n] contributions). When product and target
// disagree the reduction runs on the materialized product, exactly as the
// legacy chain did. When the target is a scalar, row vector or column
// vector with a differently shaped g, the reduction accumulates the
// elementwise products directly into the reduced buffer: same multiplicands,
// same summation order, so the values are bit-identical to the two-step
// form — valid precisely when the product would carry g's own shape (see
// productCarriesGShape) and, for scalar targets, when x shares g's flat
// layout (flatSameLayout); backward propagation always arranges both.
// Anything else — irregular seeded gradient shapes, or combinations
// unreachable for broadcast-compatible operands — falls back to the
// unfused composition, preserving the legacy values, shapes and panic
// contract exactly.
func hadamardReduce(g, x *tensor.Tensor, shape []int) *tensor.Tensor {
	if sameShape(g.Shape, shape) {
		p := tensor.Hadamard(g, x)
		if sameShape(p.Shape, shape) {
			return p
		}
		// The product's broadcast shape is wider than the target (the
		// 1D-lift quirk above): reduce it exactly as the legacy chain did.
		// sumToShapeTake returns a fresh buffer for every reduction branch,
		// and cannot alias p back to this caller.
		return sumToShapeTake(p, shape)
	}
	// The fused branches below index the pairwise products in g's flat/row
	// layout; that is exact only when the product would carry g's very
	// shape (guard) and — for the scalar branch — x shares g's flat layout.
	// Backward-propagated gradients always satisfy both; irregular manually
	// seeded shapes (outer products against g, a [1] seed over a [1,1]
	// node, …) take the legacy composition instead, panic contract
	// included.
	switch {
	case shapeSize(shape) == 1 && flatSameLayout(g, x) && !sameShape(broadcastLift(g.Shape), shape):
		// The target being scalar and x sharing g's layout means the
		// product's elements are exactly gd[k]*xd[k] in flat order. The
		// extra shape check excludes the pass-through case: when the
		// lifted product shape equals the target (say g [1] × x [1] with
		// target [1, 1]) the legacy chain's SumToShape returns the product
		// at the target's shape, not the fused [1].
		r := tensor.New(1)
		gd, xd := g.Data, x.Data
		var s float32
		for k := range gd {
			s += mul32(gd[k], xd[k]) // rounded product, then add: no FMA
		}
		r.Data[0] = s
		return r
	case len(g.Shape) == 2 && targetIsRowVec(shape) && g.Shape[1] == shapeSize(shape) && productCarriesGShape(g, x):
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
			r.Reshape(shape[0])
		}
		return r
	case len(g.Shape) == 2 && len(shape) == 2 && shape[1] == 1 && g.Shape[0] == shape[0] && productCarriesGShape(g, x):
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
		return sumToShapeTake(tensor.Hadamard(g, x), shape)
	}
}

// productCarriesGShape reports whether tensor.Hadamard(g, x) produces g's
// exact shape — broadcastBinary's 1D→[1, n] lift included — so that the
// product's element at g's flat index k is g.Data[k] paired with x's
// broadcast value (the premise the fused reduction branches rely on). The
// dispatch mirrors tensor.broadcastShapeFresh plus the 1D lift exactly; an
// unbroadcastable x returns false, leaving the legacy fallback to reproduce
// the historical forward panic.
func productCarriesGShape(g, x *tensor.Tensor) bool {
	var s []int
	switch {
	case tensor.SameShape(g, x):
		s = g.Shape
	case g.IsScalar():
		s = x.Shape
	case x.IsScalar():
		s = g.Shape
	case g.Dims() == 2 && x.IsRowVec() && g.Cols() == x.Size():
		s = g.Shape
	case x.Dims() == 2 && g.IsRowVec() && x.Cols() == g.Size():
		s = x.Shape
	case g.Dims() == 2 && g.Cols() == 1 && x.Dims() == 2 && x.Shape[0] == g.Shape[0]:
		s = x.Shape
	case x.Dims() == 2 && x.Cols() == 1 && g.Dims() == 2 && g.Shape[0] == x.Shape[0]:
		s = g.Shape
	case g.Dims() == 2 && g.Cols() == 1 && x.IsRowVec():
		s = []int{g.Shape[0], x.Size()}
	case x.Dims() == 2 && x.Cols() == 1 && g.IsRowVec():
		s = []int{x.Shape[0], g.Size()}
	default:
		return false
	}
	return sameShape(broadcastLift(s), g.Shape)
}

// broadcastLift applies broadcastBinary's output-shape normalization: 1D
// shapes are lifted to [1, n], and the empty shape (a scalar-scalar
// product) lands at [1].
func broadcastLift(s []int) []int {
	switch len(s) {
	case 0:
		return []int{1}
	case 1:
		return []int{1, s[0]}
	default:
		return s
	}
}

// flatSameLayout reports whether x's elements sit in the same flat order
// as g's (identical shapes, modulo the 1D→[1, n] lift a 1D x would take in
// a product), so that x.Data[k] is the broadcast partner of g.Data[k].
func flatSameLayout(g, x *tensor.Tensor) bool {
	return sameShape(x.Shape, g.Shape) || sameShape(elemwiseGradShape(x.Shape), g.Shape)
}

// negReduce evaluates sumToShapeTake(Neg(g), shape) for the subtracted
// operand's gradient, fusing the sign flip into the reduction. Each addend
// goes through mul32(v, -1), which reproduces tensor.Neg's v*(-1) multiply
// exactly: bit-identical to a unary minus for finite values (the exact
// product is a pure sign flip), and — unlike a unary minus, which would
// flip a NaN's sign bit — propagating NaNs the way the hardware multiply
// in the legacy Neg did.
func negReduce(g *tensor.Tensor, shape []int) *tensor.Tensor {
	if sameShape(g.Shape, shape) {
		return tensor.Neg(g)
	}
	switch {
	case shapeSize(shape) == 1:
		r := tensor.New(1)
		var s float32
		for _, v := range g.Data {
			s += mul32(v, negOne)
		}
		r.Data[0] = s
		return r
	case len(g.Shape) == 2 && targetIsRowVec(shape) && g.Shape[1] == shapeSize(shape):
		n := shapeSize(shape)
		r := tensor.New(1, n)
		for i := 0; i < g.Shape[0]; i++ {
			grow := g.Data[i*n : (i+1)*n]
			for j := 0; j < n; j++ {
				r.Data[j] += mul32(grow[j], negOne)
			}
		}
		if len(shape) == 1 {
			r.Reshape(shape[0])
		}
		return r
	case len(g.Shape) == 2 && len(shape) == 2 && shape[1] == 1 && g.Shape[0] == shape[0]:
		n := g.Shape[1]
		r := tensor.New(shape[0], 1)
		for i := 0; i < shape[0]; i++ {
			grow := g.Data[i*n : (i+1)*n]
			var s float32
			for j := 0; j < n; j++ {
				s += mul32(grow[j], negOne)
			}
			r.Data[i] = s
		}
		return r
	default:
		return sumToShapeTake(tensor.Neg(g), shape)
	}
}

// negHadamardPow2 evaluates Neg(Hadamard(ga, Pow(b, -2))) in one pass: the
// per-element power goes through the identical math.Pow(float64) call and
// float32 conversion as tensor.Pow. The final mul32(m, -1) reproduces
// tensor.Neg's v*(-1) multiplication rather than a unary minus: identical
// for every finite value (the exact product is a pure sign flip), but a
// unary minus also flips the sign bit of a NaN, while the hardware
// multiply propagates the NaN unchanged — the legacy chain's behavior.
func negHadamardPow2(ga, b *tensor.Tensor) *tensor.Tensor {
	// The legacy chain ended in a Hadamard, whose 1D-lift the result shape
	// must replicate (see elemwiseGradShape).
	r := tensor.New(elemwiseGradShape(ga.Shape)...)
	for i := range r.Data {
		pb2 := float32(math.Pow(float64(b.Data[i]), -2))
		m := ga.Data[i] * pb2 // rounded before the sign flip, as in the old chain
		r.Data[i] = mul32(m, negOne)
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

// gradMatchesElemwise reports whether a fused elementwise backward may read
// g flat against the parent's element layout — exactly the shapes for which
// the legacy composition tensor.Hadamard(g, deriv) pairs elements in flat
// order (g carrying the node's shape, or its [1, n] lift for 1D nodes).
// Any other shape — reachable only through a manually seeded Grad whose
// shape differs from the node's, which the legacy composition broadcast
// (outer products, scalar replication) — must take the legacy composition
// instead, or the fused loop would silently pair the wrong elements and
// allocate the wrong shape. Normal backward propagation always arrives in
// a matching shape, so the hot path keeps the fused loop.
func gradMatchesElemwise(g, nodeShape []int) bool {
	return sameShape(g, nodeShape) || sameShape(g, elemwiseGradShape(nodeShape))
}

// mul32 computes the float32 product a*b bit-identically to a native
// hardware float32 multiply. For finite operands the exact product fits a
// 48-bit mantissa, which float64 represents precisely, so rounding it to
// float32 is bit-identical to the hardware operation. Writing the product
// through this conversion keeps an explicit rounding node between multiply
// and a following add/sub in the SSA graph, which bars arm64 FMA/FNMS
// fusion across fused-loop statements — fusion would collapse the
// historical two-step rounding (product stored to a tensor, then
// accumulated) into one and drift ~1 ULP.
//
// Non-finite operands (NaN or ±Inf, detected by an all-ones exponent field)
// take the native a*b path instead: the float64 round-trip recanonicalizes
// NaN payloads — the widened multiply settles on one operand's payload and
// the float64→float32 conversion truncates it again — which diverges, bit
// for bit, from the hardware float32 propagation the legacy chain ran. The
// native product is a lone float32 multiply with no adjacent add/sub in
// this expression, and the branch leaves an If/Phi structure in the SSA
// graph that the FMA formation rules do not match, so fusion still cannot
// reach it (verified via go tool compile -S: FMULS+FADDS, never FMADDS).
// Finite products keep the float64 barrier.
func mul32(a, b float32) float32 {
	if math.Float32bits(a)&0x7F800000 == 0x7F800000 ||
		math.Float32bits(b)&0x7F800000 == 0x7F800000 {
		return a * b
	}
	return float32(float64(a) * float64(b))
}

// negOne carries the -1 the negation fuses multiply by. It is deliberately
// a variable rather than a constant: the compiler rewrites a multiply by a
// constant -1 into a sign flip (FNEGS), which is bit-identical for finite
// values but flips a NaN's sign bit — while the legacy tensor.Neg chain
// ran Scale's v*(-1) with a runtime operand, a genuine hardware multiply
// that propagates NaNs unchanged. Loading negOne from memory keeps the
// multiply genuine on the non-finite mul32 path.
var negOne float32 = -1

// MatMul differentiably multiplies two 2D variables: a of shape [m, k]
// times b of shape [k, n] yields a node of shape [m, n]. Panics exactly
// when tensor.MatMul does — if either operand is not 2D or the inner
// dimensions disagree. The backward propagates gradients to both
// operands without materializing transposes.
func MatMul(a, b *Variable) *Variable {
	return newOp(tensor.MatMul(a.Data, b.Data), []*Variable{a, b}, opMatMul)
}

// Add differentiably computes a + b with the tensor package's
// enumerated broadcasting rules: the forward panics on a
// non-broadcastable pair ("not broadcastable"), and two 1D operands
// yield a [1, n] node (the 1D output promotion). The backward reduces
// the output gradient to each operand's own shape with
// tensor.SumToShape, so leaf gradients match leaf shapes even after
// broadcasting.
func Add(a, b *Variable) *Variable {
	return newOp(tensor.Add(a.Data, b.Data), []*Variable{a, b}, opAdd)
}

// Sub differentiably computes a - b with the tensor package's
// enumerated broadcasting rules (forward panics on a non-broadcastable
// pair; two 1D operands yield [1, n]). The backward reduces the output
// gradient to each operand's shape, negated for b, via tensor.SumToShape.
func Sub(a, b *Variable) *Variable {
	return newOp(tensor.Sub(a.Data, b.Data), []*Variable{a, b}, opSub)
}

// Hadamard differentiably computes elementwise a * b with the tensor
// package's enumerated broadcasting rules (forward panics on a
// non-broadcastable pair; [m, 1] against a row vector is the outer
// product; two 1D operands yield [1, n]). The backward multiplies the
// output gradient by the other operand, then reduces it to each
// operand's shape via tensor.SumToShape.
func Hadamard(a, b *Variable) *Variable {
	return newOp(tensor.Hadamard(a.Data, b.Data), []*Variable{a, b}, opHadamard)
}

// Scale multiplies every element of a by the constant s, any shape.
// The backward scales the incoming gradient by the same s.
func Scale(a *Variable, s float32) *Variable {
	out := newOp(tensor.Scale(a.Data, s), []*Variable{a}, opScale)
	out.scalar = s
	return out
}

// Neg negates every element of a, any shape. It is Scale(a, -1).
func Neg(a *Variable) *Variable { return Scale(a, -1) }

// Tanh applies tanh elementwise, any shape. The backward is the fused
// g ⊙ (1 − tanh²(x)).
func Tanh(a *Variable) *Variable {
	return newOp(tensor.Tanh(a.Data), []*Variable{a}, opTanh)
}

// Sigmoid applies the logistic sigmoid elementwise, any shape. The
// backward is the fused g ⊙ σ(x) ⊙ (1 − σ(x)).
func Sigmoid(a *Variable) *Variable {
	return newOp(tensor.Sigmoid(a.Data), []*Variable{a}, opSigmoid)
}

// SigmoidHadamard differentiably computes Hadamard(Sigmoid(z), w) as a single
// fused node. The forward runs the very same two tensor operations the
// composition ran (sigmoid, then the elementwise product), so shapes,
// broadcasting and values are bit-identical; the sigmoid buffer is kept on
// the node (aux) so the backward reuses it instead of recomputing. The
// backward propagates dz = g⊙w⊙s⊙(1−s) in one fused loop and dw = g⊙s
// through the same fused reduction the Hadamard backward used, where the
// composition recorded two op nodes and ran Sigmoid's backward on top of
// Hadamard's (materializing the intermediate g⊙s gradient buffer the fusion
// avoids). Broadcasting and shape semantics follow Hadamard exactly:
// the forward panics when sigmoid(z) and w are not broadcastable. The
// node exists to keep the LTC/CfC inner loop cheap (one node, one fused
// backward, the sigmoid buffer reused); plain code can compose Sigmoid
// and Hadamard instead.
func SigmoidHadamard(z, w *Variable) *Variable {
	s := tensor.Sigmoid(z.Data)
	out := newOp(tensor.Hadamard(s, w.Data), []*Variable{z, w}, opSigmoidHadamard)
	out.aux = s
	return out
}

// Exp applies exp elementwise, any shape. The backward multiplies the
// incoming gradient by the (stored) forward output.
func Exp(a *Variable) *Variable {
	return newOp(tensor.Exp(a.Data), []*Variable{a}, opExp)
}

// Log applies natural log elementwise, any shape; the domain is not
// checked (non-positive elements give NaN/-Inf forward and backward,
// as float32 arithmetic dictates). The backward is g ⊙ (1/x).
func Log(a *Variable) *Variable {
	return newOp(tensor.Log(a.Data), []*Variable{a}, opLog)
}

// Pow raises every element of a to the constant power p, any shape.
// The backward is g ⊙ p·x^(p−1); for p == 0 it is exactly zero
// everywhere (evaluated directly, not as 0·x^-1, which would be NaN at
// x == 0).
func Pow(a *Variable, p float32) *Variable {
	out := newOp(tensor.Pow(a.Data, p), []*Variable{a}, opPow)
	out.scalar = p
	return out
}

// Softplus applies log(1 + e^x) elementwise, any shape, stably (see
// tensor.Softplus). The backward is g ⊙ σ(x).
func Softplus(a *Variable) *Variable {
	return newOp(tensor.Softplus(a.Data), []*Variable{a}, opSoftplus)
}

// Abs applies |x| elementwise, any shape. It is not differentiable at
// 0; the backward takes the gradient there as 0 (g ⊙ sign(x) with
// sign(0) = 0).
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

// ConcatCol concatenates 2D variables along the column axis: inputs of
// shapes [m, n1], [m, n2], ... yield a node of shape [m, n1+n2+...].
// Panics if called with no inputs, or per tensor.ConcatCol if any input
// is not 2D or the row counts differ. The backward slices each input's
// gradient back out of the output gradient.
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

// SliceCol differentiably extracts columns [from, to) of a 2D variable
// as a node of shape [m, to-from]. Panics per tensor.SliceCol if a is
// not 2D or the range is invalid or empty. The backward writes the
// gradient back into the corresponding columns of a zero-padded buffer
// the size of a.
func SliceCol(a *Variable, from, to int) *Variable {
	out := newOp(tensor.SliceCol(a.Data, from, to), []*Variable{a}, opSliceCol)
	out.from, out.to = from, to
	return out
}

// Col differentiably extracts column i of a 2D variable as a node of
// shape [m, 1]. It is SliceCol(a, i, i+1) and panics under the same
// conditions (a not 2D, i out of range).
func Col(a *Variable, i int) *Variable { return SliceCol(a, i, i+1) }

// SliceRow differentiably extracts row i of a 2D variable as a node of
// shape [1, n]. Panics per tensor.SliceRow if a is not 2D or i is
// outside [0, a.Data.Rows()). The backward writes the gradient back
// into row i of a zero-padded buffer the size of a.
func SliceRow(a *Variable, i int) *Variable {
	out := newOp(tensor.SliceRow(a.Data, i), []*Variable{a}, opSliceRow)
	out.from = i
	return out
}

// SumAll sums every element of a (any shape) into a scalar node of
// shape [1]. The backward broadcasts the scalar gradient to a's shape.
func SumAll(a *Variable) *Variable {
	return newOp(tensor.SumAll(a.Data), []*Variable{a}, opSumAll)
}

// MeanAll averages every element of a (any shape) into a scalar node
// of shape [1]. Panics on an empty tensor, like tensor.MeanAll. The
// backward broadcasts g/size to a's shape.
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

// LogSoftmaxRows applies the numerically stable log-softmax to each
// row of a 2D variable, yielding a node of the same shape [m, n].
// Panics per tensor.LogSoftmaxRows if a is not 2D. The backward is the
// fused g − softmax(row) ⊙ rowsum(g); a manually seeded Grad whose
// shape differs from the node's takes the legacy (shape-strict)
// reduction path and may panic on 1D seeds, exactly as the pre-fusion
// composition did.
func LogSoftmaxRows(a *Variable) *Variable {
	return newOp(tensor.LogSoftmaxRows(a.Data), []*Variable{a}, opLogSoftmaxRows)
}
