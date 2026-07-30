package nn

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/tensor"
)

// ltcEps is the membrane-update epsilon guarding the num/den division.
const ltcEps = 1e-8

// LTC is a Liquid Time-Constant cell following Hasani et al. (2021) and the
// reference implementation in mlech26l/ncps. The membrane ODE
//
//	dv/dt = -(1/tau + f(v, I)) * v + f(v, I) * A
//
// is integrated with the semi-implicit Euler scheme of the reference:
//
//	num = cm/dt * v + gleak * vleak + sum_j(act_j * erev_j)
//	den = cm/dt     + gleak        + sum_j(act_j)
//	v  <- num / (den + eps)
//
// where each synapse activates as act = w * sigmoid(sigma * (v_pre - mu)).
// Positivity of cm, gleak and synaptic weights is enforced implicitly with a
// softplus (the reference's implicit_param_constraints mode), so no weight
// clipping hook is needed inside the optimizer.
type LTC struct {
	inDim, units int
	unfolds      int
	eps          float32

	gleak, vleak, cm *autograd.Variable // [units]
	mu, sigma, w     *autograd.Variable // [units, units] recurrent synapses
	sMu, sSigma, sW  *autograd.Variable // [inDim, units] sensory synapses
	inW, inB         *autograd.Variable // [inDim]
	outW, outB       *autograd.Variable // [units]

	// Reversal potentials are fixed random +/-1 constants drawn once at
	// construction time, mirroring the reference wiring. They are NOT
	// trainable and are deliberately absent from Parameters(): learning them
	// would flip synapses between excitatory and inhibitory polarity and
	// degrade the LTC into an ordinary plastic network. They enter the graph
	// only through the numReduce* constants below.
	erev  *autograd.Variable // [units, units]
	sErev *autograd.Variable // [inDim, units]

	// Construction-time graph constants shared by every Step. maskS/maskR
	// fold the wiring into the positivity-constrained weights with one
	// matrix Hadamard per Step instead of one masked multiply per synapse.
	// denReduce*/numReduce* are sparse [pre*units, units] indicator matrices
	// whose MatMul contraction sums the per-presynaptic activation blocks;
	// numReduce* additionally scales each synapse by its (constant) reversal
	// potential. Backward accumulates gradients into these leaves that
	// are never read — exactly as the old per-synapse mask constants did,
	// but at a handful of nodes per Step instead of one per synapse per
	// unfold.
	maskS, maskR           *autograd.Variable
	denReduceS, denReduceR *autograd.Variable
	numReduceS, numReduceR *autograd.Variable

	// epsV is the membrane epsilon as a graph constant, built once.
	epsV *autograd.Variable

	wiring *Wiring
}

// NewLTC creates an LTC cell. wiring may be nil, meaning FullyConnected;
// otherwise its sensory mask must have shape [inDim, units] and its
// recurrent mask [units, units]. unfolds is the number of ODE solver steps
// per RNN step (6 in the reference). Parameter init ranges follow the
// reference implementation.
func NewLTC(inDim, units int, wiring *Wiring, unfolds int, rng *rand.Rand) *LTC {
	if inDim < 1 || units < 1 {
		panic(fmt.Sprintf("nn.NewLTC: invalid dims in=%d units=%d", inDim, units))
	}
	if unfolds < 1 {
		panic(fmt.Sprintf("nn.NewLTC: unfolds must be >= 1, got %d", unfolds))
	}
	if wiring == nil {
		wiring = FullyConnected(inDim, units)
	}
	// Shape-only validation: comparing Shape fields allocates nothing, unlike
	// building reference tensors to diff against (which used to allocate an
	// inDim*units tensor just to say "no").
	if !shapeIs(wiring.sensoryMask.Shape, inDim, units) ||
		!shapeIs(wiring.recurrentMask.Shape, units, units) {
		panic(fmt.Sprintf("nn.NewLTC: wiring mask shapes %v and %v do not match [inDim=%d, units=%d]",
			wiring.sensoryMask.Shape, wiring.recurrentMask.Shape, inDim, units))
	}
	uniform := func(lo, hi float32, shape ...int) *autograd.Variable {
		return autograd.Var(tensor.Uniform(rng, lo, hi, shape...))
	}
	// Reversal potentials are random +/- 1, as in the reference wiring.
	erevInit := func(shape ...int) *tensor.Tensor {
		t := tensor.New(shape...)
		for i := range t.Data {
			if rng.Intn(2) == 0 {
				t.Data[i] = -1
			} else {
				t.Data[i] = 1
			}
		}
		return t
	}
	c := &LTC{
		inDim: inDim, units: units, unfolds: unfolds, eps: ltcEps,
		gleak:  uniform(0.001, 1, units),
		vleak:  uniform(-0.2, 0.2, units),
		cm:     uniform(0.4, 0.6, units),
		mu:     uniform(0.3, 0.8, units, units),
		sigma:  uniform(3, 8, units, units),
		w:      uniform(0.001, 1, units, units),
		sMu:    uniform(0.3, 0.8, inDim, units),
		sSigma: uniform(3, 8, inDim, units),
		sW:     uniform(0.001, 1, inDim, units),
		inW:    autograd.Var(tensor.New(inDim).OnesLike()),
		inB:    autograd.Var(tensor.New(inDim)),
		outW:   autograd.Var(tensor.New(units).OnesLike()),
		outB:   autograd.Var(tensor.New(units)),
		// The reversal potentials must keep drawing the rng at exactly this
		// point: the draw order fixes same-seed initialization, so the
		// rng-free graph constants below come after.
		erev:       autograd.Var(erevInit(units, units)),
		sErev:      autograd.Var(erevInit(inDim, units)),
		maskS:      autograd.Const(wiring.Sensory()),
		maskR:      autograd.Const(wiring.Recurrent()),
		denReduceS: autograd.Const(sumIndicator(inDim, units)),
		denReduceR: autograd.Const(sumIndicator(units, units)),
		epsV:       autograd.Const(tensor.FromData([]float32{ltcEps}, 1)),
		wiring:     wiring,
	}
	// Numerator reductions bake in the (already drawn) reversal potentials.
	c.numReduceR = autograd.Const(reversalIndicator(c.erev.Data.Data, units, units))
	c.numReduceS = autograd.Const(reversalIndicator(c.sErev.Data.Data, inDim, units))
	return c
}

// shapeIs reports whether sh equals want, without allocating.
func shapeIs(sh []int, want ...int) bool {
	if len(sh) != len(want) {
		return false
	}
	for i := range sh {
		if sh[i] != want[i] {
			return false
		}
	}
	return true
}

// sumIndicator builds the sparse [n*m, m] reduction matrix R with
// R[i*m+j, j] = 1. A MatMul against the n per-presynaptic [batch, m]
// activation blocks laid out side by side (shape [batch, n*m]) then
// contracts the presynaptic axis: out[:, j] = Σ_i blocks[:, i*m+j]. MatMul
// skips zero entries, so the contraction costs O(batch·n·m) despite R's
// [n*m, m] extent.
func sumIndicator(n, m int) *tensor.Tensor {
	r := tensor.New(n*m, m)
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			r.Data[(i*m+j)*m+j] = 1
		}
	}
	return r
}

// reversalIndicator is sumIndicator scaled per synapse by the (constant)
// reversal potential: R[i*m+j, j] = erev[i*m+j]. The same MatMul
// contraction then yields the numerator current Σ_i act_i · erev_i, so the
// fixed +/-1 potentials never appear as per-synapse graph nodes.
func reversalIndicator(erev []float32, n, m int) *tensor.Tensor {
	r := tensor.New(n*m, m)
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			r.Data[(i*m+j)*m+j] = erev[i*m+j]
		}
	}
	return r
}

// StateSize returns the hidden state dimension.
func (c *LTC) StateSize() int { return c.units }

// Parameters returns the trainable variables of the cell. The reversal
// potentials erev/sErev are fixed constants and intentionally excluded.
func (c *LTC) Parameters() []*autograd.Variable {
	return []*autograd.Variable{
		c.gleak, c.vleak, c.cm,
		c.mu, c.sigma, c.w,
		c.sMu, c.sSigma, c.sW,
		c.inW, c.inB, c.outW, c.outB,
	}
}

// Step advances the cell by one RNN step, integrating the ODE over the time
// span ts (which must be positive and finite; NaN and +/-Inf are rejected).
// x is [batch, inDim], h is [batch, units] or nil for a zero initial state.
// It returns the (affinely mapped) output and the new raw state.
func (c *LTC) Step(x, h *autograd.Variable, ts float64) (out, hNew *autograd.Variable) {
	// NaN-aware positivity check: NaN > 0 is false, so NaN panics here too.
	// +Inf passes `ts > 0` but would silently integrate over an infinite time
	// span (the "infinite-time steady state"), which callers rarely intend;
	// reject both infinities explicitly, on the same panic path.
	if !(ts > 0) || math.IsInf(ts, 0) {
		panic(fmt.Sprintf("nn.LTC.Step: ts must be positive and finite, got %v", ts))
	}
	batch := x.Data.Rows()
	if h == nil {
		h = autograd.Var(tensor.New(batch, c.units))
	}

	// Affine input mapping.
	inputs := autograd.Add(autograd.Hadamard(x, c.inW), c.inB)

	// Positivity-constrained parameters (softplus), with the wiring mask
	// folded into the weights once per Step — one matrix Hadamard instead of
	// one masked multiply per synapse per unfold.
	cmT := c.scaledCapacitance(ts)
	gleak := autograd.Softplus(c.gleak)
	sWm := autograd.Hadamard(autograd.Softplus(c.sW), c.maskS) // [inDim, units]
	wM := autograd.Hadamard(autograd.Softplus(c.w), c.maskR)   // [units, units]

	// Sensory (input) synapses are loop-invariant over the ODE unfolds.
	numS, denS := c.synapses(inputs, c.sMu, c.sSigma, sWm, c.denReduceS, c.numReduceS)

	// Recurrent parameter rows are Step-invariant: slice them once and reuse
	// the rows across every ODE unfold below.
	muRs := c.rows(c.mu)
	sigRs := c.rows(c.sigma)
	wmRs := c.rows(wM)

	// Membrane-update terms that stay constant across the unfolds. eps joins
	// denBase here (it is a cell constant), saving one Add per unfold; the
	// regrouping changes only float32 association, by O(1e-7) relatively.
	numConst := autograd.Add(autograd.Hadamard(gleak, c.vleak), numS)
	denBase := autograd.Add(autograd.Add(autograd.Add(cmT, gleak), denS), c.epsV)

	v := h
	for t := 0; t < c.unfolds; t++ {
		numR, denR := c.synapsesRows(v, muRs, sigRs, wmRs, c.denReduceR, c.numReduceR)
		// num = cm_t .* v + gleak .* vleak + synapses
		num := autograd.Add(autograd.Add(autograd.Hadamard(cmT, v), numConst), numR)
		v = autograd.Div(num, autograd.Add(denBase, denR))
	}

	out = autograd.Add(autograd.Hadamard(v, c.outW), c.outB)
	return out, v
}

// scaledCapacitance builds cm_t = softplus(cm) * unfolds/ts, the ODE's cm/dt
// term with dt = ts/unfolds. The scalar time scale is computed in float64
// and clamped to the finite float32 domain before being converted back: a
// tiny ts (e.g. 1e-40, the hallmark of the LTC's variable-step regime) used
// to overflow the float32 division to +Inf, and Inf*0 in the state update
// then turned every output NaN.
//
// Clamping the scale alone is not enough: softplus(cm)*scale can still
// exceed MaxFloat32 elementwise. We therefore cap the product with a smooth
// differentiable min, cap(sp) = sp - softplus(sp - hi), hi = MaxFloat32/scale.
// While sp << hi (every sane ts) softplus(sp - hi) underflows to exactly 0,
// so cap(sp) is bit-identical to sp and the ODE algebra is untouched; the
// cap only engages where the unscaled product would overflow. The tiny
// relative headroom on hi absorbs float32 rounding so cm_t is always finite.
func (c *LTC) scaledCapacitance(ts float64) *autograd.Variable {
	scale64 := float64(c.unfolds) / ts
	if scale64 > math.MaxFloat32 {
		scale64 = math.MaxFloat32
	}
	hi64 := math.MaxFloat32 / scale64 / 1.0001
	if hi64 > math.MaxFloat32 {
		hi64 = math.MaxFloat32
	}
	hiV := autograd.Const(tensor.FromData([]float32{float32(hi64)}, 1))
	sp := autograd.Softplus(c.cm)
	capped := autograd.Sub(sp, autograd.Softplus(autograd.Sub(sp, hiV)))
	return autograd.Scale(capped, float32(scale64))
}

// rows slices every row of m as a [1, cols] variable. The rows of the
// recurrent mu/sigma and the masked-weight matrix are Step-invariant, so
// slicing them once per Step — instead of once per presynaptic neuron per
// ODE unfold — keeps units*(unfolds-1) SliceRow nodes per matrix out of the
// graph.
func (c *LTC) rows(m *autograd.Variable) []*autograd.Variable {
	rs := make([]*autograd.Variable, m.Data.Rows())
	for i := range rs {
		rs[i] = autograd.SliceRow(m, i)
	}
	return rs
}

// synapses accumulates numerator and denominator synaptic currents from a
// presynaptic source (inputs or previous state), vectorized per presynaptic
// neuron rather than per synapse pair. For presynaptic neuron i the whole
// row of postsynaptic targets is a single [batch, units] block,
//
//	block_i = sigmoid(sigma_i ⊙ (pre[:, i] − mu_i)) ⊙ wm_i,
//
// where wm carries the positivity-constrained weight already folded with
// the wiring mask (see Step). The blocks concatenate into [batch, n·units]
// and two MatMuls against the sparse construction-time indicators contract
// the presynaptic axis:
//
//	den[:, j] = Σ_i block_i[:, j]
//	num[:, j] = Σ_i block_i[:, j] · erev[i, j]
//
// (the fixed reversal potentials are baked into numReduce). MatMul skips
// zero entries, and its fresh-buffer left-to-right accumulation over i
// reproduces the old per-synapse Add chain bit-for-bit — mask being {0, 1}
// makes w·mask exactly w or +0, so no rounding order changes either. Graph
// size drops from O(pre²·unfolds) nodes to O(pre·unfolds). The sensory and
// recurrent paths share this single calling convention; the recurrent path
// additionally reuses one set of pre-sliced rows across all unfolds via
// synapsesRows.
func (c *LTC) synapses(
	pre, mu, sigma, wm, denReduce, numReduce *autograd.Variable,
) (num, den *autograd.Variable) {
	n := pre.Data.Cols()
	muRs := make([]*autograd.Variable, n)
	sigRs := make([]*autograd.Variable, n)
	wmRs := make([]*autograd.Variable, n)
	for i := 0; i < n; i++ {
		muRs[i] = autograd.SliceRow(mu, i)
		sigRs[i] = autograd.SliceRow(sigma, i)
		wmRs[i] = autograd.SliceRow(wm, i)
	}
	return c.synapsesRows(pre, muRs, sigRs, wmRs, denReduce, numReduce)
}

// synapsesRows is the per-unfold inner loop: one activation block per
// presynaptic neuron from the pre-sliced parameter rows, then the two
// indicator MatMul reductions.
func (c *LTC) synapsesRows(
	pre *autograd.Variable,
	muRs, sigRs, wmRs []*autograd.Variable,
	denReduce, numReduce *autograd.Variable,
) (num, den *autograd.Variable) {
	blocks := make([]*autograd.Variable, len(muRs))
	for i := range blocks {
		preCol := autograd.Col(pre, i) // [batch, 1]
		z := autograd.Hadamard(sigRs[i], autograd.Sub(preCol, muRs[i]))
		blocks[i] = autograd.Hadamard(autograd.Sigmoid(z), wmRs[i])
	}
	flat := blocks[0]
	if len(blocks) > 1 {
		flat = autograd.ConcatCol(blocks...)
		den = autograd.MatMul(flat, denReduce)
	} else {
		// A single presynaptic source: denReduce is the m×m identity, so the
		// contraction is a copy the MatMul would pay for bit-identically.
		den = flat
	}
	num = autograd.MatMul(flat, numReduce)
	return num, den
}
