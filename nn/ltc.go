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
	// only through the erevRows* row-view constants below.
	erev  *autograd.Variable // [units, units]
	sErev *autograd.Variable // [inDim, units]

	// Construction-time graph constants shared by every Step. maskS/maskR
	// fold the wiring into the positivity-constrained weights with one
	// matrix Hadamard per Step instead of one masked multiply per synapse.
	//
	// The presynaptic-axis contraction (summing the per-presynaptic
	// activation blocks, scaling by reversal potentials for the numerator)
	// used to run as MatMuls against dense [pre*units, units] indicator
	// matrices — an O(units^3) float32 materialization that put a memory
	// cliff in both the constructor and the load path (red team sweep F1;
	// the load side was bandaged by maxUnits until this root-cause fix,
	// technical-debt item #14). The contraction is now the sparse fold of
	// contract(): its persistent state is ident (one units×units identity),
	// zeroV (one scalar), the erevRows* row views (sharing erev/sErev
	// storage, no extra floats) and the plan* term lists (O(wiring nnz)
	// int32s) — O(units^2) in all, and no Step ever materializes a
	// [units^2, units] tensor.
	maskS, maskR *autograd.Variable

	// ident is the units×units identity matrix. Each contraction ends in a
	// MatMul against it: the forward is a value-preserving copy, and the
	// backward runs the genuine tensor.MatMulTransB, whose av==0 skip
	// normalizes zero incoming gradients to +0 bit-for-bit as the former
	// indicator MatMul's backward did (see contract).
	ident *autograd.Variable
	// zeroV is the scalar +0 seeding every contraction fold — the analogue
	// of MatMul's fresh zero-filled accumulator buffer (see contract).
	zeroV *autograd.Variable
	// erevRowsR/S expose row i of the reversal potentials as a [1, units]
	// graph constant: the per-presynaptic numerator coefficients. The row
	// tensors share the erev/sErev Data arrays, so Load's in-place copy of
	// the streamed polarities is picked up automatically, with no rebuild
	// (the old design re-materialized a dense indicator here per load).
	erevRowsR, erevRowsS []*autograd.Variable
	// planR/planS record, per postsynaptic neuron j, the ascending list of
	// wired presynaptic indices (mask[i, j] == 1): the terms the contraction
	// sums, as O(nnz) metadata (see synapsePlan and contract).
	planR, planS *synapsePlan

	// epsV is the membrane epsilon as a graph constant, built once.
	epsV *autograd.Variable

	wiring *Wiring
}

// NewLTC creates an LTC cell. wiring may be nil, meaning FullyConnected;
// otherwise its sensory mask must have shape [inDim, units] and its
// recurrent mask [units, units]. unfolds is the number of ODE solver
// steps per RNN step (6 in the reference). Parameter init ranges follow
// the reference implementation. rng must be non-nil and supplies every
// initialization draw, so a fixed seed reproduces a cell bit for bit.
// Panics if inDim < 1 or units < 1, if unfolds < 1, or if a non-nil
// wiring's mask shapes do not match [inDim, units] / [units, units].
// (LoadLTC additionally caps unfolds, units and inDim on the load path
// only; the constructor accepts any values >= 1 — see doc/persistence.md.)
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
		erev:   autograd.Var(erevInit(units, units)),
		sErev:  autograd.Var(erevInit(inDim, units)),
		maskS:  autograd.Const(wiring.Sensory()),
		maskR:  autograd.Const(wiring.Recurrent()),
		ident:  autograd.Const(identityMat(units)),
		zeroV:  autograd.Const(tensor.FromData([]float32{0}, 1)),
		epsV:   autograd.Const(tensor.FromData([]float32{ltcEps}, 1)),
		wiring: wiring,
		planR:  newSynapsePlan(wiring.recurrentMask),
		planS:  newSynapsePlan(wiring.sensoryMask),
	}
	// Numerator coefficients are row views of the (already drawn) reversal
	// potentials: O(pre*units) constants sharing the erev storage, replacing
	// the dense [pre*units, units] reversal indicators.
	c.erevRowsR = erevRowViews(c.erev.Data)
	c.erevRowsS = erevRowViews(c.sErev.Data)
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

// identityMat builds the n×n identity matrix: the terminal operand of the
// sparse contraction's normalizing MatMul (see contract).
func identityMat(n int) *tensor.Tensor {
	t := tensor.New(n, n)
	for i := 0; i < n; i++ {
		t.Data[i*n+i] = 1
	}
	return t
}

// erevRowViews exposes each row of a [pre, units] reversal-potential matrix
// as a [1, units] graph constant, for the per-presynaptic numerator
// coefficients of contract. The row tensors deliberately SHARE rows' Data
// arrays (read-only: forwards only read constant values, and backwards write
// gradients into separate buffers), so Save/Load's in-place overwrite of the
// cell's erev storage is reflected in the constants without any rebuild —
// the old design re-materialized a dense [pre*units, units] indicator here.
func erevRowViews(rows *tensor.Tensor) []*autograd.Variable {
	pre, units := rows.Rows(), rows.Cols()
	vs := make([]*autograd.Variable, pre)
	for i := range vs {
		vs[i] = autograd.Const(&tensor.Tensor{Shape: []int{1, units}, Data: rows.Data[i*units : (i+1)*units]})
	}
	return vs
}

// synapsePlan records the wired synapse topology per postsynaptic neuron:
// cols[j] holds the presynaptic indices i with mask[i, j] == 1, ascending —
// exactly the per-column (i, coefficient) term list the contraction sums
// (the numerator's coefficient of term (i, j) is erev[i, j]; the
// denominator's is 1). Storage is O(nnz(mask)) int32s plus one slice header
// per postsynaptic neuron — against the O(pre*units^2) float32 extent of the
// dense indicator matrices this replaced (red team F1's memory cliff,
// technical-debt item #14): at units=1024 fully wired, 4 MiB of term lists
// versus 8 GiB of indicators.
//
// The plan documents and counts the contraction's terms; contract itself
// folds whole presynaptic rows rather than walking the lists, because
// unwired synapses are arithmetically neutral: a masked block entry is
// exactly +0 (sigmoid >= 0 times a zero-masked weight), and +0 is an
// additive identity, so the all-rows fold sums precisely the wired terms,
// bit for bit (see contract's ordering-equivalence proof).
type synapsePlan struct {
	cols [][]int32
}

// newSynapsePlan builds the per-postsynaptic term lists from a binary
// [pre, post] wiring mask, in ascending presynaptic order.
func newSynapsePlan(mask *tensor.Tensor) *synapsePlan {
	pre, post := mask.Rows(), mask.Cols()
	counts := make([]int, post)
	for i := 0; i < pre; i++ {
		row := mask.Data[i*post : (i+1)*post]
		for j, v := range row {
			if v == 1 {
				counts[j]++
			}
		}
	}
	cols := make([][]int32, post)
	for j := range cols {
		cols[j] = make([]int32, 0, counts[j])
	}
	for i := 0; i < pre; i++ {
		row := mask.Data[i*post : (i+1)*post]
		for j, v := range row {
			if v == 1 {
				cols[j] = append(cols[j], int32(i))
			}
		}
	}
	return &synapsePlan{cols: cols}
}

// terms returns the total number of wired synapses: the sum of the per-column
// list lengths, i.e. the number of nonzeros in the wiring mask.
func (p *synapsePlan) terms() int {
	n := 0
	for _, c := range p.cols {
		n += len(c)
	}
	return n
}

// StateSize returns the hidden state dimension.
func (c *LTC) StateSize() int { return c.units }

// Parameters returns the cell's 13 trainable variables in a fixed
// order: gleak, vleak, cm, mu, sigma, w, sMu, sSigma, sW, inW, inB,
// outW, outB. The order is frozen: it is the cell's stream order inside
// SaveLTC and the positional key for serialize.WriteParameters /
// optimizer.SaveState. The reversal potentials erev/sErev are fixed
// constants and intentionally excluded (training them would flip
// synapse polarity); SaveLTC persists them anyway, so LoadLTC
// reproduces the cell exactly.
func (c *LTC) Parameters() []*autograd.Variable {
	return []*autograd.Variable{
		c.gleak, c.vleak, c.cm,
		c.mu, c.sigma, c.w,
		c.sMu, c.sSigma, c.sW,
		c.inW, c.inB, c.outW, c.outB,
	}
}

// Step advances the cell by one RNN step, integrating the membrane ODE
// over the time span ts in c.unfolds semi-implicit Euler substeps. x is
// [batch, inDim], h is [batch, units] or nil for a zero initial state.
// It returns the affinely mapped output v⊙outW+outB with shape
// [batch, units] and the new raw membrane state v with shape
// [batch, units]. Panics if ts is not positive and finite (NaN, +/-Inf,
// zero and negative values are all rejected), or if x/h do not have the
// expected rank and widths (the tensor-layer contract).
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
	numS, denS := c.synapses(inputs, c.sMu, c.sSigma, sWm, c.erevRowsS)

	// Membrane-update numerator terms that stay constant across the unfolds.
	numConst := autograd.Add(autograd.Hadamard(gleak, c.vleak), numS)

	// The ODE unfolds run as one fused kernel (nn/ltc_fused.go): a single
	// graph node whose forward replays the per-unfold synapse blocks,
	// contraction folds and membrane update (the denBase chain
	// ((cmT + gleak) + denS) + eps included) with identical rounding
	// boundaries, and whose hand-written VJP replays the former subgraph's
	// backward accumulation order contribution for contribution. The
	// node count per Step drops from O(unfolds × units) to O(1).
	v := c.fusedUnfolds(cmT, h, numConst, gleak, denS, wM)

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

// synapses accumulates numerator and denominator synaptic currents from a
// presynaptic source (inputs or previous state), vectorized per presynaptic
// neuron rather than per synapse pair. For presynaptic neuron i the whole
// row of postsynaptic targets is a single [batch, units] block,
//
//	block_i = sigmoid(sigma_i ⊙ (pre[:, i] − mu_i)) ⊙ wm_i,
//
// where wm carries the positivity-constrained weight already folded with
// the wiring mask (see Step). contract then reduces the presynaptic axis:
//
//	den[:, j] = Σ_i block_i[:, j]
//	num[:, j] = Σ_i block_i[:, j] · erev[i, j]
//
// with erevRows carrying the (fixed) reversal rows. The sensory path is
// its only caller (the recurrent path runs inside the fused unfold kernel,
// nn/ltc_fused.go); synapsesRows factors the per-presynaptic block build
// out of the row slicing.
func (c *LTC) synapses(
	pre, mu, sigma, wm *autograd.Variable, erevRows []*autograd.Variable,
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
	return c.synapsesRows(pre, muRs, sigRs, wmRs, erevRows)
}

// synapsesRows builds one activation block per presynaptic neuron from the
// pre-sliced parameter rows, then the sparse contraction of contract. It
// serves the sensory path (the recurrent blocks live in the fused kernel).
func (c *LTC) synapsesRows(
	pre *autograd.Variable,
	muRs, sigRs, wmRs []*autograd.Variable,
	erevRows []*autograd.Variable,
) (num, den *autograd.Variable) {
	blocks := make([]*autograd.Variable, len(muRs))
	for i := range blocks {
		preCol := autograd.Col(pre, i) // [batch, 1]
		z := autograd.Hadamard(sigRs[i], autograd.Sub(preCol, muRs[i]))
		// One fused sigmoid(z)⊙wm node per presynaptic neuron instead of the
		// Sigmoid+Hadamard pair: identical forward bits, one backward pass.
		blocks[i] = autograd.SigmoidHadamard(z, wmRs[i])
	}
	return c.contract(blocks, erevRows)
}

// contract reduces the per-presynaptic activation blocks into the numerator
// and denominator synaptic currents — the sparse replacement for the former
// MatMul(flat, indicator) reductions, which materialized dense
// [pre*units, units] indicator matrices: an O(units^3) float32 cliff that
// made NewLTC(4, 1024, …) allocate ~8 GiB of indicators (red team sweep F1,
// technical-debt item #14). Persistent contraction state is now ident (one
// units×units matrix), zeroV (one scalar), the erevRows views (sharing erev
// storage) and the per-cell synapsePlans — O(units^2) — and the per-step
// graph grows by O(pre) nodes, never by a [units^2, units] tensor.
//
// The reduction is bit-identical to the indicator MatMul it replaces — in
// forward AND backward — because it replicates that MatMul's four defining
// behaviors exactly:
//
//  1. Ascending presynaptic order. MatMul column j accumulates its nonzeros
//     in ascending flat index k = i*units+j, i.e. ascending i — the same
//     term sequence planR/planS record per postsynaptic neuron. The fold
//     below visits the blocks in index order, so each column j accumulates
//     the very same (i, coefficient) pairs in the very same order. Unwired
//     synapses are absent from the lists yet present in the fold — and
//     arithmetically neutral: a masked block entry is exactly +0 (sigmoid
//     >= 0 times a zero-masked weight), and adding +0 is the identity, so
//     the all-rows fold sums precisely the wired terms.
//  2. Operand order. The numerator's per-term product is block·erev — the
//     activation block element on the left, the coefficient on the right —
//     exactly as MatMul evaluated av·brow[j]. IEEE multiplication agrees
//     bitwise here (erev is +/-1), but the order is preserved regardless.
//  3. Zero-skip. MatMul skips left operands that are zero, -0 included
//     (tensor.MatMul's av==0 branch — the source of the F-RT1 +0 behavior,
//     deliberately kept, not "fixed"). Forward, the fold reproduces the
//     skip by arithmetic: a zero block makes the term +/-0, and acc + (+0)
//     = acc and acc + (-0) = acc for every accumulator value the fold can
//     hold (it starts at +0 and round-to-nearest never produces -0 from a
//     +0-seeded sum), so skipped and added zeros land on identical bits.
//     Backward, the terminal MatMul against ident runs the genuine
//     tensor.MatMulTransB, whose av==0 branch normalizes a zero incoming
//     gradient to a +0 contribution — bit for bit what the indicator
//     MatMul's backward computed for flat's gradient. That normalization
//     covers the multi-source path, where both contractions end in a
//     normalizing MatMul (any -0 the numerator's Hadamard backward
//     contributes sums with the denominator's +0: (+0)+(-0) = +0). It
//     does not cover the single-source corner — inDim=1 or units=1 —
//     where den takes the raw-block shortcut below: with a zero-valued
//     incoming gradient and an erev of -1, the numerator fold's Hadamard
//     backward forms (-0)*(-1) = -0 and no normalizing MatMul washes the
//     sign out, so that zero gradient keeps its sign bit (red team F9-1;
//     four differential configurations, the value 0 in every case).
//     Nothing downstream can observe it: p - LR*(-0) is bit-identical to
//     p for any p that is not itself -0, optimizer accumulators normalize
//     (+0)+(-0) = +0, and the red team measured zero training-trajectory
//     divergence — a deliberately accepted sign-bit corner, not worth a
//     per-step normalizing MatMul on the single-source den path.
//  4. +0 accumulator. The fold seeds from zeroV (scalar +0), mirroring
//     MatMul's fresh zero-filled output buffer. This is what keeps a fully
//     masked postsynaptic column at +0 (0x00000000) instead of -0 (red
//     team F-RT1): an empty term list sums to the seed, exactly as the
//     all-skipped MatMul column did.
//
// den keeps the historical single-source shortcut: with one presynaptic row
// the denominator indicator is the identity, so den is the raw block — no
// fold, no normalizing MatMul — while num still folds and normalizes, both
// exactly as before.
func (c *LTC) contract(blocks, erevRows []*autograd.Variable) (num, den *autograd.Variable) {
	if len(blocks) == 1 {
		den = blocks[0]
	} else {
		den = autograd.Add(c.zeroV, blocks[0])
		for i := 1; i < len(blocks); i++ {
			den = autograd.Add(den, blocks[i])
		}
		// The identity MatMul: a value-preserving copy forward (x·1 is exact
		// for nonzero x; zero x is skipped and stays +0), and backward the
		// av==0 gradient skip the indicator MatMul ran (constraint 3).
		den = autograd.MatMul(den, c.ident)
	}
	num = autograd.Add(c.zeroV, autograd.Hadamard(blocks[0], erevRows[0]))
	for i := 1; i < len(blocks); i++ {
		num = autograd.Add(num, autograd.Hadamard(blocks[i], erevRows[i]))
	}
	num = autograd.MatMul(num, c.ident)
	return num, den
}
