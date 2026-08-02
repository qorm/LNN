package nn

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/tensor"
)

// CfC is a Closed-form Continuous-time cell, the successor to the LTC of
// Hasani et al. (2021): it keeps the LTC membrane ODE but replaces numerical
// ODE integration with the closed-form solution derived in Hasani et al.
// (2022), "Closed-form continuous-time neural networks", Nature Machine
// Intelligence 4, 992-1003 (arXiv:2106.13898), Theorem 1 / Lemma 1 / Eq. (8)
// and the LTC-to-closed-form compilation of their Algorithm 1. Reference
// implementations: mlech26l/ncps (ncps/torch/cfc_cell.py) and raminmh/CfC
// (torch_cfc.py).
//
// The membrane ODE is the very one this library's LTC integrates numerically:
//
//	cm * dv/dt = -gleak*(v - vleak) + sum_j act_j * (erev_j - v)
//	           = -G*(v - A),   G = gleak + sum_j act_j,
//	                           A = (gleak*vleak + sum_j act_j*erev_j) / G,
//
// with each synapse activating exactly as in the LTC reference,
// act = softplus(w) * sigmoid(sigma * (v_pre - mu)), gated by the wiring mask.
// Freezing the activations over the step (the Lemma-1 approximation: the
// input integral is evaluated at the current input) makes the ODE linear in v
// with the exact closed-form solution over a time span ts:
//
//	v_new = A + (v - A) * exp(-kappa*ts),   kappa = G/cm,
//	      = v + (A - v) * F(B),             B = kappa*ts,  F(B) = 1 - exp(-B).
//
// v_new is a convex combination of v and A (F in [0, 1]), so the state stays
// bounded without any solver unfolds. The official CfC cells ship an
// MLP-backbone variant (ff1/ff2 heads behind a sigmoid time gate) instead;
// this library's convention is synapses + wiring + fixed +/-1 reversal
// potentials (see ltc.go), so the CfC here is that same LTC parameterization
// driven by the paper's closed-form update rather than the semi-implicit
// Euler loop.
//
// F(B) is the exponential-relative (exprel) factor B*[(1-exp(-B))/B]. It is
// evaluated by a per-element branch: for small B (< cfcExprelThreshold) the
// Taylor series B - B^2/2 + B^3/6 - B^4/24 (remainder <= B^5/120 < 1e-12 at
// the threshold), and for larger B the direct 1 - exp(-B). The division by B
// of the raw exprel quotient is cancelled analytically against the outer
// (A-v)*B factor before it enters the graph, so there is no divide-by-B node
// to guard at all; see decayFactor. Positivity of cm, gleak and the synaptic
// weights is enforced with a softplus, as in the LTC (implicit constraints).
type CfC struct {
	inDim, units int
	eps          float32

	gleak, vleak, cm *autograd.Variable // [units]
	mu, sigma, w     *autograd.Variable // [units, units] recurrent synapses
	sMu, sSigma, sW  *autograd.Variable // [inDim, units] sensory synapses
	inW, inB         *autograd.Variable // [inDim]
	outW, outB       *autograd.Variable // [units]

	// Reversal potentials are fixed random +/-1 constants, exactly as in the
	// LTC: NOT trainable and deliberately absent from Parameters(). They are
	// plain data; their signs enter the graph only through the erevRows*
	// row-view constants below, so backward produces no gradient on them at
	// all (the old design let them enter the graph as Var leaves and
	// accumulated dead gradients no one ever read; archived finding #10).
	erev  *tensor.Tensor // [units, units]
	sErev *tensor.Tensor // [inDim, units]

	// Construction-time graph constants shared by every Step, mirroring the
	// LTC's sparse contraction state (see ltc.go's contract for the full
	// bit-equivalence proof): the presynaptic-axis reduction used to run as
	// MatMuls against dense [pre*units, units] indicators — the O(units^3)
	// memory cliff of red team F1 / technical-debt item #14, mirrored here
	// by finding F-RT2. drive() now folds the activation blocks and ends in
	// a normalizing MatMul against ident; erevRowsR/S carry the per-
	// presynaptic numerator coefficients as row views sharing the
	// erev/sErev storage (Load overwrites that storage in place; the rows
	// track it with no rebuild), and planR/planS record the per-
	// postsynaptic term lists as O(nnz) metadata.
	ident                *autograd.Variable
	zeroV                *autograd.Variable
	erevRowsR, erevRowsS []*autograd.Variable
	planR, planS         *synapsePlan

	wiring *Wiring
}

// CfC satisfies the Cell interface.
var _ Cell = (*CfC)(nil)

// Closed-form numerics constants.
const (
	// cfcExprelThreshold switches decayFactor between the Taylor branch and
	// the direct 1-exp(-B) branch. At |B| = 1e-2 the dropped Taylor term is
	// B^5/120 ~= 8e-13, far below float32 epsilon, while the direct form
	// loses only ~2 decimal digits to cancellation there. The two branches
	// agree (value and slope) to ~1e-10 at the boundary.
	cfcExprelThreshold = 1e-2
	// cfcMaxScale caps the float64 time scale before it enters the float32
	// graph. For ts >= ~1e3 the decay exp(-kappa*ts) is already exactly 0 in
	// any realistic parameterization (kappa*ts >> 100), so capping at 1e30
	// changes no representable outcome; it guarantees the softplus cap on
	// kappa keeps the product finite and, just as importantly, non-negative
	// (a negative decay rate would turn exp(-B) into a blow-up).
	cfcMaxScale = 1e30
)

// NewCfC creates a CfC cell. wiring may be nil, meaning FullyConnected;
// otherwise its sensory mask must have shape [inDim, units] and its
// recurrent mask [units, units]. Unlike the LTC there is no unfolds
// parameter: the closed-form solution advances the full time span in a
// single step. Parameter init ranges follow the LTC reference
// implementation. rng must be non-nil and supplies every initialization
// draw, so a fixed seed reproduces a cell bit for bit. Panics if
// inDim < 1 or units < 1, or if a non-nil wiring's mask shapes do not
// match [inDim, units] / [units, units]. (LoadCfC additionally caps
// units and inDim on the load path only; the constructor accepts any
// values >= 1 — see doc/persistence.md.)
func NewCfC(inDim, units int, wiring *Wiring, rng *rand.Rand) *CfC {
	if inDim < 1 || units < 1 {
		panic(fmt.Sprintf("nn.NewCfC: invalid dims in=%d units=%d", inDim, units))
	}
	if wiring == nil {
		wiring = FullyConnected(inDim, units)
	}
	if !cfcShapeEq(wiring.sensoryMask.Shape, inDim, units) ||
		!cfcShapeEq(wiring.recurrentMask.Shape, units, units) {
		panic(fmt.Sprintf("nn.NewCfC: wiring mask shapes %v and %v do not match [inDim=%d, units=%d]",
			wiring.sensoryMask.Shape, wiring.recurrentMask.Shape, inDim, units))
	}
	uniform := func(lo, hi float32, shape ...int) *autograd.Variable {
		return autograd.Var(tensor.Uniform(rng, lo, hi, shape...))
	}
	// Reversal potentials are random +/- 1, as in the LTC wiring. Plain
	// tensors: they never become graph nodes (see the struct field docs).
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
	c := &CfC{
		inDim: inDim, units: units, eps: 1e-8,
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
		erev:   erevInit(units, units),
		sErev:  erevInit(inDim, units),
		ident:  autograd.Const(identityMat(units)),
		zeroV:  autograd.Const(tensor.FromData([]float32{0}, 1)),
		wiring: wiring,
		planR:  newSynapsePlan(wiring.recurrentMask),
		planS:  newSynapsePlan(wiring.sensoryMask),
	}
	// Numerator coefficients are row views of the (already drawn) reversal
	// potentials, exactly as in the LTC's constructor (ltc.go): O(pre*units)
	// constants sharing the erev storage, so the fixed +/-1 signs never
	// appear as graph leaves.
	c.erevRowsR = erevRowViews(c.erev)
	c.erevRowsS = erevRowViews(c.sErev)
	return c
}

// cfcShapeEq reports whether sh equals want, without allocating. Named
// independently of ltc.go's helper so the two cells never share private code.
func cfcShapeEq(sh []int, want ...int) bool {
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

// StateSize returns the hidden state dimension.
func (c *CfC) StateSize() int { return c.units }

// Parameters returns the cell's 13 trainable variables in a fixed
// order: gleak, vleak, cm, mu, sigma, w, sMu, sSigma, sW, inW, inB,
// outW, outB — the same frozen order as the LTC, which is the cell's
// stream order inside SaveCfC and the positional key for
// serialize.WriteParameters / optimizer.SaveState. The reversal
// potentials erev/sErev are fixed constants and intentionally excluded;
// SaveCfC persists them anyway, so LoadCfC reproduces the cell exactly.
func (c *CfC) Parameters() []*autograd.Variable {
	return []*autograd.Variable{
		c.gleak, c.vleak, c.cm,
		c.mu, c.sigma, c.w,
		c.sMu, c.sSigma, c.sW,
		c.inW, c.inB, c.outW, c.outB,
	}
}

// Step advances the cell by one RNN step over the time span ts, using
// the closed-form membrane solution (Lemma 1) instead of ODE unfolding.
// x is [batch, inDim], h is [batch, units] or nil for a zero initial
// state. It returns the affinely mapped output vNew⊙outW+outB with
// shape [batch, units] and the new raw state vNew with shape
// [batch, units]. Panics if ts is not positive and finite (NaN, +/-Inf,
// zero and negative values are all rejected), or if x/h do not have the
// expected rank and widths (h is validated explicitly; see
// nn/cfc_fused.go for the strictness note).
//
// The two drives, the g/a assembly, the decayRate cap chain and the
// exprel-stabilized decayFactor run as one fused kernel (nn/cfc_fused.go,
// stage 18b): a single graph node whose forward replays the per-element
// operation sequence with identical rounding boundaries, and whose
// hand-written VJP replays the graph path's backward accumulation order
// contribution for contribution, bit for bit. The input affine, the four
// softplus constraints and the output affine stay graph-level (the
// softplus chain owns the documented 1D-lift leaf gradient shapes, the
// affine leaves own the OUTPUT fold class). The node count per Step
// drops from 66 + 14*(inDim+units) to 24. drive/contract/decayRate/
// decayFactor below remain as the fused kernel's white-box oracle (the
// pre-fusion graph path, reassembled verbatim by legacyCfCStep in
// nn/cfc_fused_diff_test.go).
func (c *CfC) Step(x, h *autograd.Variable, ts float64) (out, hNew *autograd.Variable) {
	// NaN-aware positivity check: NaN > 0 is false, so NaN panics here too;
	// both infinities are rejected explicitly, as in the LTC.
	if !(ts > 0) || math.IsInf(ts, 0) {
		panic(fmt.Sprintf("nn.CfC.Step: ts must be positive and finite, got %v", ts))
	}
	batch := x.Data.Rows()
	if h == nil {
		h = autograd.Var(tensor.New(batch, c.units))
	}

	// Affine input mapping (graph-level: chained-input topologies enter
	// the step here).
	inputs := autograd.Add(autograd.Hadamard(x, c.inW), c.inB)

	// Positivity-constrained parameters (softplus, graph-level: the chain
	// owns the documented [1, units] 1D-lift leaf gradient shapes).
	gleak := autograd.Softplus(c.gleak)
	cm := autograd.Softplus(c.cm)
	wPos := autograd.Softplus(c.w)
	sWPos := autograd.Softplus(c.sW)

	// The closed-form update runs as one fused kernel (nn/cfc_fused.go).
	vNew := c.fusedStep(h, inputs, gleak, cm, wPos, sWPos, ts)

	out = autograd.Add(autograd.Hadamard(vNew, c.outW), c.outB)
	return out, vNew
}

// drive accumulates numerator and denominator synaptic currents from a
// presynaptic source (inputs or previous state), self-contained in cfc.go
// (it does not share the LTC's synapse routine, but it does share the LTC's
// sparse contraction machinery and its bit-equivalence contract — see
// ltc.go's contract for the full proof). Row i of the parameter matrices
// parameterizes the synapses of presynaptic neuron i, whose wiring mask row
// comes from maskRow(i); the sensory and recurrent paths share this single
// calling convention. Each iteration produces a [batch, units] outer product
// (column [batch, 1] x row [1, units]); contract then reduces the
// presynaptic axis:
//
//	den[:, j] = Σ_i block_i[:, j]
//	num[:, j] = Σ_i block_i[:, j] · erev[i, j]
//
// with the fixed +/-1 reversal potentials carried by the erevRows row-view
// constants — exactly the ltc.go scheme. The masked block entries are
// exactly +0, so the fold sums precisely the wired terms bit for bit, and
// the reversal potentials never enter the graph as leaves (no dead
// gradient, archived finding #10).
func (c *CfC) drive(
	pre, mu, sigma, w *autograd.Variable,
	erevRows []*autograd.Variable,
	maskRow func(i int) *tensor.Tensor,
) (num, den *autograd.Variable) {
	n := pre.Data.Cols()
	blocks := make([]*autograd.Variable, n)
	for i := 0; i < n; i++ {
		muR := autograd.SliceRow(mu, i)
		sigR := autograd.SliceRow(sigma, i)
		wR := autograd.SliceRow(w, i)
		preCol := autograd.Col(pre, i) // [batch, 1]
		act := autograd.Sigmoid(autograd.Hadamard(sigR, autograd.Sub(preCol, muR)))
		act = autograd.Hadamard(act, wR)
		blocks[i] = autograd.Hadamard(act, autograd.Const(maskRow(i)))
	}
	return c.contract(blocks, erevRows)
}

// contract is the CfC's copy of the LTC's sparse presynaptic reduction
// (ltc.go's contract, whose doc comment carries the bit-equivalence proof):
// a +0-seeded ascending fold of the blocks (the denominator) and of the
// blocks scaled by their reversal rows (the numerator), each ended by a
// MatMul against the units×units identity that replicates the former
// indicator MatMul's zero-skip in the backward pass. That normalization
// holds on the multi-source path; the single-source corner (inDim=1 or
// units=1, a zero-valued incoming gradient, an erev of -1) can leave a -0
// sign on a zero gradient via the numerator fold's Hadamard backward —
// the value is 0 and nothing downstream can observe the sign, a
// deliberately accepted corner; ltc.go's contract comment carries the full
// analysis, and red-team scans measured zero trajectory divergence for the
// CfC. The single-source denominator shortcut is preserved: with one
// presynaptic row the indicator was the identity, so den is the raw block.
func (c *CfC) contract(blocks, erevRows []*autograd.Variable) (num, den *autograd.Variable) {
	if len(blocks) == 1 {
		den = blocks[0]
	} else {
		den = autograd.Add(c.zeroV, blocks[0])
		for i := 1; i < len(blocks); i++ {
			den = autograd.Add(den, blocks[i])
		}
		den = autograd.MatMul(den, c.ident)
	}
	num = autograd.Add(c.zeroV, autograd.Hadamard(blocks[0], erevRows[0]))
	for i := 1; i < len(blocks); i++ {
		num = autograd.Add(num, autograd.Hadamard(blocks[i], erevRows[i]))
	}
	num = autograd.MatMul(num, c.ident)
	return num, den
}

// decayRate computes B = kappa*ts with kappa = G/(cm + eps), protecting the
// float32 graph from extreme ts. The time scale is computed in float64 and
// capped at cfcMaxScale before conversion; the conductance ratio is then
// capped with the smooth differentiable min from the LTC's capacitance
// scaling, cap(k) = k - softplus(k - hi), hi = MaxFloat32/scale/1.0001. While
// k << hi (every sane ts, where hi >= ~3e8) softplus(k - hi) = log(1 + e^{k-hi})
// underflows to exactly 0 in float32, so cap(k) is bit-identical to k and the
// closed-form algebra is untouched; the cap only engages where the unscaled
// product would overflow float32. Because hi stays >= ~3.4e8 under the scale
// cap, softplus(k - hi) < k for every representable k, so B never goes
// negative here - a negative B would turn the decay exp(-B) into growth.
func (c *CfC) decayRate(g, cm, epsV *autograd.Variable, ts float64) *autograd.Variable {
	scale64 := ts
	if scale64 > cfcMaxScale {
		scale64 = cfcMaxScale
	}
	hi64 := math.MaxFloat32 / scale64 / 1.0001
	if hi64 > math.MaxFloat32 {
		hi64 = math.MaxFloat32
	}
	hiV := autograd.Const(tensor.FromData([]float32{float32(hi64)}, 1))
	kappa := autograd.Div(g, autograd.Add(cm, epsV))
	capped := autograd.Sub(kappa, autograd.Softplus(autograd.Sub(kappa, hiV)))
	return autograd.Scale(capped, float32(scale64))
}

// decayFactor computes F(B) = 1 - exp(-B) = B*exprel(B), the closed-form
// update's decay factor, stably across the whole non-negative float32 domain.
// B comes from decayRate and is finite and >= 0.
//
// The famous closed-form-CT trap is the raw quotient (1-exp(-B))/B at B -> 0:
// 1-exp(-B) cancels to 0 in finite precision and dividing by B then yields
// garbage (and a dead gradient). We sidestep it in two ways:
//
//   - Small B (< cfcExprelThreshold): Taylor-expand the whole B*exprel(B)
//     product as B - B^2/2 + B^3/6 - B^4/24. No division anywhere, full
//     float32 precision, and the gradient dF/dB = 1 - B + B^2/2 - B^3/6 -> 1
//     stays alive as B -> 0. The series is evaluated on B masked to [0, 1e-2)
//     so no power can overflow where this branch is inactive.
//   - Larger B: the direct 1 - exp(-B). The division by B of the exprel
//     quotient has cancelled analytically against the outer B factor, so the
//     "original formula" path needs no divide-by-B guard either; for huge B,
//     exp(-B) underflows to exactly 0 and F saturates at 1 (v_new -> A).
//
// The branch mask is a per-element constant built from B's data, so gradients
// flow through exactly the active branch per element; the two branches agree
// to ~1e-10 in value and slope at the threshold, so crossing it between
// finite-difference evaluations is invisible to gradcheck.
func (c *CfC) decayFactor(b *autograd.Variable) *autograd.Variable {
	maskT := tensor.New(b.Data.Shape...)
	oneMinusT := tensor.New(b.Data.Shape...)
	for i, v := range b.Data.Data {
		if v < cfcExprelThreshold {
			maskT.Data[i] = 1
		} else {
			oneMinusT.Data[i] = 1
		}
	}
	m := autograd.Const(maskT)
	oneMinus := autograd.Const(oneMinusT)

	// Taylor branch, on B masked into [0, threshold): finite powers guaranteed.
	bt := autograd.Hadamard(m, b)
	taylor := autograd.Sub(bt, autograd.Scale(autograd.Pow(bt, 2), 0.5))
	taylor = autograd.Add(taylor, autograd.Scale(autograd.Pow(bt, 3), 1.0/6.0))
	taylor = autograd.Sub(taylor, autograd.Scale(autograd.Pow(bt, 4), 1.0/24.0))

	// Direct branch: exp(-B) in [0, 1] for B >= 0, so 1 - exp(-B) never
	// cancels catastrophically here (B >= threshold >> float32 epsilon).
	ones := autograd.Const(tensor.New(b.Data.Shape...).OnesLike())
	direct := autograd.Sub(ones, autograd.Exp(autograd.Neg(b)))

	return autograd.Add(autograd.Hadamard(m, taylor), autograd.Hadamard(oneMinus, direct))
}
