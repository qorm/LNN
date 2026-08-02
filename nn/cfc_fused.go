package nn

import (
	"fmt"
	"math"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/tensor"
)

// This file holds the CfC's fused closed-form-step kernel (stage 18b), the
// sibling of the LTC's fused ODE-unfold kernel (nn/ltc_fused.go). Step's two
// synaptic drives and the closed-form membrane update — the g/a assembly,
// the decayRate cap chain and the exprel-stabilized decayFactor — used to
// record 66 + 14*(inDim+units) graph nodes per Step (P17 audit: 190 nodes at
// inDim=1/units=8, 346 at 4/16, vs 84 for the fused LTC step at 4/16): the
// closed form has no unfolds, so the whole per-step subgraph — not a
// sub-loop of it — collapses into ONE autograd.FusedOp node. The forward
// runs the identical float32 operation sequence in a single loop nest, and
// the backward is a hand-written VJP that replays the replaced subgraph's
// backward sweep contribution for contribution. What stays graph-level (and
// why): the input affine, the four softplus constraints and the output
// affine — the softplus chain owns the documented [1, units] 1D-lift leaf
// gradient shapes, the affine leaves own the OUTPUT fold class, and the
// input affine is where chained-input topologies enter the step (the LTC
// kernel made the same boundary call). After fusion a Step records 9 op
// nodes + 15 leaves = 24 nodes at any dims.
//
// # Bit-identity contract (forward)
//
// Every value the kernel produces replicates the graph path bit for bit, by
// the same method as the LTC kernel: the graph's rounding boundaries are
// per-element native float32 operations stored to tensors, so the fused
// loops keep the same per-element operation sequence as SEPARATE Go
// statements, and every product that feeds an addition goes through mul32's
// float64 conversion barrier (the arm64 FMA argument of nn/ltc_fused.go's
// header — the same objdump-verified reasoning applies here). The per-op
// correspondences relied on:
//
//   - The activation block = sigmoid(z)⊙w⊙mask in that multiply order (the
//     CfC drive uses Sigmoid+Hadamard+Hadamard, not the LTC's fused
//     SigmoidHadamard): two native roundings, then the fold additions. The
//     mask product feeds the fold sum, so it carries the mul32 barrier.
//   - The contraction folds = +0-seeded ascending accumulations (the
//     zeroV-seeded Add chains), unwired synapses arithmetically neutral
//     (+0 terms); the kernel accumulates the same terms in the same order
//     from zeroed buffers. The normalizing identity MatMul is the per-element
//     zero wash for finite folds and the literal tensor.MatMul only for a
//     fold holding +/-Inf or NaN (normFoldIdentity, shared with the LTC
//     kernel) — and the single-source denominator shortcut (inDim == 1 or
//     units == 1) skips fold and wash exactly as contract does. num always
//     folds and normalizes.
//   - Div forward = num * pow(den, -1) (NOT num/den), the inverse computed
//     through float64 math.Pow with the float32 conversion — for both
//     divisors (G+eps for A, cm+eps for kappa).
//   - decayRate's float64 pre-computation (the cfcMaxScale cap and the
//     hi = MaxFloat32/scale/1.0001 headroom) is replicated verbatim, and the
//     cap chain softplus is softplus32, a bit-exact replica of
//     tensor.Softplus (x > 20 passthrough, else log1p(exp) through float64).
//   - decayFactor's per-element branch: the mask is recomputed from the
//     kernel's own B values with the graph's comparison (B < 1e-2), and BOTH
//     branches are evaluated and mask-added in the graph's order
//     (m⊙taylor + (1-m)⊙direct), exactly as the graph does — the branch is
//     value selection, not graph structure, so there is no control-flow
//     divergence to replay. The Taylor powers go through float64 math.Pow
//     with the Scale factors 1/2, 1/6, 1/24, and the direct branch's Neg is
//     a genuine runtime multiply by -1 (negOne), NaN-propagating like
//     tensor.Neg.
//
// # Bit-identity contract (backward)
//
// The graph path's backward sweep over one Step is a single DAG sweep (no
// unfolds, so no cross-unfold chaining and no denBase-inside-the-kernel
// interleaving — the two complications the LTC kernel carried). The
// accumulation orders below were pinned by reading topological positions of
// the probe graph (sweep = reverse topological order), NOT by reading the
// construction order — two of the construction-order guesses provably fail
// (the gleak and tail positions), so the orders here are the measured ones:
//
//   - g.Grad = the kappa-side contribution FIRST (decayRate's Div backward
//     sweeps before A's Div backward), then the a-side (A's denominator
//     chain) — the a-Div denominator gradient is negHadamardPow2,
//     replicated by the shared divDenGrad.
//   - gleakSp.Grad = g-Add's SumRows reduction FIRST, then the
//     Hadamard(gleak, vleak) row reduction (g⊙vleak) — the DFS descends
//     into A's numerator chain before g's own Add, so the glv node appends
//     near the bottom of the sweep. vleak gets the same reduction's other
//     operand product (g⊙gleak), one contribution, delivered last.
//   - b.Grad = the DIRECT branch's contribution first (the oneMinus
//     subtree appends after the taylor subtree), then the taylor side.
//     The taylor side accumulates four contributions in sweep order:
//     pow4-chain, pow3-chain, the t1 passthrough, pow2-chain. BOTH branch
//     derivatives are computed and mask-added — never an if/else skip:
//     the inactive side's masked product is a signed zero whose addition
//     is not the identity on a -0 accumulator ((-0)+(+0) = +0), so
//     skipping it could flip a zero sign bit.
//   - Each activation block's gradient = the den-fold delivery FIRST
//     (ownership), then += the num-fold's g⊙erev product through mul32 —
//     the fold backwards' order (the den fold chain sweeps before the num
//     fold chain because the num chain appends first).
//   - h receives, in this order: the terminal Add's passthrough (the graph
//     hands h the vNew gradient buffer itself — the kernel delivers the
//     fused node's own Grad tensor after every internal use of it is done),
//     then the Sub branch's -(g⊙f), then the literal per-column Col-scatter
//     buffers in DESCENDING presynaptic order. These three classes are
//     mutually contiguous in the graph sweep relative to every EXTERNAL
//     contribution to h (the output affine's, a chained input's, a
//     loss-side seed's): h's only consumers inside the step are the
//     terminal Add and Sub, so no external contribution can interleave
//     between them under either record-high association — which is why the
//     CfC kernel needs NO hvN-style delivery node (the LTC's dense cmT
//     product sat unfolds deep, where chained contributions could land
//     between the scatters and the dense buffer).
//   - The parameter row scatters (SliceRow backwards) deliver one
//     zero-padded row buffer per presynaptic row in descending row order;
//     within one row the delivery order is pre-col scatter, w, sigma, mu
//     (the reverse of the per-row append order). The sensory drive sweeps
//     after the recurrent drive (its fold chain appends first).
//   - batch == 1 takes hadamardReduce's and negReduce's same-shape
//     ADOPTION arms (a -0 contribution keeps its sign bit instead of
//     washing to +0); units == 1 takes the flat scalar arms and the Sub
//     backward's preCol passthrough (dsub adopted per row, no SumCols);
//     the single-source denominator corner keeps the raw incoming gradient
//     with no wash (the F9-1 sign corner, ltc.go's contract comment).
//   - erev/sErev are live aliases of the cell's reversal storage (the
//     graph's erevRows constants share it, so both paths observe an
//     in-place polarity rewrite by Load identically); the wiring masks are
//     read from the wiring's mask storage (Load rebuilds the wiring; masks
//     do not mutate between a Step's forward and backward). mu, sigma,
//     sMu, sSigma and h are FROZEN at forward (the graph froze them in
//     per-Step SliceRow/Col nodes); vleak is read live at backward exactly
//     as the graph's Hadamard backward reads the live leaf.
//
// # Irregular seeded gradients
//
// The replaced subgraph's terminal node is an Add (the LTC's was a Div),
// so a manually seeded state gradient that is not exactly [batch, units]
// takes sumToShapeTake's reduction arms, never a broadcast: batch == 1 and
// a [k, units] seed takes the row-vector arm (SumRows, then the regular
// replay with the reduced plane); units == 1 and a [batch, k] seed takes
// the column-vector arm (SumCols likewise); batch == units == 1 takes the
// scalar arm (SumAll delivered as a [1] gradient to h, after which the
// Sub side's [1, 1] delivery panics with the engine's shape-mismatch
// message); every other shape panics with the SumToShape "cannot reduce"
// message before any delivery. The kernel reproduces the arms and both
// panic texts at the same points.
//
// A panicked backward leaves the graph path's internal accumulators
// unreplayable (the aborted sweep has already delivered partial
// gradients, so a retried Backward panics again on the polluted graph);
// the fused kernel panics before any delivery and stays clean on a
// retried call. Discard the graph after a caught panic — the kernel
// does not replicate the post-panic pollution (it could not, and the
// clean behavior is strictly preferable anyway).
//
// # State-shape validation
//
// The kernel validates h's shape up front and panics on every deviation
// ("state shape %v incompatible with [batch, units]"). The graph path
// panicked in the tensor layer for most wrong shapes but SILENTLY
// broadcast a broadcast-compatible wrong state (e.g. [batch, 1] or
// [1, units]) — inputs the Cell contract already forbids (h must be
// [batch, StateSize()]). The fused Step is deliberately strict there, the
// same call the LTC's fused kernel made (TestLTCFusedStateShapePanic);
// the divergence class is documented here and pinned in
// nn/cfc_fused_diff_test.go.
//
// # NaN semantics
//
// Identical acceptance level to the LTC kernel (see nn/ltc_fused.go):
// NaN-ness and every finite value are bit-faithful across the two paths;
// the payload and sign bit of a NaN produced by an operation whose
// operands are BOTH NaNs are NOT part of the contract.
type cfcFusedStash struct {
	batch, inDim, units int
	eps                 float32 // c.eps (1e-8)
	scale32, hi32       float32 // decayRate's capped scale and cap headroom

	// Forward-frozen operands (the graph path froze them in per-Step
	// SliceRow/Col nodes; reading the parents' live Data in the backward
	// would diverge under a mid-flight mutation).
	h         []float32 // batch*units
	mu, sig   []float32 // units*units, frozen
	sMu, sSig []float32 // inDim*units, frozen
	inputsD   []float32 // internal node data (engine-frozen, aliased)
	gleakD    []float32 // [units] softplus output (internal, aliased)
	cmD       []float32 // [units] softplus output (internal, aliased)
	wD, sWD   []float32 // softplus-constrained weights (internal, aliased)
	maskS     []float32 // wiring sensory mask storage (load-rebuilt, immutable)
	maskR     []float32 // wiring recurrent mask storage
	erev      []float32 // live alias of c.erev.Data
	sErev     []float32 // live alias of c.sErev.Data
	ident     *tensor.Tensor
	g, numA   []float32 // assembled conductance and numerator planes
	subAH     []float32 // a - h (the Hadamard(Sub(a, h), f) operand)
	b, bt     []float32 // decay rate and its masked Taylor operand
	expN      []float32 // exp(-B) (opExp backward reads the frozen output)
	f         []float32 // the decay factor F(B)
	parents   []*autograd.Variable
}

// fusedStep replaces Step's two drives and the closed-form membrane update
// with a single fused op node: parents [h, inputs, gleakSp, cmSp, wPos,
// sWPos, vleak, mu, sigma, sMu, sSigma], output the new state [batch,
// units]. h is the FIRST parent: the DFS descent from the output root hits
// the state leaf before any parameter chain, which is what keeps every
// trainable leaf in the state-rest fold class (the CfC has no spine class —
// nn/remat.go — and the fused step preserves that; outW/outB stay
// OUTPUT-class on the graph-level output affine).
func (c *CfC) fusedStep(h, inputs, gleakSp, cmSp, wPos, sWPos *autograd.Variable, ts float64) *autograd.Variable {
	batch := inputs.Data.Rows()
	units, inDim := c.units, c.inDim
	if h.Data.Dims() != 2 || h.Data.Rows() != batch || h.Data.Cols() != units {
		panic(fmt.Sprintf("nn.CfC.Step: state shape %v incompatible with [%d, %d]", h.Data.Shape, batch, units))
	}
	bu := batch * units
	st := &cfcFusedStash{
		batch: batch, inDim: inDim, units: units, eps: c.eps,
		h:       append([]float32(nil), h.Data.Data...),
		mu:      append([]float32(nil), c.mu.Data.Data...),
		sig:     append([]float32(nil), c.sigma.Data.Data...),
		sMu:     append([]float32(nil), c.sMu.Data.Data...),
		sSig:    append([]float32(nil), c.sSigma.Data.Data...),
		inputsD: inputs.Data.Data,
		gleakD:  gleakSp.Data.Data,
		cmD:     cmSp.Data.Data,
		wD:      wPos.Data.Data,
		sWD:     sWPos.Data.Data,
		maskS:   c.wiring.sensoryMask.Data,
		maskR:   c.wiring.recurrentMask.Data,
		// Live aliases, not copies: match the graph's shared-storage
		// erevRows constants (see the header).
		erev:  c.erev.Data,
		sErev: c.sErev.Data,
		ident: c.ident.Data,
		g:     make([]float32, bu),
		numA:  make([]float32, bu),
		subAH: make([]float32, bu),
		b:     make([]float32, bu),
		bt:    make([]float32, bu),
		expN:  make([]float32, bu),
		f:     make([]float32, bu),
	}
	// decayRate's float64 pre-computation, verbatim (cfc.go).
	scale64 := ts
	if scale64 > cfcMaxScale {
		scale64 = cfcMaxScale
	}
	hi64 := math.MaxFloat32 / scale64 / 1.0001
	if hi64 > math.MaxFloat32 {
		hi64 = math.MaxFloat32
	}
	st.scale32 = float32(scale64)
	st.hi32 = float32(hi64)

	vleakD := c.vleak.Data.Data // live at forward, exactly as the graph read it
	shape := []int{batch, units}

	// The two drives: +0-seeded ascending folds of the per-presynaptic
	// activation blocks (denominator and numerator), then the normalizing
	// identity-MatMul zero wash — num always, den only multi-source
	// (contract's single-source shortcut, see the header).
	denS := make([]float32, bu)
	numS := make([]float32, bu)
	cfcDriveForward(st.inputsD, inDim, batch, units, st.sMu, st.sSig, st.sWD, st.maskS, st.sErev, denS, numS)
	numS = normFoldIdentity(numS, shape, st.ident)
	if inDim > 1 {
		denS = normFoldIdentity(denS, shape, st.ident)
	}
	denR := make([]float32, bu)
	numR := make([]float32, bu)
	cfcDriveForward(st.h, units, batch, units, st.mu, st.sig, st.wD, st.maskR, st.erev, denR, numR)
	numR = normFoldIdentity(numR, shape, st.ident)
	if units > 1 {
		denR = normFoldIdentity(denR, shape, st.ident)
	}

	data := make([]float32, bu)
	for b := 0; b < batch; b++ {
		row := b * units
		for j := 0; j < units; j++ {
			k := row + j
			// G = gleak + (denS + denR); A's numerator = gleak⊙vleak +
			// (numS + numR); A = numA * pow(G+eps, -1) — the graph's
			// per-element adds and the Div's pow-then-product, with mul32
			// wherever a product feeds an addition/subtraction (FMA
			// barrier, see the header).
			denSum := denS[k] + denR[k]
			g := st.gleakD[j] + denSum
			st.g[k] = g
			gEps := g + st.eps
			numSum := numS[k] + numR[k]
			glv := mul32(st.gleakD[j], vleakD[j])
			numA := glv + numSum
			st.numA[k] = numA
			a := mul32(numA, float32(math.Pow(float64(gEps), -1)))
			// B = kappa*ts with the smooth cap: kappa = G * pow(cm+eps,
			// -1), capped = kappa - softplus(kappa - hi), B = capped*scale.
			cmEps := st.cmD[j] + st.eps
			kappa := mul32(g, float32(math.Pow(float64(cmEps), -1)))
			capped := kappa - softplus32(kappa-st.hi32)
			bv := capped * st.scale32
			st.b[k] = bv
			// F(B): the mask, BOTH branches, and the graph-order
			// mask add (see the header).
			var mf, omf float32
			if bv < cfcExprelThreshold {
				mf = 1
			} else {
				omf = 1
			}
			bt := mul32(mf, bv) // feeds the Taylor sub chain
			st.bt[k] = bt
			t1 := bt - mul32(float32(math.Pow(float64(bt), 2)), 0.5)
			t2 := t1 + mul32(float32(math.Pow(float64(bt), 3)), 1.0/6.0)
			taylor := t2 - mul32(float32(math.Pow(float64(bt), 4)), 1.0/24.0)
			e := float32(math.Exp(float64(bv * negOne)))
			st.expN[k] = e
			direct := 1 - e
			fv := mul32(mf, taylor) + mul32(omf, direct)
			st.f[k] = fv
			// v_new = v + (A - v)*F.
			sub := a - st.h[k]
			st.subAH[k] = sub
			data[k] = st.h[k] + mul32(sub, fv)
		}
	}
	parents := []*autograd.Variable{h, inputs, gleakSp, cmSp, wPos, sWPos, c.vleak, c.mu, c.sigma, c.sMu, c.sSigma}
	st.parents = parents
	return autograd.FusedOp(&tensor.Tensor{Shape: shape, Data: data}, parents, st.backward)
}

// cfcDriveForward accumulates one drive's denominator and numerator folds:
// block_i = sigmoid(sig_i⊙(pre_i − mu_i))⊙w_i⊙mask_i, den += block,
// num += block⊙erev, ascending presynaptic order from +0-seeded buffers —
// the graph path's SliceRow/Col/Sub/Hadamard/Sigmoid/Hadamard/Hadamard
// chain per row, element for element (see the header for the mul32 sites).
func cfcDriveForward(pre []float32, n, batch, units int, mu, sig, w, mask, er, denF, numF []float32) {
	for i := 0; i < n; i++ {
		muR := mu[i*units:]
		sigR := sig[i*units:]
		wR := w[i*units:]
		erR := er[i*units:]
		mkR := mask[i*units:]
		for b := 0; b < batch; b++ {
			row := b * units
			pv := pre[b*n+i]
			for j := 0; j < units; j++ {
				k := row + j
				sub := pv - muR[j]
				z := sigR[j] * sub
				s := sigmoid32(z)
				blk := mul32(s*wR[j], mkR[j]) // the mask product feeds the fold add
				denF[k] += blk
				numF[k] += mul32(blk, erR[j])
			}
		}
	}
}

// backward replays the replaced subgraph's backward sweep (see the header
// for the pinned orders). v.Grad is the fully accumulated incoming state
// gradient; irregular shapes take the terminal Add's reduction arms or its
// panic (see the header's irregular-seed section).
func (st *cfcFusedStash) backward(v *autograd.Variable) {
	batch, inDim, units := st.batch, st.inDim, st.units
	bu := batch * units
	p := st.parents // captured at construction; a v.Parents() call here would allocate per backward (copy semantics)
	h, inputs, gleakSp, cmSp := p[0], p[1], p[2], p[3]
	wPos, sWPos, vleak, mu, sigma, sMu, sSigma := p[4], p[5], p[6], p[7], p[8], p[9], p[10]
	shape := []int{batch, units}

	// The terminal Add's a-branch arms (sumToShapeTake), in its own order:
	// same shape (the regular path), scalar (batch == units == 1), row
	// vector (batch == 1), column vector (units == 1), else the panic.
	g := v.Grad
	var d1 *tensor.Tensor
	var dv []float32
	switch {
	case len(g.Shape) == 2 && g.Shape[0] == batch && g.Shape[1] == units:
		// Regular: the graph hands h the vNew gradient buffer itself
		// (sumToShapeTake's ownership transfer); the kernel delivers the
		// fused node's own Grad after every internal use of it is done.
		d1, dv = g, g.Data
	case batch == 1 && units == 1:
		// Scalar arm: SumAll delivered as a [1] gradient to h; the Sub
		// side's [1, 1] delivery then panics with the engine's
		// shape-mismatch message (no further delivery is observable).
		var s float32
		for _, x := range g.Data {
			s += x
		}
		addGrad(h, &tensor.Tensor{Shape: []int{1}, Data: []float32{s}})
		panic(fmt.Sprintf("autograd: gradient shape mismatch: accumulated %v vs incoming %v", h.Grad.Shape, shape))
	case batch == 1 && g.Dims() == 2 && g.Shape[1] == units:
		// Row-vector arm: SumRows over the seed, then the regular replay
		// with the reduced plane (a fresh buffer, as SumRows returns).
		dv = make([]float32, bu)
		for r := 0; r < g.Shape[0]; r++ {
			for j := 0; j < units; j++ {
				dv[j] += g.Data[r*units+j]
			}
		}
		d1 = &tensor.Tensor{Shape: shape, Data: dv}
	case units == 1 && g.Dims() == 2 && g.Shape[0] == batch:
		// Column-vector arm: SumCols over the seed -> [batch, 1].
		dv = make([]float32, bu)
		for b := 0; b < batch; b++ {
			var s float32
			for j := 0; j < g.Shape[1]; j++ {
				s += g.Data[b*g.Shape[1]+j]
			}
			dv[b] = s
		}
		d1 = &tensor.Tensor{Shape: shape, Data: dv}
	default:
		panic(fmt.Sprintf("tensor.SumToShape: cannot reduce shape %v to %v", g.Shape, shape))
	}

	// ---- membrane chain replay (all planes before any delivery) ----

	// decayFactor's branch masks, recomputed from the kernel's own B values
	// with the graph's comparison (the graph froze them as constants built
	// from the same B data).
	m := make([]float32, bu)
	om := make([]float32, bu)
	for k, bv := range st.b {
		if bv < cfcExprelThreshold {
			m[k] = 1
		} else {
			om[k] = 1
		}
	}

	// Terminal Hadamard/Sub: dSub = g⊙f (the Sub side's gradient, and A's
	// incoming gradient through the passthrough), d2 = Neg(dSub) for h,
	// gF = g⊙subAH (f's incoming gradient).
	dSub := make([]float32, bu)
	d2 := make([]float32, bu)
	gF := make([]float32, bu)
	for k := 0; k < bu; k++ {
		ds := dv[k] * st.f[k] // hadamardReduce same-shape product
		dSub[k] = ds
		d2[k] = ds * negOne // tensor.Neg (a genuine runtime multiply)
		gF[k] = dv[k] * st.subAH[k]
	}

	// f chain: gF enters both mask Hadamards. Taylor side first in
	// accumulation ORDER pow4 -> pow3 -> passthrough -> pow2 (sweep order,
	// see the header); its buffer accumulates from the pow4 contribution.
	btG := make([]float32, bu)
	for k := 0; k < bu; k++ {
		gt3 := mul32(gF[k], m[k]) // Had(m, taylor) delivery product; feeds the accumulations below
		gd4 := (gt3 * negOne) * (1.0 / 24.0)
		btG[k] = mul32(gd4, mul32(4, float32(math.Pow(float64(st.bt[k]), 3))))
		gd3 := gt3 * (1.0 / 6.0)
		btG[k] += mul32(gd3, mul32(3, float32(math.Pow(float64(st.bt[k]), 2))))
		btG[k] += gt3 // the t1 passthrough
		gd2 := (gt3 * negOne) * 0.5
		btG[k] += mul32(gd2, mul32(2, float32(math.Pow(float64(st.bt[k]), 1))))
	}
	// b.Grad: the DIRECT branch's contribution first (Neg, ⊙expN,
	// Scale(-1) — the double negation is replayed literally, not
	// simplified), then the taylor side's Had(m, b) delivery.
	bG := make([]float32, bu)
	for k := 0; k < bu; k++ {
		p1 := gF[k] * om[k]
		p2 := p1 * negOne
		p3 := p2 * st.expN[k]
		bG[k] = p3 * negOne
	}
	for k := 0; k < bu; k++ {
		bG[k] += mul32(btG[k], m[k])
	}

	// decayRate chain: gCap = bG⊙scale; kappa.Grad = gCap (capped-Sub
	// passthrough, first), then += the softplus cap chain's contribution
	// (Neg, then ⊙sigmoid(subKH) — the barrier rounds the product the
	// graph stored before accumulating). cmEps is batch-invariant (the
	// broadcast Add(cm, epsV)), a [units] row.
	ivK := make([]float32, bu)
	subKH := make([]float32, bu)
	cmEpsJ := make([]float32, units)
	for j := 0; j < units; j++ {
		cmEpsJ[j] = st.cmD[j] + st.eps
	}
	for b := 0; b < batch; b++ {
		row := b * units
		for j := 0; j < units; j++ {
			k := row + j
			ivK[k] = float32(math.Pow(float64(cmEpsJ[j]), -1))
			kap := mul32(st.g[k], ivK[k])
			subKH[k] = kap - st.hi32
		}
	}
	kapG := make([]float32, bu)
	for k := 0; k < bu; k++ {
		gcv := bG[k] * st.scale32
		kapG[k] = gcv // first contribution (passthrough)
		kapG[k] += mul32(gcv*negOne, sigmoid32(subKH[k]))
	}

	// kappa's Div backward: g gets kapG⊙ivK (its FIRST contribution). The
	// denominator arm is the BROADCAST one: hadamardReduce row-reduces
	// kapG⊙g to the [1, units] denominator shape with mul32-rounded
	// products into a +0-seeded accumulator (b⁻² is constant along the
	// broadcast axis, so the reduction precedes the pow — opDiv's
	// documented reduce-then-scale), then negHadamardPow2 per element,
	// then the cmEps-Add a-branch's one-row SumRows (+0-seeded copy; the
	// units == 1 scalar arm's SumAll washes identically).
	gG := make([]float32, bu)
	for k := 0; k < bu; k++ {
		gG[k] = kapG[k] * ivK[k]
	}
	gaK := make([]float32, units)
	for b := 0; b < batch; b++ {
		for j := 0; j < units; j++ {
			gaK[j] += mul32(kapG[b*units+j], st.g[b*units+j])
		}
	}
	cmG := make([]float32, units)
	for j := 0; j < units; j++ {
		pb2 := float32(math.Pow(float64(cmEpsJ[j]), -2))
		m := gaK[j] * pb2
		cmG[j] += mul32(m, negOne)
	}

	// A's Div backward (d_a = dSub): numA gets dSub⊙ivA; G+eps gets
	// divDenGrad(dSub⊙numA, gEps) — g's SECOND contribution (+= after the
	// kappa side, see the header).
	gEps := make([]float32, bu)
	dNumA := make([]float32, bu)
	gaA := make([]float32, bu)
	for k := 0; k < bu; k++ {
		ge := st.g[k] + st.eps
		gEps[k] = ge
		dNumA[k] = dSub[k] * float32(math.Pow(float64(ge), -1))
		gaA[k] = dSub[k] * st.numA[k]
	}
	dGEps := divDenGrad(gaA, gEps)
	for k := 0; k < bu; k++ {
		gG[k] += dGEps[k]
	}

	// g-Add backward: gleakSp's FIRST contribution is the SumRows
	// reduction of gG (SumAll when units == 1).
	gleakG1 := cfcReduceRows(gG, batch, units)

	// numA-Add backward: the glv side's incoming gradient is dNumA reduced
	// to [1, units] (passthrough at batch == 1, SumAll at units == 1,
	// SumRows otherwise); numS/numR carry dNumA's values into the washes.
	gLV := dNumA
	switch {
	case units == 1:
		var s float32
		for _, x := range dNumA {
			s += x
		}
		gLV = []float32{s}
	case batch > 1:
		gLV = make([]float32, units)
		for b := 0; b < batch; b++ {
			for j := 0; j < units; j++ {
				gLV[j] += dNumA[b*units+j]
			}
		}
	}

	// Hadamard(gleak, vleak) backward: gleakSp's SECOND contribution
	// (gLV⊙vleak) and vleak's only one (gLV⊙gleak) — the row/scalar
	// reduction arms with mul32-rounded products and a +0-seeded
	// accumulator. vleak is read LIVE here, exactly as the graph's
	// Hadamard backward reads the live leaf.
	vleakD := vleak.Data.Data
	gleakG2 := make([]float32, units)
	vleakG := make([]float32, units)
	if units == 1 {
		for k, x := range gLV {
			gleakG2[0] += mul32(x, vleakD[k])
			vleakG[0] += mul32(x, st.gleakD[k])
		}
	} else {
		for i := 0; i < len(gLV)/units; i++ {
			for j := 0; j < units; j++ {
				gleakG2[j] += mul32(gLV[i*units+j], vleakD[j])
				vleakG[j] += mul32(gLV[i*units+j], st.gleakD[j])
			}
		}
	}

	// Fold gradients: the MatMulTransB-against-identity zero wash for
	// finite gradients (literal call only for non-finite ones). num always
	// washes; den skips the wash on its single-source shortcut. Both
	// drives' num washes see the same values (dNumA), as do both den
	// washes (gG) — the graph runs two identical washes per pair.
	numFG := normFoldIdentityTransB(dNumA, shape, st.ident)
	var denW []float32
	if inDim > 1 || units > 1 {
		denW = normFoldIdentityTransB(gG, shape, st.ident)
	}
	denFGS, denFGR := gG, gG
	if inDim > 1 {
		denFGS = denW
	}
	if units > 1 {
		denFGR = denW
	}

	// ---- deliveries, in the graph sweep's order (see the header) ----

	// h's passthrough, cmSp, h's -(g⊙f), gleakSp's first contribution.
	addGrad(h, d1)
	glShape := []int{units}
	addGrad(cmSp, &tensor.Tensor{Shape: glShape, Data: cmG})
	addGrad(h, &tensor.Tensor{Shape: shape, Data: d2})
	addGrad(gleakSp, &tensor.Tensor{Shape: glShape, Data: gleakG1})

	// The recurrent drive's block chains, descending presynaptic order;
	// within one row the delivery order is pre-col scatter, w, sigma, mu
	// (the reverse of the per-row append order).
	cfcDriveBackward(st.h, units, batch, units, st.mu, st.sig, st.wD, st.maskR, st.erev, denFGR, numFG,
		func(i int, dpre []float32) {
			buf := tensor.New(batch, units)
			for b := 0; b < batch; b++ {
				buf.Data[b*units+i] = dpre[b]
			}
			addGrad(h, buf)
		},
		func(i int, dwR, dsR, dmR []float32) {
			addGrad(wPos, &tensor.Tensor{Shape: []int{units, units}, Data: cfcScatterRow(units, units, i, dwR)})
			addGrad(sigma, &tensor.Tensor{Shape: []int{units, units}, Data: cfcScatterRow(units, units, i, dsR)})
			addGrad(mu, &tensor.Tensor{Shape: []int{units, units}, Data: cfcScatterRow(units, units, i, dmR)})
		})

	// The sensory drive's block chains, descending (its fold chain appends
	// first, so it sweeps after the recurrent drive).
	cfcDriveBackward(st.inputsD, inDim, batch, units, st.sMu, st.sSig, st.sWD, st.maskS, st.sErev, denFGS, numFG,
		func(i int, dpre []float32) {
			buf := tensor.New(batch, inDim)
			for b := 0; b < batch; b++ {
				buf.Data[b*inDim+i] = dpre[b]
			}
			addGrad(inputs, buf)
		},
		func(i int, dwR, dsR, dmR []float32) {
			addGrad(sWPos, &tensor.Tensor{Shape: []int{inDim, units}, Data: cfcScatterRow(inDim, units, i, dwR)})
			addGrad(sSigma, &tensor.Tensor{Shape: []int{inDim, units}, Data: cfcScatterRow(inDim, units, i, dsR)})
			addGrad(sMu, &tensor.Tensor{Shape: []int{inDim, units}, Data: cfcScatterRow(inDim, units, i, dmR)})
		})

	// gleakSp's second contribution and vleak's only one (the glv node's
	// position at the bottom of the sweep, see the header).
	addGrad(gleakSp, &tensor.Tensor{Shape: glShape, Data: gleakG2})
	addGrad(vleak, &tensor.Tensor{Shape: glShape, Data: vleakG})
}

// cfcDriveBackward replays one drive's per-presynaptic block backward
// chains in descending row order: the block gradient (den-fold values,
// then += the num-fold's g⊙erev product), the mask and weight Hadamard
// backwards, opSigmoid's fused loop, the z Hadamard and Sub backwards with
// their reduction arms (batch == 1 adoption, units == 1 scalar/passthrough
// — see the header), then hands the per-row results to the delivery
// callbacks (the Col and SliceRow backwards' literal scatter buffers).
func cfcDriveBackward(
	pre []float32, n, batch, units int,
	mu, sig, w, mask, er, denFG, numFG []float32,
	deliverPre func(i int, dpre []float32),
	deliverRows func(i int, dwR, dsR, dmR []float32),
) {
	for i := n - 1; i >= 0; i-- {
		muR := mu[i*units:]
		sigR := sig[i*units:]
		wR := w[i*units:]
		erR := er[i*units:]
		mkR := mask[i*units:]
		dpre := make([]float32, batch)
		dmR := make([]float32, units)
		dsR := make([]float32, units)
		dwR := make([]float32, units)
		for b := 0; b < batch; b++ {
			row := b * units
			pv := pre[b*n+i]
			for j := 0; j < units; j++ {
				k := row + j
				// Recompute the block's forward values from the frozen
				// operands (same expression structure as the forward,
				// hence the same bits).
				sub := pv - muR[j]
				z := sigR[j] * sub
				s := sigmoid32(z)
				// Block gradient: den-fold values, then += the num-fold's
				// g⊙erev product (the fold backwards' ownership order).
				gNum := mul32(numFG[k], erR[j])
				g := denFG[k] + gNum
				// Mask Hadamard backward (act side), then the weight
				// Hadamard backward (s side) — plain tensor stores.
				gm := g * mkR[j]
				gw := gm * wR[j]
				// opSigmoid's fused backward loop.
				dz := mul32(gw, mul32(s, 1-s))
				dsub := mul32(dz, sigR[j]) // barrier: feeds the SumCols add
				// The row accumulators replicate the graph's ownership
				// semantics: with batch == 1 the reductions ADOPT their
				// product buffers (the signed-zero corners); otherwise
				// they accumulate into a +0-seeded row buffer.
				if batch == 1 {
					dwR[j] = mul32(gm, s)
					dsR[j] = mul32(dz, sub)
					dmR[j] = mul32(dsub, negOne)
				} else {
					dwR[j] += mul32(gm, s)
					dsR[j] += mul32(dz, sub)
					dmR[j] += mul32(dsub, negOne)
				}
				// With units == 1 the Sub backward's preCol side is a
				// passthrough (SumToShape's same-shape arm): dsub is
				// adopted per row, at any batch.
				if units == 1 {
					dpre[b] = dsub
				} else {
					dpre[b] += dsub
				}
			}
		}
		deliverPre(i, dpre)
		deliverRows(i, dwR, dsR, dmR)
	}
}

// cfcReduceRows replicates sumToShapeTake's reduction of a [batch, units]
// gradient to a [units] row target: SumAll into one element when
// units == 1 (the scalar arm fires first), SumRows otherwise — +0-seeded
// ascending accumulation in both arms.
func cfcReduceRows(g []float32, batch, units int) []float32 {
	if units == 1 {
		var s float32
		for _, x := range g {
			s += x
		}
		return []float32{s}
	}
	r := make([]float32, units)
	for b := 0; b < batch; b++ {
		for j := 0; j < units; j++ {
			r[j] += g[b*units+j]
		}
	}
	return r
}

// cfcScatterRow replicates the SliceRow backward: the row accumulator
// written into row i of a zero-padded [pre, units] buffer.
func cfcScatterRow(pre, units, i int, acc []float32) []float32 {
	g := make([]float32, pre*units)
	copy(g[i*units:(i+1)*units], acc)
	return g
}

// softplus32 replicates tensor.Softplus's numerically stable softplus bit
// for bit (the tensor kernel is unexported): the x > 20 passthrough, then
// log1p(exp(x)) through float64 with the float32 conversion.
func softplus32(x float32) float32 {
	if x > 20 {
		return x
	}
	return float32(math.Log1p(math.Exp(float64(x))))
}
