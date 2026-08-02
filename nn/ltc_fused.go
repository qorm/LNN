package nn

import (
	"fmt"
	"math"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/tensor"
)

// This file holds the LTC's fused ODE-unfold kernel (technical-debt item
// 16b, extended to the sensory synapse path in stage 19a). Step's unfolds
// loop used to record ~6*units graph nodes per unfold (Col/Sub/Hadamard/
// SigmoidHadamard per presynaptic neuron, two fold chains, two
// normalizing MatMuls, the membrane Add/Div) — the stage-15 profiling
// showed ~80% of the step's wall clock was per-node interpreter overhead,
// not math. The fused kernel replaces the whole loop with ONE
// autograd.FusedOp node: the forward runs the identical float32 operation
// sequence in a single loop nest, and the backward is a hand-written VJP
// that replays the replaced subgraph's backward sweep contribution for
// contribution. Stage 19a moved the boundary outward again: the sensory
// drive, the numConst = gleakSp⊙vleak + numS assembly and the denS input
// of the denBase chain — ~10*inDim+2 further nodes per Step, the largest
// remaining interpreter-overhead share of the fused step (P1 profile:
// 14.8%) — are internalized too, so a whole Step is now inputs affine +
// cmT chain + three softplus/mask nodes + ONE fused node (+ the hvN
// delivery node) + output affine. What stays graph-level, and why: the
// input affine (chained-input topologies enter the step through it), the
// cmT chain (its spine fold class is the graph structure itself), the
// softplus constraints and the mask folds (they own the documented
// [1, units] 1D-lift leaf gradient shapes), and the output affine (its
// operands own the OUTPUT fold class).
//
// # Bit-identity contract (forward)
//
// Every value the kernel produces replicates the graph path bit for bit.
// The graph path's rounding boundaries come from its kernel structure —
// each tensor op evaluates one native float32 operation per element and
// stores it — so the fused loop reproduces those boundaries by keeping the
// same per-element operation sequence as SEPARATE Go statements. That is
// also the FMA argument: Go forms FMADDS only inside a single a*b+c
// expression, and no statement below fuses — the fused kernel's own loops
// compile to FMULS+FADDS with zero FMADDS on arm64 (objdump-verified), so
// the two paths share the same rounding structure. The one place FMADDS
// does appear is inside tensor.MatMul's inner loop on arm64 — that
// predates this kernel, and the graph path ran the very same MatMul, so
// the contraction is identical in both baselines and cancels out of the
// comparison. Where the graph's fused backward loops used mul32 (an
// explicit float64 rounding barrier, see autograd), the replay below
// calls the identical mul32 formula, and the barrier holds (the same
// objdump check).
//
// The per-op correspondences relied on:
//
//   - SigmoidHadamard forward = sigmoid(z) then a native product with the
//     weight row (tensor.sigmoid's stable form, replicated by sigmoid32).
//   - The contraction fold = a +0-seeded ascending accumulation (the
//     zeroV-seeded Add chain), with unwired synapses arithmetically
//     neutral (+0 terms); the kernel accumulates the same terms in the
//     same order from a zeroed buffer.
//   - The terminal MatMul against the units identity is a value-preserving
//     copy with a zero wash: for output element j the only nonzero
//     identity coefficient is ident[j,j] = 1, so a nonzero input x
//     contributes x*1 = x while +/-0 inputs are skipped by MatMul's av==0
//     branch and the accumulator stays +0 (adding +/-0 to +0 keeps +0).
//     The kernel scans each fold and takes that per-element wash (exact
//     for every finite fold, forward and backward); only a fold holding
//     +/-Inf or NaN runs the literal tensor.MatMul/MatMulTransB, whose
//     av*0 = NaN spreading across the row the wash cannot reproduce
//     (reachable in the backward through non-finite seeded gradients,
//     and in the forward through adversarially loaded weights).
//   - Div forward = num * pow(den, -1) (NOT num/den), Div backward =
//     g⊙inv and -(g⊙num)⊙pow(den,-2) with the mul32-negated tail
//     (autograd's negHadamardPow2, replicated by divDenGrad).
//   - With a single presynaptic neuron (units == 1) the denominator skips
//     the fold and the normalizing MatMul entirely — den is the raw block
//     (contract's single-source shortcut), so its gradient is the raw
//     incoming gradient with NO wash.
//
// # Bit-identity contract (backward)
//
// The graph path's backward sweep over one step processes unfolds in
// strictly reverse order (reverse topological order of the per-step graph:
// the unfold-t subgraph is appended after unfold t-1's). Within unfold t:
// Div, the den fold (den MatMulTransB wash, then the Add chain handing
// every block the same den-fold gradient values), then num likewise, then
// the per-presynaptic block chains in DESCENDING presynaptic order
// (SigmoidHadamard's fused dz/dw, the z Hadamard, the Sub, the Col
// scatter into the previous state's gradient), and finally the
// Hadamard(cmT, v) backward (row-reduced cmT contribution, dense state
// contribution) — whose state contribution therefore lands AFTER every
// Col scatter. The shared accumulators replicate their graph
// accumulation orders: the frozen mu/sigma row gradients and the masked
// weight row gradients sum one fresh row buffer per unfold in reverse
// unfold order (ownership on the first, += afterwards), exactly as the
// shared SliceRow nodes did before their single backward ran; cmT's
// gradient sums the per-unfold row reductions in reverse unfold order,
// and the graph-level denBase chain contributes afterwards, as in the
// graph path.
//
// Delivery granularity to the graph is the one the graph path had:
//   - numConst / denBase gradients accumulate one cloned contribution per
//     unfold (the cloned numerator gradient / the denominator gradient),
//     reverse unfold order — INSIDE the kernel since 19a (neither is a
//     graph node anymore). The numConst chain's replay then distributes:
//     the glvHad node (gleakSp's second contribution and vleak's only
//     one, the row/scalar reduction arms, vleak read live exactly as the
//     graph's Hadamard backward reads it) and the numS fold wash. The
//     denBase chain's replay distributes gleakSp's FIRST contribution
//     (matching the chain-before-numConst order) and the denS fold wash
//     (multi-source; the inDim == 1 single-source shortcut keeps the raw
//     gradient, the F9-1 twin of units == 1).
//   - mu, sigma and wM receive one zero-padded row-scatter buffer per
//     presynaptic row, in descending row order (the SliceRow backwards);
//     sMu, sSigma and sWm likewise over the sensory rows.
//   - inputs receives the sensory Col backwards' literal per-column
//     scatter buffers, in descending column order.
//   - h receives the literal per-column scatter buffers in descending
//     column order from the fused node's backward, then the dense
//     cmT-product buffer from hvN, a separate delivery node (see
//     fusedUnfolds): the graph path's Hadamard(cmT, v) node completes its
//     DFS subtree right where the state descent ends — after the cm chain
//     and the previous step's subgraph, before numConst's — and hvN
//     occupies the same topological slot, so the dense contribution lands
//     where the graph path's did relative to every EXTERNAL contribution
//     to the state gradient: the output-affine contribution of a
//     record-high seed lands before the whole sequence
//     (((affine + col...) + cmT-prod), the interleaving nn.UnrollRemat
//     documents); a contribution from a chained input subgraph (the next
//     step's input built from this step's output, appended during
//     numConst's expansion) lands BETWEEN the scatters and the dense
//     buffer (((col... + affine) + cmT-prod); a late loss-side
//     contribution lands after everything (((col... + cmT-prod) +
//     affine). A single atomic VJP can reproduce only the first and
//     third associations; the delivery node reproduces all three. The
//     stage-19a parent change re-derives the slot instead of assuming
//     it: numConst is gone, and everything that used to append during
//     its subtree expansion — the sensory drive's chain, the gleak
//     chain, vleak, and the chained input subgraph (built on this step's
//     output through the graph-level input affine) — now appends during
//     the inputs parent's subtree expansion, which occupies exactly the
//     post-hvN position numConst's subtree occupied (parents [cmT, h,
//     hvN, inputs, ...]: hvN still completes right after the state
//     descent, before every one of those expansions), so all three
//     associations carry over mechanically.
//   - cmT receives its contributions in the graph path's exact order:
//     the Hadamard(cmT, v) reductions of unfolds T..2, then the denBase
//     chain's reduction, then unfold 1's Hadamard reduction — the denBase
//     chain's contribution interleaves BETWEEN the Hadamard ones in the
//     graph path (the chain's backward runs inside the last unfold's
//     sweep segment), which is why the chain lives inside the kernel
//     (gleakSp's row is delivered first, matching the
//     chain-before-numConst order; denS's side is internal since 19a).
//     The cm chain itself is graph-level (the fused node's first
//     parent), so cm keeps its spine fold class and its UnrollRemat
//     σ-sweep behavior is untouched.
//
// # NaN semantics
//
// NaN-ness and every finite value are bit-faithful across the two paths:
// each NaN-producing operation in the kernel is the graph path's own
// float32 operation (or its bit-identical mul32 replica with the native
// non-finite arm), so a NaN appears at exactly the same elements. The
// payload and sign bit of a NaN produced by an operation whose operands
// are BOTH NaNs are NOT part of the contract: they depend on the
// compiler's instruction selection rather than on any replicated
// rounding structure. This is the same acceptance level as the
// single-source sign-bit corner F9-1 — finite values and NaN-ness stay
// exact; double-NaN collision payloads do not.

// ltcFusedStash carries the forward values the hand-written VJP replays
// from. v/num/den/inv hold one [batch*units] plane per unfold (v holds
// unfolds+1: the input state plus every substep result); mu/sig/sMu/sSig
// are frozen copies of the parameter matrices (the graph path froze them
// in per-Step SliceRow nodes; reading the parents' live Data in the
// backward would diverge under a mid-flight parameter mutation). inputsD
// and swmD alias internal graph nodes (engine-frozen: no op mutates an
// existing tensor). erev/sErev deliberately alias the cell's reversal
// storage: the graph path's erevRows constants share that storage, so
// both paths observe an in-place polarity rewrite (Load) identically.
type ltcFusedStash struct {
	batch, units, unfolds, inDim int
	v                            []float32 // (unfolds+1) * batch*units
	num, den, inv                []float32 // unfolds * batch*units
	mu, sig                      []float32 // units*units, frozen
	sMu, sSig                    []float32 // inDim*units, frozen
	inputsD                      []float32 // [batch, inDim] internal node data (engine-frozen, aliased)
	swmD                         []float32 // [inDim, units] masked sensory weights (internal, aliased)
	erev                         []float32 // live alias of c.erev.Data.Data
	sErev                        []float32 // live alias of c.sErev.Data.Data
	ident                        *tensor.Tensor
	hv                           []float32            // unfold 0's dense cmT-product buffer, handed to the delivery node
	parents                      []*autograd.Variable // the fused node's parent list, captured at construction (a Parents() call per backward would allocate under its copy semantics)
}

// fusedUnfolds replaces Step's sensory synapse path and ODE unfold loop
// with a single fused op node: parents [cmT, h, hvN, inputs, gleakSp,
// vleak, mu, sigma, wM, sMu, sSigma, sWm], output the new membrane state
// [batch, units]. The parent order matters (see the file header): cmT
// first keeps the cm chain on the DFS descent (spine class), h second
// continues the state-chain descent, and hvN — the delivery node for the
// dense cmT-product state contribution, a parentless FusedOp whose
// backward replays the stashed buffer — sits in the slot the graph path's
// Hadamard(cmT, v) node completes in (right after the state descent,
// before numConst's subtree). The stage-19a boundary moved the sensory
// path inside the kernel: numConst and denS are gone from the graph, and
// everything that used to append during numConst's subtree expansion (the
// sensory drive's chain, the gleak chain, vleak, and — in the chained
// topology — the next step's input subgraph built from this step's
// output) now appends during the inputs parent's subtree expansion, at
// the same post-hvN slot, so the three documented interleavings of
// external state-gradient contributions with the kernel's h deliveries
// are preserved exactly. The denBase = ((cmT + gleakSp) + denS) + eps
// chain and the numConst = gleakSp⊙vleak + numS assembly run INSIDE the
// kernel (forward: the same per-element operations; backward: replayed),
// because the graph path's cmT gradient accumulates the chain's
// contribution BETWEEN the last unfold's and the first unfold's
// Hadamard(cmT, v) contributions — an interleaving no graph-level
// delivery can reproduce.
func (c *LTC) fusedUnfolds(cmT, h, inputs, gleakSp, sWm, wM *autograd.Variable) *autograd.Variable {
	batch := inputs.Data.Rows()
	units, unfolds, inDim := c.units, c.unfolds, c.inDim
	if h.Data.Dims() != 2 || h.Data.Rows() != batch || h.Data.Cols() != units {
		panic(fmt.Sprintf("nn.LTC.Step: state shape %v incompatible with [%d, %d]", h.Data.Shape, batch, units))
	}
	bu := batch * units
	st := &ltcFusedStash{
		batch: batch, units: units, unfolds: unfolds, inDim: inDim,
		v:   make([]float32, (unfolds+1)*bu),
		num: make([]float32, unfolds*bu),
		den: make([]float32, unfolds*bu),
		inv: make([]float32, unfolds*bu),
		mu:  append([]float32(nil), c.mu.Data.Data...),
		sig: append([]float32(nil), c.sigma.Data.Data...),
		sMu: append([]float32(nil), c.sMu.Data.Data...),
		sSig: append([]float32(nil),
			c.sSigma.Data.Data...),
		inputsD: inputs.Data.Data,
		swmD:    sWm.Data.Data,
		// Live aliases, not copies: match the graph's shared-storage
		// erevRows constants (see ltcFusedStash).
		erev:  c.erev.Data.Data,
		sErev: c.sErev.Data.Data,
	}
	st.ident = c.ident.Data
	copy(st.v[:bu], h.Data.Data)
	cmTD := cmT.Data.Data    // [1, units]
	glD := gleakSp.Data.Data // [units]
	wmD := wM.Data.Data
	shape := []int{batch, units}

	// The sensory drive, once (loop-invariant): the same block formula as
	// the recurrent side (sigmoid(σ⊙(pre−μ))⊙wm with the wiring mask
	// folded into wm), a +0-seeded ascending fold per block column, then
	// the normalizing identity-MatMul zero wash — num always, den only
	// multi-source (contract's single-source shortcut at inDim == 1, the
	// twin of the recurrent units == 1 shortcut).
	denS := make([]float32, bu)
	numS := make([]float32, bu)
	for i := 0; i < inDim; i++ {
		muR := st.sMu[i*units:]
		sigR := st.sSig[i*units:]
		wmR := st.swmD[i*units:]
		erR := st.sErev[i*units:]
		for b := 0; b < batch; b++ {
			row := b * units
			pv := st.inputsD[b*inDim+i]
			for j := 0; j < units; j++ {
				k := row + j
				// Every product that FEEDS AN ADDITION goes through
				// mul32's float64 conversion barrier (the arm64 FMA
				// argument of the header).
				sub := pv - muR[j]
				z := sigR[j] * sub
				s := sigmoid32(z)
				blk := mul32(s, wmR[j])
				denS[k] += blk
				numS[k] += mul32(blk, erR[j])
			}
		}
	}
	numS = normFoldIdentity(numS, shape, c.ident.Data)
	if inDim > 1 {
		denS = normFoldIdentity(denS, shape, c.ident.Data)
	}

	// numConst = gleakSp⊙vleak + numS (the graph's Hadamard-then-Add: the
	// product feeds the Add, so it carries the mul32 barrier) and
	// denBase's A1 plane ((cmT + gleakSp), one native add per element,
	// exactly the graph's broadcastBinary). vleak is read live at forward,
	// exactly as the graph read it.
	vleakD := c.vleak.Data.Data
	ncD := make([]float32, bu)
	for b := 0; b < batch; b++ {
		row := b * units
		for j := 0; j < units; j++ {
			ncD[row+j] = mul32(glD[j], vleakD[j]) + numS[row+j]
		}
	}
	dsD := denS
	a1 := make([]float32, units)
	for j := 0; j < units; j++ {
		a1[j] = cmTD[j] + glD[j]
	}
	// Fold scratch, re-zeroed per unfold: the zeroV-seeded fold. The
	// normalizing identity MatMul runs literally (tensor.MatMul against
	// the cell's ident): it is a value-preserving copy for finite folds
	// (x*1 = x, skipped zeros stay +0), and — unlike any per-element
	// shortcut — it also reproduces the av==0 loop's NaN spreading when
	// an overflowed fold entry meets the identity's zero coefficients
	// (Inf*0 = NaN across the whole row), the graph path's exact
	// behavior under adversarially loaded weights.
	denF := make([]float32, bu)
	numF := make([]float32, bu)
	for t := 0; t < unfolds; t++ {
		v := st.v[t*bu : (t+1)*bu]
		out := st.v[(t+1)*bu : (t+2)*bu]
		num := st.num[t*bu : (t+1)*bu]
		den := st.den[t*bu : (t+1)*bu]
		inv := st.inv[t*bu : (t+1)*bu]
		clear(denF)
		clear(numF)
		for i := 0; i < units; i++ {
			muR := st.mu[i*units:]
			sigR := st.sig[i*units:]
			wmR := wmD[i*units:]
			erR := st.erev[i*units:]
			for b := 0; b < batch; b++ {
				row := b * units
				vbi := v[row+i]
				for j := 0; j < units; j++ {
					k := row + j
					// Sub(preCol, muR), then Hadamard(sigR, sub),
					// then SigmoidHadamard's sigmoid⊙w. Every product
					// that FEEDS AN ADDITION goes through mul32's
					// float64 conversion barrier: this toolchain's FMA
					// formation reaches across statements (a*b whose
					// SSA value feeds a later + is contracted into
					// FMADDS on arm64), which would drop the graph
					// kernels' store/load rounding boundary and drift
					// ~1 ULP; the barrier is bit-identical to the
					// native product (see mul32).
					sub := vbi - muR[j]
					z := sigR[j] * sub
					s := sigmoid32(z)
					blk := mul32(s, wmR[j])
					denF[k] += blk
					prod := mul32(blk, erR[j])
					numF[k] += prod
				}
			}
		}
		// The contractions: the identity MatMul is the per-element zero
		// wash for finite folds (x*1 = x, skipped zeros stay +0 —
		// normFoldIdentity scans once and takes that fast path), and only
		// a fold holding +/-Inf or NaN (overflow via adversarially loaded
		// weights) runs the literal MatMul, whose av*0 = NaN spreading
		// the fast path does not reproduce. The single-source denominator
		// shortcut (units == 1) skips fold normalization entirely — denR
		// is the raw fold.
		numF = normFoldIdentity(numF, shape, c.ident.Data)
		if units > 1 {
			denF = normFoldIdentity(denF, shape, c.ident.Data)
		}
		for b := 0; b < batch; b++ {
			row := b * units
			for j := 0; j < units; j++ {
				k := row + j
				cmV := mul32(cmTD[j], v[k]) // barrier: the product feeds the Add chain (FMA hazard, see above)
				a := cmV + ncD[k]
				num[k] = a + numF[k]
				// den = ((A1 + denS) + eps) + denR: the graph's denBase
				// chain element for element, then the unfold's den_t Add.
				a2 := a1[j] + dsD[k]
				db := a2 + c.eps
				den[k] = db + denF[k]
				iv := float32(math.Pow(float64(den[k]), -1))
				inv[k] = iv
				out[k] = num[k] * iv
			}
		}
	}
	data := &tensor.Tensor{Shape: []int{batch, units}, Data: st.v[unfolds*bu:]}
	// hvN delivers the dense cmT-product buffer: its backward replays the
	// buffer the fused backward stashes, and its parentless subtree makes
	// it append to the topological order in the Hadamard(cmT, v) node's
	// slot (see the doc comment above), so its backward runs after any
	// contribution appended during numConst's expansion — a chained input
	// subgraph's, for one — exactly where the graph path's dense
	// contribution landed.
	hvN := autograd.FusedOp(tensor.New(1), nil, func(v *autograd.Variable) {
		addGrad(h, &tensor.Tensor{Shape: []int{batch, units}, Data: st.hv})
	})
	parents := []*autograd.Variable{cmT, h, hvN, inputs, gleakSp, c.vleak, c.mu, c.sigma, wM, c.sMu, c.sSigma, sWm}
	st.parents = parents
	return autograd.FusedOp(data, parents, st.backward)
}

// backward replays the replaced subgraph's backward sweep (see the file
// header for the order derivation). v.Grad holds the fully accumulated
// incoming state gradient [batch, units]. An irregular manually seeded
// Grad is replayed through the graph path's own terminal Div backward
// (seedDivBackward): the graph path broadcast-then-reduced such seeds per
// consuming op, mixing each operand into its own reduction, so no single
// broadcast [batch, units] plane reproduces the values — and a seed that
// broadcasts PAST [batch, units] before reducing back (outer products at
// units == 1, multi-row seeds at batch == 1) is not the first bu elements
// of the expanded buffer.
func (st *ltcFusedStash) backward(v *autograd.Variable) {
	batch, units, unfolds, inDim := st.batch, st.units, st.unfolds, st.inDim
	bu := batch * units
	p := st.parents // captured at construction; a v.Parents() call here would allocate per backward (copy semantics)
	cmT, h, inputs, gleakSp, vleak := p[0], p[1], p[3], p[4], p[5]
	mu, sigma, wM, sMu, sSigma, sWm := p[6], p[7], p[8], p[9], p[10], p[11]
	cmTD := cmT.Data.Data
	wmD := wM.Data.Data
	glD := gleakSp.Data.Data // internal node data (engine-frozen, aliased)

	// The last unfold's Div backward: the regular seed feeds the fused
	// replay directly; an irregular seed goes through the engine's own Div
	// backward over the stashed planes (values, delivery shapes and the
	// panic contract of the graph path's per-op broadcast-then-reduce).
	var dv []float32
	var irreg *irregSeed
	if g := v.Grad; len(g.Shape) == 2 && g.Shape[0] == batch && g.Shape[1] == units {
		dv = g.Data
	} else {
		irreg = seedDivBackward(g, st.num[(unfolds-1)*bu:unfolds*bu], st.den[(unfolds-1)*bu:unfolds*bu], batch, units)
		if len(irreg.dnum.Shape) != 2 {
			// The graph path's num-fold normalization is a genuine
			// MatMulTransB against the identity, whose shape check
			// rejects a non-2D numerator gradient (a flat seed over a
			// [1, 1] state) before any math — and before any leaf
			// delivery (the fold's backward step precedes every
			// leaf's in the sweep). Surface the identical panic here,
			// ahead of the kernel's first delivery.
			tensor.MatMulTransB(&tensor.Tensor{Shape: irreg.dnum.Shape, Data: irreg.dnum.Data}, st.ident)
		}
	}
	// Scratch planes, allocated ONCE per VJP and reused across unfolds and
	// presynaptic rows (stage 19b): each is either fully overwritten before
	// every use or explicitly cleared, and NONE is ever handed to addGrad,
	// wrapped in a delivered tensor, or stashed for the hvN delivery node.
	// Delivery buffers — the h scatter buffers, the SliceRow scatter
	// tensors, the gleakSp/vleak/cmT contributions, unfold 0's dense hv
	// buffer and the adopted cmT reductions — stay freshly allocated per
	// delivery, because addGrad's ownership transfer (first contribution
	// adopted, the caller may hold it afterwards) is part of the bitwise
	// contract.
	dnumS := make([]float32, bu)                                     // per-unfold numerator gradient
	gaS := make([]float32, bu)                                       // Div backward's ga, overwritten in place by divDenGradInto (becomes dden)
	washA := make([]float32, bu)                                     // per-unfold num-fold wash; reused for the sensory num wash after the loop
	washB := make([]float32, bu)                                     // per-unfold den-fold wash; reused for the sensory den wash after the loop
	hvS := make([]float32, bu)                                       // dense cmT-product buffer for unfolds >= 1 (unfold 0's is a fresh delivery)
	dvBufs := [2][]float32{make([]float32, bu), make([]float32, bu)} // propagated state gradient ping-pong
	rTS := make([]float32, units)                                    // middle-unfold cmT reduction; reused for the gLV reduction after the loop
	dpreS := make([]float32, batch)                                  // per-row pre-gradient (unfolds >= 1, then the sensory drive)
	dmS := make([]float32, units)                                    // per-row mu gradient (merged into muAcc)
	dsS := make([]float32, units)                                    // per-row sigma gradient (merged into sigAcc)
	dwS := make([]float32, units)                                    // per-row weight gradient (merged into wmAcc)
	colGArea := make([]float32, units*batch)                         // unfold-0 per-column holding for the h delivery
	muAcc := make([]float32, units*units)                            // cross-unfold row accumulators: the latest
	sigAcc := make([]float32, units*units)                           // unfold's row is COPY-adopted (bit-identical
	wmAcc := make([]float32, units*units)                            // to accumRow's first-buffer adoption, -0
	ncGH := make([]float32, bu)                                      // signs included), later unfolds +=
	denBaseGH := make([]float32, bu)                                 // denBase's accumulated gradient (same semantics)

	var cmTHi []float32 // cmT's Hadamard contributions of unfolds 1.. (ownership semantics)
	var rT0 []float32   // unfold 0's Hadamard contribution (delivered last)
	dvPar := 0          // dvBufs ping-pong parity
	for t := unfolds - 1; t >= 0; t-- {
		num := st.num[t*bu : (t+1)*bu]
		den := st.den[t*bu : (t+1)*bu]
		inv := st.inv[t*bu : (t+1)*bu]
		vPrev := st.v[t*bu : (t+1)*bu]

		// Div backward: da = g⊙inv, db = -(g⊙num)⊙den^-2. The
		// irregular-seed replay takes the engine's own deliveries for
		// the last unfold (see above), delivery shape included; the
		// regular path matches hadamardReduce's same-shape arm.
		var dnum, dden []float32
		if irreg != nil && t == unfolds-1 {
			dnum, dden = irreg.dnum.Data, irreg.dden.Data
		} else {
			dnum = dnumS
			ga := gaS
			for k := 0; k < bu; k++ {
				dnum[k] = dv[k] * inv[k]
				ga[k] = dv[k] * num[k]
			}
			dden = divDenGradInto(ga, den)
		}
		// numConst's gradient accumulates one cloned contribution per
		// unfold, reverse order (the graph's A_t Add backwards) — INSIDE
		// the kernel since 19a (numConst is no longer a graph node; its
		// chain is replayed after the loop). denBase's gradient
		// accumulates the same way, also inside the kernel. The first
		// (latest unfold's) plane is copy-adopted, bit-identical to the
		// ownership adoption including -0 signs.
		if t == unfolds-1 {
			copy(ncGH, dnum)
			copy(denBaseGH, dden)
		} else {
			for k := range ncGH {
				ncGH[k] += dnum[k]
			}
			for k := range denBaseGH {
				denBaseGH[k] += dden[k]
			}
		}

		// Hadamard(cmT, vPrev) backward: cmT's reduction (fused
		// product-reduce, mul32-barriered as in autograd), and the
		// dense state contribution g⊙cmT. With units == 1 the target
		// [1, 1] takes hadamardReduce's SCALAR arm (single-element
		// shapes are scalars): a flat [1] total, not a [1, 1] row.
		// The two ADOPTED reductions (the latest unfold's, which becomes
		// cmTHi, and unfold 0's rT0) are delivered to cmT afterwards —
		// they stay fresh; the middle unfolds' reductions are only
		// accumulated into cmTHi and take the reused scratch.
		var rT []float32
		if t == 0 || cmTHi == nil {
			rT = make([]float32, units)
		} else {
			rT = rTS
			clear(rT)
		}
		if units == 1 {
			for k := 0; k < bu; k++ {
				rT[0] += mul32(dnum[k], vPrev[k])
			}
		} else {
			for b := 0; b < batch; b++ {
				row := b * units
				for j := 0; j < units; j++ {
					rT[j] += mul32(dnum[row+j], vPrev[row+j])
				}
			}
		}
		if t == 0 {
			rT0 = rT
		} else if cmTHi == nil {
			cmTHi = rT
		} else {
			for j := range cmTHi {
				cmTHi[j] += rT[j]
			}
		}
		// The dense cmT-product buffer: unfold 0's is a fresh delivery
		// (stashed for the hvN node), the later unfolds' only feed the
		// state-gradient propagation and take the reused scratch.
		var hv []float32
		if t == 0 {
			hv = make([]float32, bu)
		} else {
			hv = hvS
		}
		for b := 0; b < batch; b++ {
			row := b * units
			for j := 0; j < units; j++ {
				hv[row+j] = dnum[row+j] * cmTD[j]
			}
		}

		// Fold gradients: MatMulTransB against the identity — the
		// zero-wash fast path for finite gradients, the literal call
		// only when a non-finite entry (non-finite seeds) demands the
		// loop's NaN spreading (see the forward). The single-source
		// denominator shortcut skips fold and MatMul alike.
		shape := []int{batch, units}
		numFG := normFoldIdentityTransBInto(washA, dnum, shape, st.ident)
		denFG := dden
		if units > 1 {
			denFG = normFoldIdentityTransBInto(washB, dden, shape, st.ident)
		}

		// Per-presynaptic block chains, descending order. The state
		// scatter semantics replicate the Col backwards' literal
		// buffers: with units >= 2 every column ends up zero-washed
		// (each element is owned by one buffer as +0 or receives +0
		// adds); with units == 1 the single buffer is owned as-is.
		var dvNew []float32
		if t != 0 {
			dvNew = dvBufs[dvPar]
			dvPar ^= 1
		}
		for i := units - 1; i >= 0; i-- {
			muR := st.mu[i*units:]
			sigR := st.sig[i*units:]
			wmR := wmD[i*units:]
			erR := st.erev[i*units:]
			// The per-row buffers are the reused scratch, cleared per
			// row; at unfold 0 the pre-gradient accumulates directly
			// into the colG holding for the h delivery below.
			var dpre []float32
			if t == 0 {
				dpre = colGArea[i*batch : (i+1)*batch]
			} else {
				dpre = dpreS
			}
			dmR := dmS
			dsR := dsS
			dwR := dwS
			clear(dpre)
			clear(dmR)
			clear(dsR)
			clear(dwR)
			for b := 0; b < batch; b++ {
				row := b * units
				vbi := vPrev[row+i]
				for j := 0; j < units; j++ {
					k := row + j
					// Recompute the block's forward values from the
					// frozen operands (same expression structure as
					// the forward, hence the same bits).
					sub := vbi - muR[j]
					z := sigR[j] * sub
					s := sigmoid32(z)
					// Block gradient: den-fold values, then += the
					// num-fold's g⊙erev product (ownership order of
					// the graph's fold backwards). The product feeds
					// the addition, so it goes through mul32's barrier
					// (cross-statement FMA hazard, see the forward).
					gNum := mul32(numFG[k], erR[j])
					g := denFG[k] + gNum
					// SigmoidHadamard backward (regular 2D path).
					gw := mul32(g, wmR[j])
					dz := mul32(gw, mul32(s, 1-s))
					dsub := mul32(dz, sigR[j]) // barrier: feeds the SumCols add
					// The row accumulators replicate the graph's
					// ownership semantics: with batch == 1 the z and
					// SigmoidHadamard reductions take hadamardReduce's
					// (and negReduce's) SAME-SHAPE arm — the buffer is
					// adopted, not summed into a +0-seeded one — so a
					// -0 contribution keeps its sign bit instead of
					// washing to +0 ((+0)+(-0) = +0). With units == 1
					// the Sub backward's preCol side is a passthrough
					// (SumToShape's same-shape arm): dsub is adopted
					// per row, at any batch.
					if batch == 1 {
						dwR[j] = mul32(g, s)
						dsR[j] = mul32(dz, sub)
						dmR[j] = mul32(dsub, negOne)
					} else {
						dwR[j] += mul32(g, s)
						dsR[j] += mul32(dz, sub)
						dmR[j] += mul32(dsub, negOne)
					}
					if units == 1 {
						dpre[b] = dsub
					} else {
						dpre[b] += dsub
					}
				}
			}
			// The cross-unfold merge: the latest unfold's row is
			// copy-adopted (accumRow's first-buffer adoption, bit for
			// bit), later unfolds +=.
			muRow := muAcc[i*units : (i+1)*units]
			sigRow := sigAcc[i*units : (i+1)*units]
			wmRow := wmAcc[i*units : (i+1)*units]
			if t == unfolds-1 {
				copy(muRow, dmR)
				copy(sigRow, dsR)
				copy(wmRow, dwR)
			} else {
				for j := range muRow {
					muRow[j] += dmR[j]
					sigRow[j] += dsR[j]
					wmRow[j] += dwR[j]
				}
			}
			if t == 0 {
				// dpre already landed in the colG holding above.
			} else if units == 1 {
				copy(dvNew, dpre)
			} else {
				for b := 0; b < batch; b++ {
					dvNew[b*units+i] = washZero(dpre[b])
				}
			}
		}
		if t == 0 {
			// h delivery: the literal scatter buffers, descending; the
			// dense cmT-product buffer is stashed for the hvN delivery
			// node, whose backward replays it at the Hadamard(cmT, v)
			// node's DFS slot (see fusedUnfolds and the file header).
			for i := units - 1; i >= 0; i-- {
				buf := tensor.New(batch, units)
				for b := 0; b < batch; b++ {
					buf.Data[b*units+i] = colGArea[i*batch+b]
				}
				addGrad(h, buf)
			}
			st.hv = hv
		} else {
			for k := range dvNew {
				dvNew[k] += hv[k]
			}
			dv = dvNew
		}
	}

	// Row-scatter deliveries, descending (the SliceRow backwards). The
	// scatter tensors are fresh deliveries; the accumulator rows are only
	// read into them.
	for i := units - 1; i >= 0; i-- {
		scatter := func(acc []float32) *tensor.Tensor {
			g := tensor.New(units, units)
			copy(g.Data[i*units:(i+1)*units], acc)
			return g
		}
		addGrad(mu, scatter(muAcc[i*units:(i+1)*units]))
		addGrad(sigma, scatter(sigAcc[i*units:(i+1)*units]))
		addGrad(wM, scatter(wmAcc[i*units:(i+1)*units]))
	}

	// The denBase chain replay: denBase = Add(A2, epsV) hands A2 the
	// accumulated gradient unchanged (epsV's scalar sum is unobservable);
	// A2 = Add(A1, denS) reduces it to A1's shape and clones it to denS;
	// A1 = Add(cmT, gleakSp) hands cmT the reduction and gleakSp the row.
	// With units == 1 A1's [1, 1] target is scalar-shaped: the reduction
	// is a flat SumAll into [1] (see the Hadamard reduction above).
	var gA1 []float32
	if units == 1 {
		gA1 = make([]float32, 1)
		for k := 0; k < bu; k++ {
			gA1[0] += denBaseGH[k]
		}
	} else {
		gA1 = make([]float32, units)
		for b := 0; b < batch; b++ {
			row := b * units
			for j := 0; j < units; j++ {
				gA1[j] += denBaseGH[row+j]
			}
		}
	}
	glShape := []int{units}
	if units == 1 {
		glShape = []int{1}
	}
	addGrad(gleakSp, &tensor.Tensor{Shape: glShape, Data: append([]float32(nil), gA1...)})

	// The sensory side of the 19a boundary (numConst/denS are internal
	// since 19a): denS's gradient is A2's b-branch clone of denBaseG
	// (identical values), whose fold backward is the MatMulTransB zero
	// wash for finite gradients (literal call only for non-finite ones)
	// — multi-source only; the inDim == 1 single-source shortcut keeps
	// the raw incoming gradient with no wash (the F9-1 corner). numS's
	// gradient is numConst's b-branch clone of ncG (identical values);
	// num always folds and normalizes, so its wash always runs. Both
	// washes write into the (dead after the loop) wash scratch planes.
	shape := []int{batch, units}
	denFGS := denBaseGH
	if inDim > 1 {
		denFGS = normFoldIdentityTransBInto(washB, denBaseGH, shape, st.ident)
	}
	numFGS := normFoldIdentityTransBInto(washA, ncGH, shape, st.ident)

	// numConst = Add(glvHad, numS): the glv side's incoming gradient is
	// ncG reduced to [1, units] — passthrough at batch == 1
	// (sumToShapeTake's same-shape arm), a flat SumAll at units == 1 (the
	// scalar arm), SumRows otherwise. The SumRows plane reuses the (dead)
	// cmT-reduction scratch, cleared first.
	gLV := ncGH
	switch {
	case units == 1:
		var s float32
		for _, x := range ncGH {
			s += x
		}
		gLV = []float32{s}
	case batch > 1:
		clear(rTS)
		for b := 0; b < batch; b++ {
			for j := 0; j < units; j++ {
				rTS[j] += ncGH[b*units+j]
			}
		}
		gLV = rTS
	}

	// The sensory drive's block chains, descending presynaptic order (the
	// same per-element formulas as the recurrent side: SigmoidHadamard's
	// fused dz/dw, the z Hadamard, the Sub, the Col scatter). Within one
	// row the delivery order is pre-col scatter, w, sigma, mu (the reverse
	// of the per-row append order); the sensory chain sweeps before the
	// glvHad node below (its append position inside the numConst
	// subtree), hence these deliveries precede gleakSp's second
	// contribution and vleak's. The per-row buffers reuse the (dead after
	// the loop) scratch, cleared per row; every delivered tensor is fresh.
	swmD := st.swmD
	for i := inDim - 1; i >= 0; i-- {
		muR := st.sMu[i*units:]
		sigR := st.sSig[i*units:]
		wmR := swmD[i*units:]
		erR := st.sErev[i*units:]
		dpre := dpreS
		dmR := dmS
		dsR := dsS
		dwR := dwS
		clear(dpre)
		clear(dmR)
		clear(dsR)
		clear(dwR)
		for b := 0; b < batch; b++ {
			row := b * units
			pv := st.inputsD[b*inDim+i]
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
				gNum := mul32(numFGS[k], erR[j])
				g := denFGS[k] + gNum
				// SigmoidHadamard backward (regular 2D path).
				gw := mul32(g, wmR[j])
				dz := mul32(gw, mul32(s, 1-s))
				dsub := mul32(dz, sigR[j]) // barrier: feeds the SumCols add
				// The row accumulators replicate the graph's ownership
				// semantics: with batch == 1 the reductions ADOPT their
				// product buffers (the signed-zero corners); otherwise
				// they accumulate into a +0-seeded row buffer.
				if batch == 1 {
					dwR[j] = mul32(g, s)
					dsR[j] = mul32(dz, sub)
					dmR[j] = mul32(dsub, negOne)
				} else {
					dwR[j] += mul32(g, s)
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
		// The Col backward's literal scatter buffer into inputs, then the
		// SliceRow backwards' zero-padded row buffers.
		buf := tensor.New(batch, inDim)
		for b := 0; b < batch; b++ {
			buf.Data[b*inDim+i] = dpre[b]
		}
		addGrad(inputs, buf)
		addGrad(sWm, &tensor.Tensor{Shape: []int{inDim, units}, Data: cfcScatterRow(inDim, units, i, dwR)})
		addGrad(sSigma, &tensor.Tensor{Shape: []int{inDim, units}, Data: cfcScatterRow(inDim, units, i, dsR)})
		addGrad(sMu, &tensor.Tensor{Shape: []int{inDim, units}, Data: cfcScatterRow(inDim, units, i, dmR)})
	}

	// Hadamard(gleakSp, vleak) backward (the numConst subtree's glv node):
	// gleakSp's SECOND contribution (gLV⊙vleak) and vleak's only one
	// (gLV⊙gleakSp) — the row/scalar reduction arms with mul32-rounded
	// products and a +0-seeded accumulator. vleak is read LIVE here,
	// exactly as the graph's Hadamard backward reads the live leaf.
	vleakD := vleak.Data.Data
	gleakG2 := make([]float32, units)
	vleakG := make([]float32, units)
	if units == 1 {
		for k, x := range gLV {
			gleakG2[0] += mul32(x, vleakD[k])
			vleakG[0] += mul32(x, glD[k])
		}
	} else {
		for i := 0; i < len(gLV)/units; i++ {
			for j := 0; j < units; j++ {
				gleakG2[j] += mul32(gLV[i*units+j], vleakD[j])
				vleakG[j] += mul32(gLV[i*units+j], glD[j])
			}
		}
	}
	addGrad(gleakSp, &tensor.Tensor{Shape: glShape, Data: gleakG2})
	addGrad(vleak, &tensor.Tensor{Shape: glShape, Data: vleakG})

	// cmT's contributions land in the graph path's exact order: the
	// Hadamard contributions of unfolds T..2, the denBase chain's, then
	// unfold 1's Hadamard contribution (see the file header). The [1]
	// scalar-arm shape for units == 1 (see the reduction above).
	cmTShape := []int{1, units}
	if units == 1 {
		cmTShape = []int{1}
	}
	if cmTHi != nil {
		addGrad(cmT, &tensor.Tensor{Shape: cmTShape, Data: cmTHi})
	}
	addGrad(cmT, &tensor.Tensor{Shape: cmTShape, Data: gA1})
	addGrad(cmT, &tensor.Tensor{Shape: cmTShape, Data: rT0})
}

// irregSeed holds the graph path's terminal-Div backward deliveries for
// an irregular manually seeded gradient on the fused node's output:
// dnum/dden are the last unfold's num/den gradient planes (always
// batch*units elements), with dnum carrying the exact delivery shape the
// graph path hands the numerator chain (a flat [1] in the scalar corners
// — shape is part of the backward contract).
type irregSeed struct {
	dnum, dden *tensor.Tensor
}

// seedDivBackward replays the replaced subgraph's terminal Div backward
// for an irregular seeded gradient: a genuine Div node over the last
// unfold's stashed num/den planes, differentiated with the seed. The
// graph path broadcast-then-reduced such seeds per consuming op
// (hadamardReduce's fused arms and its legacy fallback), mixing each
// operand into its own reduction — ((g₁+g₂)·x is not g₁·x+g₂·x in
// float32) — so no pre-reduced [batch, units] plane can reproduce both
// branches; replaying the op itself reproduces the values, the delivery
// shapes and the broadcast panic contract exactly, by construction.
func seedDivBackward(seed *tensor.Tensor, num, den []float32, batch, units int) *irregSeed {
	shape := []int{batch, units}
	numV := autograd.Var(&tensor.Tensor{Shape: shape, Data: num})
	denV := autograd.Var(&tensor.Tensor{Shape: shape, Data: den})
	out := autograd.Div(numV, denV)
	out.Grad = seed
	out.Backward()
	return &irregSeed{dnum: numV.Grad, dden: denV.Grad}
}

// addGrad replicates autograd's gradient accumulation (ownership on the
// first contribution, elementwise += afterwards, shape-mismatch panic)
// for the fused backward's deliveries; the message matches the engine's.
func addGrad(v *autograd.Variable, g *tensor.Tensor) {
	if v.Grad == nil {
		v.Grad = g
		return
	}
	if !tensor.SameShape(v.Grad, g) {
		panic(fmt.Sprintf("autograd: gradient shape mismatch: accumulated %v vs incoming %v", v.Grad.Shape, g.Shape))
	}
	for i := range v.Grad.Data {
		v.Grad.Data[i] += g.Data[i]
	}
}

// divDenGrad replicates autograd's negHadamardPow2: -(ga)⊙pow(den, -2)
// with the power through float64 math.Pow and the final sign flip as a
// genuine multiply (NaN-propagating, like the legacy tensor.Neg).
func divDenGrad(ga, den []float32) []float32 {
	r := make([]float32, len(ga))
	for i := range r {
		pb2 := float32(math.Pow(float64(den[i]), -2))
		m := ga[i] * pb2
		r[i] = mul32(m, negOne)
	}
	return r
}

// divDenGradInto is divDenGrad writing its result into ga in place
// (strictly elementwise: each output depends only on the same index's
// inputs), the stage-19b scratch-reuse form. ga must not be read for its
// old values afterwards — and it is not: the per-unfold ga feeds only
// this computation.
func divDenGradInto(ga, den []float32) []float32 {
	for i := range ga {
		pb2 := float32(math.Pow(float64(den[i]), -2))
		m := ga[i] * pb2
		ga[i] = mul32(m, negOne)
	}
	return ga
}

// sigmoid32 replicates tensor.sigmoid's numerically stable logistic
// sigmoid bit for bit (the tensor kernel is unexported).
func sigmoid32(x float32) float32 {
	if x >= 0 {
		return 1 / (1 + float32(math.Exp(float64(-x))))
	}
	e := float32(math.Exp(float64(x)))
	return e / (1 + e)
}

// washZero encodes the Col-backward scatter's effect on one state
// element: owned as a +0 buffer entry or receiving +0 adds, so a -0
// contribution maps to +0 and every other value (including +/-Inf and
// NaN: x + (+0) = x) passes through unchanged.
func washZero(x float32) float32 {
	if x == 0 {
		return 0
	}
	return x
}

// hasNonFinite reports whether fold holds any +/-Inf or NaN.
func hasNonFinite(fold []float32) bool {
	for _, x := range fold {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return true
		}
	}
	return false
}

// normFoldIdentity applies the contraction's normalizing identity MatMul
// to a fold plane. For finite folds the MatMul is the per-element zero
// wash (a nonzero x contributes x*1 = x through the single nonzero
// identity coefficient; a +/-0 is skipped by the av==0 branch and the
// +0 accumulator remains), so the scan takes that fast path. Only a
// fold holding +/-Inf or NaN takes the literal MatMul: there the
// identity's zero coefficients turn the non-finite entry into NaN
// across the whole row (Inf*0), which no per-element rule reproduces.
func normFoldIdentity(fold []float32, shape []int, ident *tensor.Tensor) []float32 {
	if hasNonFinite(fold) {
		return tensor.MatMul(&tensor.Tensor{Shape: shape, Data: fold}, ident).Data
	}
	for k, x := range fold {
		fold[k] = washZero(x)
	}
	return fold
}

// normFoldIdentityTransB is normFoldIdentity for the backward's
// MatMulTransB: the same finite-value zero wash, the same literal
// fallback for non-finite gradients (reachable through non-finite
// seeded gradients). The input plane is never mutated.
func normFoldIdentityTransB(grad []float32, shape []int, ident *tensor.Tensor) []float32 {
	if hasNonFinite(grad) {
		return tensor.MatMulTransB(&tensor.Tensor{Shape: shape, Data: grad}, ident).Data
	}
	r := make([]float32, len(grad))
	for k, x := range grad {
		r[k] = washZero(x)
	}
	return r
}

// normFoldIdentityTransBInto is normFoldIdentityTransB writing the wash
// into the caller's (reused) dst plane — the stage-19b scratch-reuse
// form. dst must not alias grad. The non-finite fallback still allocates
// the literal MatMulTransB result (the rare adversarial path).
func normFoldIdentityTransBInto(dst, grad []float32, shape []int, ident *tensor.Tensor) []float32 {
	if hasNonFinite(grad) {
		return tensor.MatMulTransB(&tensor.Tensor{Shape: shape, Data: grad}, ident).Data
	}
	for k, x := range grad {
		dst[k] = washZero(x)
	}
	return dst
}

// mul32 replicates autograd's float32 multiply with the explicit
// float64 rounding barrier (finite operands), which bars arm64 FMA
// fusion across statements exactly as in the engine's fused backward
// loops; non-finite operands take the native multiply so NaN payloads
// propagate the hardware way.
func mul32(a, b float32) float32 {
	if math.Float32bits(a)&0x7F800000 == 0x7F800000 ||
		math.Float32bits(b)&0x7F800000 == 0x7F800000 {
		return a * b
	}
	return float32(float64(a) * float64(b))
}

// negOne carries the -1 of the negating multiply, a variable (not a
// constant) so the compiler keeps a genuine NaN-propagating hardware
// multiply instead of a sign-flip — mirroring autograd's negOne.
var negOne float32 = -1
