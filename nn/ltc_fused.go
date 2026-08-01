package nn

import (
	"fmt"
	"math"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/tensor"
)

// This file holds the LTC's fused ODE-unfold kernel (technical-debt item
// 16b). Step's unfolds loop used to record ~6*units graph nodes per unfold
// (Col/Sub/Hadamard/SigmoidHadamard per presynaptic neuron, two fold
// chains, two normalizing MatMuls, the membrane Add/Div) — the stage-15
// profiling showed ~80% of the step's wall clock was per-node interpreter
// overhead, not math. The fused kernel replaces the whole loop with ONE
// autograd.FusedOp node: the forward runs the identical float32 operation
// sequence in a single loop nest, and the backward is a hand-written VJP
// that replays the replaced subgraph's backward sweep contribution for
// contribution.
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
//   - numConst / denBase receive one contribution per unfold (the cloned
//     numerator gradient / the denominator gradient), reverse unfold
//     order — their graph-level backwards then distribute to the sensory
//     and leak chains bit-identically, because those chains ARE the graph.
//   - mu, sigma and wM receive one zero-padded row-scatter buffer per
//     presynaptic row, in descending row order (the SliceRow backwards).
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
//     third associations; the delivery node reproduces all three.
//   - cmT receives its contributions in the graph path's exact order:
//     the Hadamard(cmT, v) reductions of unfolds T..2, then the denBase
//     chain's reduction, then unfold 1's Hadamard reduction — the denBase
//     chain's contribution interleaves BETWEEN the Hadamard ones in the
//     graph path (the chain's backward runs inside the last unfold's
//     sweep segment), which is why the chain lives inside the kernel
//     (gleakSp/denS are the parents; its backward is replayed here:
//     gleakSp gets its row first, matching the chain-before-numConst
//     order, and denS gets its single clone). The cm chain itself is
//     graph-level (the fused node's first parent), so cm keeps its spine
//     fold class and its UnrollRemat σ-sweep behavior is untouched.
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
// unfolds+1: the input state plus every substep result); mu/sig are
// frozen copies of the recurrent parameter matrices (the graph path
// froze them in per-Step SliceRow nodes; reading the parents' live Data
// in the backward would diverge under a mid-flight parameter mutation).
// erev deliberately aliases the cell's reversal storage: the graph path's
// erevRows constants share that storage, so both paths observe an
// in-place polarity rewrite (Load) identically.
type ltcFusedStash struct {
	batch, units, unfolds int
	v                     []float32 // (unfolds+1) * batch*units
	num, den, inv         []float32 // unfolds * batch*units
	mu, sig               []float32 // units*units, frozen
	erev                  []float32 // live alias of c.erev.Data.Data
	ident                 *tensor.Tensor
	hv                    []float32            // unfold 0's dense cmT-product buffer, handed to the delivery node
	parents               []*autograd.Variable // the fused node's parent list, captured at construction (a Parents() call per backward would allocate under its copy semantics)
}

// fusedUnfolds replaces Step's ODE unfold loop with a single fused op
// node: parents [cmT, h, hvN, numConst, gleakSp, denS, mu, sigma, wM],
// output the new membrane state [batch, units]. The parent order matters
// (see the file header): cmT first keeps the cm chain on the DFS descent
// (spine class), h second continues the state-chain descent, and hvN —
// the delivery node for the dense cmT-product state contribution, a
// parentless FusedOp whose backward replays the stashed buffer — sits in
// the slot the graph path's Hadamard(cmT, v) node completes in (right
// after the state descent, before numConst's subtree), so external
// contributions to the state gradient interleave with the kernel's h
// deliveries exactly as in the graph path. The denBase =
// ((cmT + gleakSp) + denS) + eps chain runs INSIDE the kernel (forward:
// the same per-element adds; backward: replayed), because the graph
// path's cmT gradient accumulates the chain's contribution BETWEEN the
// last unfold's and the first unfold's Hadamard(cmT, v) contributions —
// an interleaving no graph-level delivery can reproduce.
func (c *LTC) fusedUnfolds(cmT, h, numConst, gleakSp, denS, wM *autograd.Variable) *autograd.Variable {
	batch := numConst.Data.Rows()
	units, unfolds := c.units, c.unfolds
	if h.Data.Dims() != 2 || h.Data.Rows() != batch || h.Data.Cols() != units {
		panic(fmt.Sprintf("nn.LTC.Step: state shape %v incompatible with [%d, %d]", h.Data.Shape, batch, units))
	}
	bu := batch * units
	st := &ltcFusedStash{
		batch: batch, units: units, unfolds: unfolds,
		v:   make([]float32, (unfolds+1)*bu),
		num: make([]float32, unfolds*bu),
		den: make([]float32, unfolds*bu),
		inv: make([]float32, unfolds*bu),
		mu:  append([]float32(nil), c.mu.Data.Data...),
		sig: append([]float32(nil), c.sigma.Data.Data...),
		// Live alias, not a copy: matches the graph's shared-storage
		// erevRows constants (see ltcFusedStash).
		erev: c.erev.Data.Data,
	}
	st.ident = c.ident.Data
	copy(st.v[:bu], h.Data.Data)
	cmTD := cmT.Data.Data     // [1, units]
	ncD := numConst.Data.Data // [batch, units]
	glD := gleakSp.Data.Data  // [units]
	dsD := denS.Data.Data     // [batch, units]
	wmD := wM.Data.Data
	// denBase's A1 plane ((cmT + gleakSp), one native add per element,
	// exactly the graph's broadcastBinary).
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
	shape := []int{batch, units}
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
	parents := []*autograd.Variable{cmT, h, hvN, numConst, gleakSp, denS, c.mu, c.sigma, wM}
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
	batch, units, unfolds := st.batch, st.units, st.unfolds
	bu := batch * units
	p := st.parents // captured at construction; a v.Parents() call here would allocate per backward (copy semantics)
	cmT, h, numConst, gleakSp, denS := p[0], p[1], p[3], p[4], p[5]
	mu, sigma, wM := p[6], p[7], p[8]
	cmTD := cmT.Data.Data
	wmD := wM.Data.Data

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
	var cmTHi []float32    // cmT's Hadamard contributions of unfolds 1.. (ownership semantics)
	var rT0 []float32      // unfold 0's Hadamard contribution (delivered last)
	var denBaseG []float32 // denBase's accumulated gradient (ownership semantics)
	muRG := make([][]float32, units)
	sigRG := make([][]float32, units)
	wmRG := make([][]float32, units)
	colG := make([][]float32, units) // per-column state scatters of unfold 0 (h delivery)
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
		dnumShape := []int{batch, units}
		if irreg != nil && t == unfolds-1 {
			dnum, dden = irreg.dnum.Data, irreg.dden.Data
			dnumShape = irreg.dnum.Shape
		} else {
			dnum = make([]float32, bu)
			ga := make([]float32, bu)
			for k := 0; k < bu; k++ {
				dnum[k] = dv[k] * inv[k]
				ga[k] = dv[k] * num[k]
			}
			dden = divDenGrad(ga, den)
		}
		// numConst gets one cloned contribution per unfold, reverse
		// order (the graph's A_t Add backwards); denBase's gradient
		// accumulates the same way but stays inside the kernel (the
		// chain is replayed after the loop).
		addGrad(numConst, &tensor.Tensor{Shape: dnumShape, Data: append([]float32(nil), dnum...)})
		if denBaseG == nil {
			denBaseG = dden
		} else {
			for k := range denBaseG {
				denBaseG[k] += dden[k]
			}
		}

		// Hadamard(cmT, vPrev) backward: cmT's reduction (fused
		// product-reduce, mul32-barriered as in autograd), and the
		// dense state contribution g⊙cmT. With units == 1 the target
		// [1, 1] takes hadamardReduce's SCALAR arm (single-element
		// shapes are scalars): a flat [1] total, not a [1, 1] row.
		var rT []float32
		if units == 1 {
			rT = make([]float32, 1)
			for k := 0; k < bu; k++ {
				rT[0] += mul32(dnum[k], vPrev[k])
			}
		} else {
			rT = make([]float32, units)
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
		hv := make([]float32, bu)
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
		numFG := normFoldIdentityTransB(dnum, shape, st.ident)
		denFG := dden
		if units > 1 {
			denFG = normFoldIdentityTransB(dden, shape, st.ident)
		}

		// Per-presynaptic block chains, descending order. The state
		// scatter semantics replicate the Col backwards' literal
		// buffers: with units >= 2 every column ends up zero-washed
		// (each element is owned by one buffer as +0 or receives +0
		// adds); with units == 1 the single buffer is owned as-is.
		dvNew := make([]float32, bu)
		for i := units - 1; i >= 0; i-- {
			muR := st.mu[i*units:]
			sigR := st.sig[i*units:]
			wmR := wmD[i*units:]
			erR := st.erev[i*units:]
			dpre := make([]float32, batch)
			dmR := make([]float32, units)
			dsR := make([]float32, units)
			dwR := make([]float32, units)
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
			muRG[i] = accumRow(muRG[i], dmR)
			sigRG[i] = accumRow(sigRG[i], dsR)
			wmRG[i] = accumRow(wmRG[i], dwR)
			if t == 0 {
				colG[i] = dpre
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
					buf.Data[b*units+i] = colG[i][b]
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

	// Row-scatter deliveries, descending (the SliceRow backwards).
	for i := units - 1; i >= 0; i-- {
		scatter := func(acc []float32) *tensor.Tensor {
			g := tensor.New(units, units)
			copy(g.Data[i*units:(i+1)*units], acc)
			return g
		}
		addGrad(mu, scatter(muRG[i]))
		addGrad(sigma, scatter(sigRG[i]))
		addGrad(wM, scatter(wmRG[i]))
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
			gA1[0] += denBaseG[k]
		}
	} else {
		gA1 = make([]float32, units)
		for b := 0; b < batch; b++ {
			row := b * units
			for j := 0; j < units; j++ {
				gA1[j] += denBaseG[row+j]
			}
		}
	}
	glShape := []int{units}
	if units == 1 {
		glShape = []int{1}
	}
	addGrad(gleakSp, &tensor.Tensor{Shape: glShape, Data: append([]float32(nil), gA1...)})
	addGrad(denS, &tensor.Tensor{Shape: []int{batch, units}, Data: append([]float32(nil), denBaseG...)})

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

// accumRow merges one unfold's row contribution into a shared row
// accumulator with the graph's ownership semantics: the first (latest
// unfold's) buffer is adopted, later ones are added elementwise.
func accumRow(acc, row []float32) []float32 {
	if acc == nil {
		return row
	}
	for j := range acc {
		acc[j] += row[j]
	}
	return acc
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
