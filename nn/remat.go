package nn

import (
	"fmt"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/tensor"
)

// UnrollRemat drives cell over the input sequence xs exactly like Unroll,
// then differentiates lossFn(ys) through time with rematerialization
// (gradient checkpointing) instead of one whole-graph Backward. The result
// is the contract of "Unroll, build the loss, call loss.Backward()" —
// every gradient accumulates into the leaves bit for bit as the
// whole-graph backward would accumulate it — while peak graph memory drops
// from O(len(xs) × per-step graph) to O(chunkSize × per-step graph) plus
// the detached per-step outputs and states (one [batch, StateSize()]
// tensor per step). The drop is not guaranteed: adversarial loss visit
// orders can defeat it entirely (see the chunkSize bullet under Contract
// details).
//
//	params := nn.ParametersOf(cell, readout)
//	for _, p := range params {
//		p.ZeroGrad()
//	}
//	ys, hN, loss := nn.UnrollRemat(cell, params, xs, nil, 1.0, 8, lossFn)
//	// gradients are now in the leaves, as after loss.Backward()
//
// params must list EVERY trainable leaf the cell's Step consumes
// (typically ParametersOf of the model — cell and loss-side modules
// alike): the reconciliation pick below restores the listed leaves
// individually, and an unlisted leaf would not be restored between sweeps,
// silently accumulating one step-side gradient copy per sweep (a 2-3x
// wrong value). The structural probe therefore enforces completeness:
// when the cell exposes Parameters() (both provided cells do), any
// Step-consumed parameter missing from the list panics, named by index;
// a cell that does not expose the list opts out of the audit, and
// completeness is the caller's contract. Leaves consumed only by the loss
// (a readout's) need no listing but may be listed; xs and h0 gradients
// always participate, they need no listing.
//
// # Why one backward is not enough: the whole-graph backward's fold order
//
// A bit-exact remat must replay, for every leaf, the exact sequence in
// which the whole-graph backward accumulates that leaf's gradient —
// float32 addition is not associative. Let π be the order in which the
// loss's DFS first visits the seeded outputs (read off the loss graph's
// own topological order), and call a seeded step a record high when it is
// larger than every seeded step visited before it. The unroll graph's DFS
// post-order has the recursive shape S(k) = [spine(k), S(k-1), rest(k)],
// where spine(k) is the part of step k's subgraph the build pass reaches
// while DESCENDING the state chain (in the LTC, the cm capacitance chain,
// consumed by Hadamard(cmT, v) with cmT as first parent) and rest(k) is
// everything else. Expanding the recursion, the whole backward folds:
//
//   - state-subgraph contributions (rest class) in strictly REVERSE step
//     order, for every loss shape;
//   - output-affine contributions (output class — the operands of the
//     step's output branch, e.g. outW/outB) in REVERSE π order over the
//     seeded steps;
//   - spine contributions (spine class — the LTC's cm) run by run of the
//     record highs in reverse visit order, each run ascending.
//
// And inside the state gradient of a seeded step k, the output affine's
// contribution lands BEFORE the later steps' contribution when k is not a
// record high (its y is visited after the descent that already covered
// step k+1), and after it when k is — an association no precomputed
// state-gradient tensor can reproduce, because ((A+r₁)+r₂) differs from
// (A+(r₁+r₂)) by one rounding. That association is exact only when steps
// k and k+1 are recomputed in ONE graph with the state node internal.
//
// # The algorithm
//
//  1. Forward step by step (chunkSize groups the walk but every step's
//     state is Detached, so the live graph never exceeds one step),
//     keeping the detached outputs and per-step states. The detached
//     outputs feed lossFn — called exactly once, so a lossFn closing over
//     parameters (a regularizer) behaves as over a full Unroll — and
//     loss.Backward() yields the per-step output seeds, bit-identical to
//     the whole-graph backward's.
//  2. A structural probe (one extra Step plus one TopoOrder walk) sorts
//     the swept leaves into the three fold classes, and the loss graph's
//     TopoOrder yields π. The contributing steps [0, max(π)] are cut into
//     recompute units: record-high runs split at chunkSize boundaries,
//     never between a non-record-high seeded step and its successor (the
//     association above — such a boundary would corrupt the state
//     gradient's fold by one rounding; adversarial visit orders can force
//     units longer than chunkSize, documented below).
//  3. Rest sweep: units are recomputed in strictly reverse step order and
//     backpropagated, threading the state gradient through the detached
//     state leaves. Each unit's synthetic scalar root lists its seed
//     terms so the unit backward replays the whole backward's restricted
//     sequence: the descent-triggering term first (the unit's record-high
//     output term if it has one, else the state term), the other output
//     terms in π order, and the state term last when a record-high term
//     leads. This delivers the rest class's final gradients and exact
//     per-boundary state gradients, cloned aside.
//  4. σ sweep (only when the probe found spine-class leaves — always for
//     the LTC, never for the CfC): the same unit backwards replayed with
//     units in σ order — runs in reverse visit order, units ascending
//     within each run — injecting the cloned boundary state gradients.
//  5. Affine pass (only when the probe found output-class leaves and π
//     does not ascend): one y-seeded backward per seeded step in reverse
//     π order, replaying the output branch's fold.
//  6. The pick: each swept leaf keeps the sweep result of its class;
//     loss-side contributions are snapshotted and restored around the
//     later sweeps so leaves consumed by the loss itself (a regularizer)
//     keep those contributions under every sweep (the loss-order
//     precondition below guarantees the whole backward really does
//     deliver them first).
//
// Costs: a CfC unroll with an ascending-π loss takes the rest sweep alone
// (two forwards, one backward — the ideal remat price); the LTC adds the
// σ sweep (three forwards, two backwards); a non-ascending loss adds the
// affine pass for both. Cells with no spine-class leaf never pay for σ.
//
// # Contract details
//
//   - The returned ys are the DETACHED per-step outputs (no graph behind
//     them): their Data is bit-identical to Unroll's outputs, and after
//     the call each ys[i].Grad holds the loss-side seed for step i (nil
//     when the loss does not depend on it). hN is the detached final
//     state, or h0 unchanged for an empty xs. Both are safe to read and
//     to feed into further computation; differentiating through them into
//     the consumed graph is impossible by construction.
//   - loss is lossFn's return value, kept for inspection (loss.Value());
//     its backward has already run.
//   - Gradients accumulate into the leaves as in Backward: ZeroGrad
//     beforehand is the caller's job, and reruns accumulate linearly.
//   - chunkSize < 1 panics. ts and the xs/h0 shapes are validated by the
//     cell's Step, with the same panics Unroll would produce. chunkSize
//     bounds the recompute unit size except across the forced merges
//     described above — and the worst case is honest: a loss visiting
//     outputs in an adversarial order (a run of consecutive
//     non-record-high seeded steps; a descending visit order with a small
//     chunkSize is the extreme) merges them into one O(len(xs))-long
//     unit, so peak graph memory is then O(len(xs) × per-step graph) —
//     the full-unroll figure — paid on top of the sweeps' recomputation.
//     In that corner UnrollRemat costs strictly more peak memory AND more
//     compute than one whole-graph backward; that is the price of bitwise
//     fidelity, not a bug.
//   - If the loss closes over a step-consumed leaf (a regularizer over a
//     cell parameter, an input or the initial state), every LOSS-SIDE
//     CONSUMER of that leaf must be visited by the loss's DFS after the
//     seeded outputs: the whole-graph backward then delivers the
//     loss-side contributions before the step-side ones, the order every
//     sweep replays. Spell a penalty data-first — Add(data, penalty),
//     never Add(penalty, data) — or make it a constant (a penalty over a
//     detached copy is loss-side only, so any position is legal). A
//     consumer structure that already lands after the outputs is legal
//     as spelled — a gate Hadamard(g, ys[last]) consumes the parameter
//     through a node visited after the seeded output, so both operand
//     orders pass. Folding the penalty into the seeded per-step terms is
//     NOT a remedy: those consumers are visited before the last output
//     and the drift returns. The probe checks the consumer positions in
//     the loss graph and panics on a violation (the two float32
//     associations would drift by a few ULP).
//
// # Requirement on the cell
//
// Step's graph structure must be a pure function of (x, h): the same ops
// in the same order at every step, independent of tensor VALUES. The
// classification probe runs one step and transfers its structure to the
// whole unroll; a value-dependent branch (consuming a parameter on the
// state descent on some steps and on the output branch on others) puts
// the leaf in different fold classes per step — a per-step variance the
// one-step probe cannot see, and the failure is a rounding-order drift
// on that leaf (measured at ~2 ULP), silent by construction.
//
// Within one step's structure, every trainable leaf SHARED ACROSS STEPS
// must have its consumers in exactly one fold class — one structural
// position per step. The provided LTC and CfC satisfy it (cm's consumers
// sit on the descent only, outW/outB's on the output branch only,
// everything else inside the state subgraph only), and the classification
// probe (step 2) decides each leaf's class from the topological position
// of its consumers. A shared leaf whose probe-step consumers span MORE
// THAN ONE class — say both on the state descent and on the output
// branch — is caught and panics at probe time: its gradient would fold
// in a loss-dependent interleaving no sweep decomposition reproduces, and
// the probe exposes every consumer position, so the failure is reported
// instead of drifting. Single-step leaves (xs, h0) are exempt from the
// multi-class check: each is consumed at exactly one step, there is no
// cross-step fold to corrupt, and the pick keeps the rest sweep's
// snapshot for them — an input tapped by both the state subgraph and the
// output branch (a skip connection) stays exact, verified against the
// whole-graph reference across loss shapes. The value-dependent per-step
// form above remains the residual case the probe cannot detect.
func UnrollRemat(cell Cell, params []*autograd.Variable, xs []*autograd.Variable, h0 *autograd.Variable, ts float64, chunkSize int, lossFn func(ys []*autograd.Variable) *autograd.Variable) (ys []*autograd.Variable, hN, loss *autograd.Variable) {
	if chunkSize < 1 {
		panic(fmt.Sprintf("nn.UnrollRemat: chunkSize must be >= 1, got %d", chunkSize))
	}
	n := len(xs)
	ys = make([]*autograd.Variable, n)
	if n == 0 {
		// Unroll's empty-sequence contract: no steps, h0 straight through.
		// The loss over an empty output slice still runs, and backpropagates
		// into whatever it referenced.
		loss = lossFn(ys)
		loss.Backward()
		return ys, h0, loss
	}

	// Pass 1: forward, detaching every step output and every step's input
	// state (states[k] feeds step k; states[0] is h0 itself, so its
	// gradient reaches the caller's leaf). The live graph is one step's.
	states := make([]*autograd.Variable, n)
	states[0] = h0
	h := h0
	for i := 0; i < n; i++ {
		var y *autograd.Variable
		y, h = cell.Step(xs[i], h, ts)
		ys[i] = autograd.Detach(y)
		if i+1 < n {
			h = autograd.Detach(h) // cut the graph at every step boundary
			states[i+1] = h
		}
	}
	hN = autograd.Detach(h)

	loss = lossFn(ys)

	// swept holds every leaf the sweeps accumulate into and the pick
	// reconciles; lossSnap preserves the loss-side contributions (the
	// whole backward delivers them first, so later sweeps must start from
	// them, not from zero).
	swept := make([]*autograd.Variable, 0, len(params)+len(xs)+1)
	swept = append(swept, params...)
	swept = append(swept, xs...)
	if h0 != nil {
		swept = append(swept, h0)
	}

	// The structural probe sorts every swept leaf into a fold class, and
	// the visit order π drives the unit cuts and the σ/affine sweeps. Two
	// hard preconditions ride on the same probe (see the doc comment):
	// every Step-consumed trainable leaf must be listed in params, and a
	// loss closing over a step-consumed leaf must visit every loss-side
	// consumer of it after the seeded outputs. Both checks read graph
	// structure only, so they run BEFORE the loss backward: a panic then
	// leaves no loss-side gradients in the leaves (a recover-and-retry
	// would double-count them).
	singleStep := make(map[*autograd.Variable]bool, len(xs)+1)
	for _, x := range xs {
		singleStep[x] = true
	}
	if h0 != nil {
		singleStep[h0] = true
	}
	classes, consumed := classifyFoldClasses(cell, xs[0], ts, swept, singleStep)
	assertParamsComplete(cell, swept, consumed)
	// The loss graph's topological order feeds both π and the loss-side
	// consumer-position check: walk it once. TopoOrder is a deterministic
	// pure read of the graph and nothing mutates the loss graph before its
	// backward below, so the two consumers observe exactly the order two
	// separate walks would have produced — one DFS walk and one escaped
	// result slice fewer per call (the visited set itself is pooled).
	lossTopo := autograd.TopoOrder(loss)
	pi := visitOrder(lossTopo, ys)
	assertLossSideOrder(lossTopo, ys, xs, h0, pi, swept, consumed)

	// The loss sees exactly the values it would see over a full Unroll, so
	// its backward computes the whole-graph backward's output seeds, bit
	// for bit. Steps the loss does not depend on keep a nil Grad — the
	// zero-seed case the sweeps skip, mirroring the whole-graph backward,
	// where such a step's output subgraph never enters the topological
	// order at all.
	loss.Backward()
	lossSnap := snapLeafGrads(swept)

	runs, isRH := recordHighRuns(pi)
	units := cutUnits(runs, pi, isRH, chunkSize)

	needSigma, needAffine := false, false
	{
		ascending := true
		for i := 1; i < len(pi); i++ {
			if pi[i] < pi[i-1] {
				ascending = false
				break
			}
		}
		for _, p := range swept {
			switch classes[p] {
			case classSpine:
				needSigma = true
			case classOutput:
				if !ascending {
					needAffine = true
				}
			}
		}
	}

	// unitBackward recomputes the steps of unit u and backpropagates the
	// saved seeds through them: output terms and the state term dh are
	// listed by the ordering rule that makes the unit backward replay the
	// whole-graph backward's restricted sequence — the descent-triggering
	// term first (the unit's record-high output term if it has one, else
	// the state term), the remaining output terms in π order, and the
	// state term last when a record-high term leads. It returns without
	// doing anything when no seed reaches the unit — the case in which
	// the whole-graph backward visits none of its nodes.
	unitBackward := func(u rematUnit, dh *tensor.Tensor) {
		if dh == nil {
			seeded := false
			for _, k := range u.seeds {
				if ys[k].Grad != nil {
					seeded = true
					break
				}
			}
			if !seeded {
				// Unreachable by construction: units cover only [0, max(π)],
				// dh is nil only for the topmost unit, and that unit holds
				// max(π) itself — a record-high seed, and every seed in π
				// has a non-nil Grad (it sits in the loss graph, whose
				// backward reaches all its leaves). The guard stays as a
				// nil-root firewall against future unit-cut changes.
				return
			}
		}
		h := states[u.s]
		ysRe := make([]*autograd.Variable, u.e-u.s)
		for i := u.s; i < u.e; i++ {
			ysRe[i-u.s], h = cell.Step(xs[i], h, ts)
		}
		// A term seeds v with the tensor g: SumAll(Hadamard(v, g)), whose
		// backward hands v the seed unchanged (1⊙g = g, bit for bit).
		var root *autograd.Variable
		inject := func(v *autograd.Variable, g *tensor.Tensor) {
			term := autograd.SumAll(autograd.Hadamard(v, autograd.Const(g)))
			if root == nil {
				root = term
			} else {
				root = autograd.Add(root, term)
			}
		}
		injectY := func(k int) {
			if ys[k].Grad != nil {
				inject(ysRe[k-u.s], ys[k].Grad)
			}
		}
		injectH := func() {
			if dh != nil {
				inject(h, dh) // h is the unit's output state after the loop
			}
		}
		if u.recordHigh >= 0 {
			// The record-high output term triggers the descent (first),
			// the other output terms follow in π order, the state term is
			// last — so the state's dh lands before the record-high
			// affine's contribution into the top state, and every
			// non-record-high affine lands before its step's rest.
			injectY(u.recordHigh)
			for _, k := range u.seeds {
				if k != u.recordHigh {
					injectY(k)
				}
			}
			injectH()
		} else {
			// No record-high output in this unit: the state term triggers
			// the descent, then the output terms in π order.
			injectH()
			for _, k := range u.seeds {
				injectY(k)
			}
		}
		root.Backward()
	}

	// Rest sweep: strictly reverse step order — the rest class's fold for
	// every loss shape. Threads the state gradient dh through the detached
	// state leaves and clones each unit's incoming dh aside for the σ
	// sweep.
	dhInto := make([]*tensor.Tensor, len(units))
	{
		var dh *tensor.Tensor
		for ui := len(units) - 1; ui >= 0; ui-- {
			u := units[ui]
			dhInto[ui] = cloneGrad(dh)
			unitBackward(u, dh)
			if u.s > 0 {
				dh = states[u.s].Grad
			}
		}
	}
	if !needSigma && !needAffine {
		return ys, hN, loss
	}
	revSnap := snapLeafGrads(swept)

	// σ sweep: the spine class's run-structured fold — runs in reverse
	// visit order, units ascending within each run — replaying the same
	// unit backwards from the loss-side contributions with the rest
	// sweep's boundary state gradients.
	var sigSnap map[*autograd.Variable]*tensor.Tensor
	if needSigma {
		restoreLeafGrads(swept, lossSnap)
		for ri := len(runs) - 1; ri >= 0; ri-- {
			for ui := range units {
				if units[ui].run == ri {
					unitBackward(units[ui], dhInto[ui])
				}
			}
		}
		sigSnap = snapLeafGrads(swept)
	}

	// Affine pass: the output class's reverse-visit fold, replayed from
	// the loss-side contributions.
	if needAffine {
		restoreLeafGrads(swept, lossSnap)
		affinePass(cell, xs, states, ys, ts, pi)
	}

	// The pick: each leaf keeps the sweep result matching its fold class.
	for _, p := range swept {
		switch classes[p] {
		case classSpine:
			p.Grad = sigSnap[p]
		case classOutput:
			if !needAffine {
				p.Grad = revSnap[p]
			} // else: keep the affine pass's accumulation
		default:
			p.Grad = revSnap[p]
		}
	}
	return ys, hN, loss
}

// rematUnit is one recompute unit: steps [s, e) recomputed together, with
// the seeded steps it contains (in π order), its record-high seed (-1
// when the unit holds no record-high output), and its run's index.
type rematUnit struct {
	s, e       int
	seeds      []int
	recordHigh int
	run        int
}

// rematRun is one record-high run of π: the contiguous step range [s, e)
// the descent from the run's record-high output covers.
type rematRun struct {
	s, e int
}

// recordHighRuns expands π into the record-high runs partitioning
// [0, max(π)]: each run ends at a record high of π and starts after the
// previous one. The runs come back in visit order, which is also
// ascending step order; isRH marks the record-high steps themselves.
func recordHighRuns(pi []int) (runs []rematRun, isRH map[int]bool) {
	isRH = make(map[int]bool, len(pi))
	start := 0
	high := -1
	for _, i := range pi {
		if i <= high {
			continue
		}
		runs = append(runs, rematRun{s: start, e: i + 1})
		isRH[i] = true
		start = i + 1
		high = i
	}
	return runs, isRH
}

// cutUnits splits each run into recompute units of at most chunkSize
// steps, never cutting between a non-record-high seeded step k and k+1:
// the affine contribution of a non-record-high step must fold into its
// state gradient BEFORE the later step's contributions — an association
// exact only when both steps share one recompute (see UnrollRemat). Each
// unit's seed list comes out in π order, and units carry their run index.
func cutUnits(runs []rematRun, pi []int, isRH map[int]bool, chunkSize int) []rematUnit {
	seeded := make(map[int]bool, len(pi))
	rank := make(map[int]int, len(pi))
	for pos, k := range pi {
		seeded[k] = true
		rank[k] = pos
	}
	var units []rematUnit
	for ri, run := range runs {
		for s := run.s; s < run.e; {
			e := s + chunkSize
			if e > run.e {
				e = run.e
			}
			// Extend across forced merges: a boundary between a
			// non-record-high seeded step e-1 and step e is forbidden.
			for e < run.e && seeded[e-1] && !isRH[e-1] {
				e++
			}
			u := rematUnit{s: s, e: e, recordHigh: -1, run: ri}
			for k := s; k < e; k++ {
				if seeded[k] {
					u.seeds = append(u.seeds, k)
				}
			}
			// Sort the seed list into π order (ascending rank).
			for i := 1; i < len(u.seeds); i++ {
				for j := i; j > 0 && rank[u.seeds[j-1]] > rank[u.seeds[j]]; j-- {
					u.seeds[j-1], u.seeds[j] = u.seeds[j], u.seeds[j-1]
				}
			}
			if isRH[e-1] {
				u.recordHigh = e - 1
			}
			units = append(units, u)
			s = e
		}
	}
	return units
}

// affinePass replays the output branch's gradient fold in reverse visit
// order: one y-seeded backward per seeded step. Only output-class leaves
// (the output affine's operands) are meant to keep its results; every
// other leaf it touches is restored by the caller's pick.
func affinePass(cell Cell, xs, states, ys []*autograd.Variable, ts float64, pi []int) {
	for j := len(pi) - 1; j >= 0; j-- {
		k := pi[j]
		y, _ := cell.Step(xs[k], states[k], ts)
		root := autograd.SumAll(autograd.Hadamard(y, autograd.Const(ys[k].Grad)))
		root.Backward()
	}
}

// visitOrder returns π, the order in which the loss's DFS first visits
// the seeded outputs (outputs the loss does not reach never appear), read
// off lossTopo, the loss graph's own topological order (computed once by
// the caller — see UnrollRemat).
func visitOrder(lossTopo []*autograd.Variable, ys []*autograd.Variable) []int {
	index := make(map[*autograd.Variable]int, len(ys))
	for i, y := range ys {
		index[y] = i
	}
	var pi []int
	seen := make(map[int]bool)
	for _, v := range lossTopo {
		if i, ok := index[v]; ok && !seen[i] {
			seen[i] = true
			pi = append(pi, i)
		}
	}
	return pi
}

// assertParamsComplete panics when a trainable leaf the Step consumes is
// missing from the caller's params list. The trainable set is read off
// the cell's Module interface (both provided cells implement it): any of
// its Parameters() with a consumer in the probe graph must be listed.
// An unlisted leaf is not restored between sweeps, so the σ/affine sweeps
// would accumulate a second copy of its step-side gradient on top of the
// rest sweep's — a silent 2-3x wrong value. A cell that does not expose
// Parameters() opts out of the audit (its params contract reverts to the
// caller; the probe cannot know its trainable set). And the audit is only
// as honest as the cell's Parameters(): a Module that omits a
// Step-consumed leaf from its own list defeats the trainable-set source,
// and that leaf is then multi-counted silently — the same opt-out,
// reached by omission.
func assertParamsComplete(cell Cell, swept []*autograd.Variable, consumed map[*autograd.Variable]bool) {
	m, ok := cell.(Module)
	if !ok {
		return
	}
	listed := make(map[*autograd.Variable]bool, len(swept))
	for _, p := range swept {
		listed[p] = true
	}
	for i, p := range m.Parameters() {
		if consumed[p] && !listed[p] {
			panic(fmt.Sprintf("nn.UnrollRemat: cell parameter #%d (shape %v) is consumed by Step but missing from the params list; list every trainable leaf (e.g. nn.ParametersOf(cell, readout)) — an unlisted leaf's gradient would silently accumulate once per sweep", i, p.Data.Shape))
		}
	}
}

// assertLossSideOrder panics when the loss closes over a step-consumed
// leaf (a regularizer over a cell parameter, an input, or the initial
// state) and some LOSS-SIDE CONSUMER of that leaf is visited by the
// loss's DFS before the highest-indexed seeded output. In the
// whole-graph backward such a consumer runs after the step-side
// contributions, while every sweep replays loss-side first — two
// float32 associations that drift apart.
//
// The consumer-position rule is exact under the skeleton-order argument:
// the detached loss graph's DFS visits the loss-side nodes in the same
// relative order as the whole graph's, and every step-side consumer has
// appended by the time the highest-indexed seeded output completes (its
// subtree covers every earlier step — the record-high structure of π).
// A consumer always appends after its whole subtree, so a leaf VISITED
// early can still be exact when every one of its consumers appends late
// (a gate Hadamard(g, ys[last])) — checking the leaf's own visit
// position instead rejects that provably-exact shape, a strict
// over-approximation this rule avoids.
//
// lossTopo is the loss graph's topological order, computed once by the
// caller (see UnrollRemat).
func assertLossSideOrder(lossTopo []*autograd.Variable, ys, xs []*autograd.Variable, h0 *autograd.Variable, pi []int, swept []*autograd.Variable, consumed map[*autograd.Variable]bool) {
	if len(pi) == 0 {
		return
	}
	pos := make(map[*autograd.Variable]int, len(lossTopo))
	for i, v := range lossTopo {
		pos[v] = i
	}
	lastY := pi[0]
	for _, k := range pi {
		if k > lastY {
			lastY = k
		}
	}
	posY := pos[ys[lastY]]
	xConsumed := consumed[xs[0]]
	stepSide := func(p *autograd.Variable) bool {
		if consumed[p] || p == h0 {
			return true
		}
		if xConsumed {
			for _, x := range xs {
				if p == x {
					return true
				}
			}
		}
		return false
	}
	for _, p := range swept {
		if !stepSide(p) {
			continue
		}
		for _, n := range lossTopo {
			if pos[n] >= posY {
				continue
			}
			for _, parent := range n.Parents() {
				if parent == p {
					panic(fmt.Sprintf("nn.UnrollRemat: the loss closes over a step-consumed leaf (shape %v) whose loss-side consumer is visited before the seeded outputs; the whole-graph backward would deliver that loss-side contribution after the step-side ones, an order no sweep replays — rewrite the loss so every consumer of the leaf is visited after the seeded outputs (for a penalty term: spell it data-first, Add(data, penalty), or make the penalty a constant)", p.Data.Shape))
				}
			}
		}
	}
}

// snapLeafGrads clones the current gradients of the given leaves into a
// map (nil Grad stays nil).
func snapLeafGrads(leaves []*autograd.Variable) map[*autograd.Variable]*tensor.Tensor {
	snap := make(map[*autograd.Variable]*tensor.Tensor, len(leaves))
	for _, p := range leaves {
		snap[p] = cloneGrad(p.Grad)
	}
	return snap
}

// restoreLeafGrads puts a snapLeafGrads snapshot back into the leaves.
func restoreLeafGrads(leaves []*autograd.Variable, snap map[*autograd.Variable]*tensor.Tensor) {
	for _, p := range leaves {
		p.Grad = snap[p]
	}
}

// cloneGrad clones a gradient tensor, preserving nil.
func cloneGrad(g *tensor.Tensor) *tensor.Tensor {
	if g == nil {
		return nil
	}
	return g.Clone()
}

// Fold classes: how a leaf's whole-graph backward contributions are
// ordered. The class decides which sweep's accumulation the leaf keeps.
const (
	// classStateRest leaves are consumed inside the per-step state
	// subgraph: the unwind appends those consumers bottom-up, so the
	// backward folds them in strictly reverse step order for every loss
	// shape. The rest sweep reproduces it.
	classStateRest = iota
	// classSpine leaves are reached by the DFS while descending the state
	// chain (the LTC's cm): their consumers append in the loss's
	// record-high visit runs, and the backward folds run by run in reverse
	// run order, each run ascending. The σ sweep reproduces it.
	classSpine
	// classOutput leaves are consumed on the step's output branch, above
	// the state (the output affine's operands outW/outB): their consumers
	// append as the loss visits the corresponding output, so the backward
	// folds them in the loss's reverse visit order over the seeded steps.
	// The affine pass reproduces it (when the visits ascend, this order
	// coincides with strictly reverse and the rest sweep suffices).
	classOutput
)

// classifyFoldClasses probes the cell's per-step graph once — one Step
// from a zero state, a synthetic scalar root shaped exactly like the
// sweeps' injection roots, and the DFS post-order of that root — and
// sorts every swept leaf into a fold class by the topological position of
// the leaf's CONSUMER nodes: a consumer appended before the step's
// input-state leaf (the descent bottom — its chain completed while the
// build pass was still descending, so across steps such consumers append
// top-down) makes the leaf spine-class; a consumer appended after the
// step's output state makes it output-class; anything between is
// state-rest. The classification is structural (values never enter it),
// so the one-step probe transfers to the whole unroll. Leaves with no
// consumer in the probe graph (loss-graph-only leaves, e.g. a readout's)
// map to the zero class: they receive no step-side contributions, and the
// rest sweep's snapshot already holds their final value.
//
// The probe also returns the set of leaves the Step consumes (one
// consumer or more), backing UnrollRemat's params-completeness check.
//
// A leaf SHARED ACROSS STEPS whose consumers sit in MORE THAN ONE fold
// class panics here: such a leaf's gradient folds in a loss-dependent
// interleaving no sweep decomposition reproduces (see UnrollRemat's
// requirement on the cell), and the one-step probe exposes every consumer
// position, so the failure is reported instead of drifting. Single-step
// leaves (the exempt set: xs and h0) are not checked — each is consumed
// at exactly one step, there is no cross-step fold to corrupt, and the
// pick keeps the rest sweep's snapshot for them, so an input tapped both
// by the state subgraph and the output branch (a skip connection) stays
// legal.
//
// # Why the probe result is deliberately NOT cached (stage-19a C9, vetoed)
//
// Caching this classification across UnrollRemat calls (keyed by cell and
// params) was considered and rejected: no sound cache key exists. The
// probe's output depends on everything Step's graph structure can depend
// on — the cell's internal parameter-variable identities and wiring, the
// per-call swept/exempt leaf sets, and ts — and UnrollRemat accepts ANY
// Cell implementation, whose Step may close over mutable state invisible
// to this package (a swapped wiring, a reconfigured unfold count, a
// ts-dependent branch). Detecting such a change would require deep
// identity snapshots of every value Step reads, which the Cell interface
// cannot provide; the provided cells expose no generation counter, and
// adding one is outside this change's scope. A stale hit would not just
// skip work: it would skip the multi-class panic above for a structure
// that changed after caching, and it would mis-sort leaves whose class
// moved — a rounding-order drift that is silent by construction. The
// probe also feeds assertParamsComplete's consumed set, a per-call
// completeness audit that must never consult a stale answer. Recomputing
// per call (one Step, one TopoOrder) is the price of a provably current
// classification; the loss-graph TopoOrder, in contrast, IS shared
// between visitOrder and assertLossSideOrder within one call, where
// determinism makes the two walks provably identical.
func classifyFoldClasses(cell Cell, x *autograd.Variable, ts float64, swept []*autograd.Variable, exempt map[*autograd.Variable]bool) (map[*autograd.Variable]int, map[*autograd.Variable]bool) {
	h := autograd.Var(tensor.New(x.Data.Rows(), cell.StateSize()))
	y, hNew := cell.Step(x, h, ts)
	inject := func(v *autograd.Variable) *autograd.Variable {
		return autograd.SumAll(autograd.Hadamard(v, autograd.Const(v.Data.OnesLike())))
	}
	root := autograd.Add(inject(y), inject(hNew))
	topo := autograd.TopoOrder(root)
	pos := make(map[*autograd.Variable]int, len(topo))
	for i, v := range topo {
		pos[v] = i
	}
	bottom, top := pos[h], pos[hNew]
	classOf := func(cpos int) int {
		switch {
		case cpos < bottom:
			return classSpine
		case cpos > top:
			return classOutput
		default:
			return classStateRest
		}
	}
	consumed := make(map[*autograd.Variable]bool)
	consumerClasses := make(map[*autograd.Variable]map[int]bool)
	for _, n := range topo {
		for _, parent := range n.Parents() {
			consumed[parent] = true
			cs := consumerClasses[parent]
			if cs == nil {
				cs = make(map[int]bool)
				consumerClasses[parent] = cs
			}
			cs[classOf(pos[n])] = true
		}
	}
	classes := make(map[*autograd.Variable]int, len(swept))
	for _, p := range swept {
		cs := consumerClasses[p]
		if len(cs) > 1 && !exempt[p] {
			panic(fmt.Sprintf("nn.UnrollRemat: shared leaf of shape %v is consumed in %d fold classes per step; every trainable leaf shared across steps needs its consumers in exactly one (see UnrollRemat)", p.Data.Shape, len(cs)))
		}
		// The earliest position at which anything in the step consumes p;
		// maxInt when nothing does.
		cpos := len(topo)
		for _, n := range topo {
			for _, parent := range n.Parents() {
				if parent == p {
					if pos[n] < cpos {
						cpos = pos[n]
					}
					break
				}
			}
		}
		if cpos == len(topo) {
			// No consumer in the probe step (h0, other xs elements,
			// loss-graph-only leaves): nothing step-side to re-fold, or a
			// single-step fold every sweep reproduces — the rest sweep's
			// snapshot already holds the final value.
			classes[p] = classStateRest
		} else {
			classes[p] = classOf(cpos)
		}
	}
	return classes, consumed
}
