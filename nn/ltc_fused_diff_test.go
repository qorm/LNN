package nn

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/tensor"
)

// Differential acceptance for the fused ODE-unfold kernel (nn/ltc_fused.go):
// the fused Step must reproduce the former graph path BIT FOR BIT — forward
// outputs and state, and every leaf gradient (Float32bits, never
// tolerance). legacyLTCStep rebuilds the pre-fusion graph exactly as Step
// built it before 16b (per-unfold synapse rows, fold contraction, Div
// membrane update); the matrix below sweeps dims, batch, wiring density,
// ts and seedings, single-step and unrolled.

// legacyLTCStep is the pre-fusion Step, kept verbatim as the oracle.
func legacyLTCStep(c *LTC, x, h *autograd.Variable, ts float64) (out, hNew *autograd.Variable) {
	batch := x.Data.Rows()
	if h == nil {
		h = autograd.Var(tensor.New(batch, c.units))
	}
	inputs := autograd.Add(autograd.Hadamard(x, c.inW), c.inB)
	cmT := c.scaledCapacitance(ts)
	gleak := autograd.Softplus(c.gleak)
	sWm := autograd.Hadamard(autograd.Softplus(c.sW), c.maskS)
	wM := autograd.Hadamard(autograd.Softplus(c.w), c.maskR)
	numS, denS := c.synapses(inputs, c.sMu, c.sSigma, sWm, c.erevRowsS)
	rows := func(m *autograd.Variable) []*autograd.Variable {
		rs := make([]*autograd.Variable, m.Data.Rows())
		for i := range rs {
			rs[i] = autograd.SliceRow(m, i)
		}
		return rs
	}
	muRs := rows(c.mu)
	sigRs := rows(c.sigma)
	wmRs := rows(wM)
	numConst := autograd.Add(autograd.Hadamard(gleak, c.vleak), numS)
	denBase := autograd.Add(autograd.Add(autograd.Add(cmT, gleak), denS), c.epsV)
	v := h
	for t := 0; t < c.unfolds; t++ {
		numR, denR := c.synapsesRows(v, muRs, sigRs, wmRs, c.erevRowsR)
		num := autograd.Add(autograd.Add(autograd.Hadamard(cmT, v), numConst), numR)
		v = autograd.Div(num, autograd.Add(denBase, denR))
	}
	out = autograd.Add(autograd.Hadamard(v, c.outW), c.outB)
	return out, v
}

// boundLegacy returns the legacy Step oracle bound to c, with Step's signature.
func boundLegacy(c *LTC) func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable) {
	return func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable) {
		return legacyLTCStep(c, x, h, ts)
	}
}

// diffBits compares two tensors element by element with Float32bits.
func fusedDiffBits(t *testing.T, name string, got, want *tensor.Tensor) {
	t.Helper()
	if got == nil || want == nil {
		t.Fatalf("%s: nil tensor (got %v, want %v)", name, got == nil, want == nil)
	}
	if !tensor.SameShape(got, want) {
		t.Fatalf("%s: shape drift: got %v, want %v", name, got.Shape, want.Shape)
	}
	for k := range got.Data {
		if math.Float32bits(got.Data[k]) != math.Float32bits(want.Data[k]) {
			t.Fatalf("%s: element %d: got %v (bits %#x), want %v (bits %#x)",
				name, k, got.Data[k], math.Float32bits(got.Data[k]),
				want.Data[k], math.Float32bits(want.Data[k]))
		}
	}
}

// ltcDiffCase is one fused-vs-legacy configuration.
type ltcDiffCase struct {
	name           string
	inDim, units   int
	unfolds, batch int
	wiring         float32 // sensory/recurrent density; < 0 means fully connected
	ts             float64
}

func (tc ltcDiffCase) build(t *testing.T) (*LTC, *autograd.Variable, *autograd.Variable) {
	t.Helper()
	rng := rand.New(rand.NewSource(500))
	var w *Wiring
	if tc.wiring >= 0 {
		w = RandomSparse(tc.inDim, tc.units, tc.wiring, tc.wiring, rng)
	}
	cell := NewLTC(tc.inDim, tc.units, w, tc.unfolds, rng)
	x := autograd.Var(tensor.Uniform(rng, -1, 1, tc.batch, tc.inDim))
	h := autograd.Var(tensor.Uniform(rng, -1, 1, tc.batch, tc.units))
	return cell, x, h
}

func ltcDiffCases() []ltcDiffCase {
	return []ltcDiffCase{
		{"base", 4, 16, 4, 8, -1, 0.1},
		{"base ts=1", 4, 16, 4, 8, -1, 1.0},
		{"tiny", 2, 3, 2, 1, -1, 0.1},
		{"unfolds=1", 3, 5, 1, 4, -1, 0.1},
		{"unfolds=8", 2, 4, 8, 3, -1, 1.0},
		{"units=1", 3, 1, 4, 5, -1, 0.1},
		{"inDim=1", 1, 6, 3, 4, -1, 0.1},
		{"inDim=1 units=1", 1, 1, 2, 3, -1, 1.0},
		{"sparse", 3, 5, 4, 4, 0.4, 0.1},
		{"sparse ts=1", 3, 5, 2, 2, 0.6, 1.0},
		{"zero masks", 2, 3, 4, 2, 0, 0.1},
		{"batch=1", 2, 4, 3, 1, -1, 0.1},
	}
}

// TestLTCFusedForwardBitExact pins the fused forward: out and hNew of
// Step must be Float32bits-identical to the legacy graph path over the
// whole matrix.
func TestLTCFusedForwardBitExact(t *testing.T) {
	for _, tc := range ltcDiffCases() {
		t.Run(tc.name, func(t *testing.T) {
			cell, x, h := tc.build(t)
			outF, hF := cell.Step(x, h, tc.ts)
			outL, hL := legacyLTCStep(cell, x, h, tc.ts)
			fusedDiffBits(t, "out", outF.Data, outL.Data)
			fusedDiffBits(t, "state", hF.Data, hL.Data)
		})
	}
}

// fusedLeafGrads snapshots the gradients of the cell's 13 parameters plus
// any extra leaves, keyed by a stable label, as Float32bits.
func fusedLeafGrads(cell *LTC, extra map[string]*autograd.Variable) map[string][]uint32 {
	out := make(map[string][]uint32)
	names := []string{"gleak", "vleak", "cm", "mu", "sigma", "w", "sMu", "sSigma", "sW", "inW", "inB", "outW", "outB"}
	for i, p := range cell.Parameters() {
		g := p.Grad
		if g == nil {
			out[names[i]] = nil
			continue
		}
		bits := make([]uint32, len(g.Data))
		for k, v := range g.Data {
			bits[k] = math.Float32bits(v)
		}
		out[names[i]] = bits
	}
	for name, v := range extra {
		g := v.Grad
		if g == nil {
			out[name] = nil
			continue
		}
		bits := make([]uint32, len(g.Data))
		for k, x := range g.Data {
			bits[k] = math.Float32bits(x)
		}
		out[name] = bits
	}
	return out
}

func fusedZeroAll(cell *LTC, extra map[string]*autograd.Variable) {
	for _, p := range cell.Parameters() {
		p.ZeroGrad()
	}
	for _, v := range extra {
		v.ZeroGrad()
	}
}

func fusedCmpGrads(t *testing.T, a, b map[string][]uint32) {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("gradient key count %d vs %d", len(a), len(b))
	}
	for name, ga := range a {
		gb, ok := b[name]
		if !ok {
			t.Fatalf("missing gradient %q", name)
		}
		if (ga == nil) != (gb == nil) {
			t.Fatalf("%s: nil-ness differs (%v vs %v)", name, ga == nil, gb == nil)
		}
		if len(ga) != len(gb) {
			t.Fatalf("%s: gradient length %d vs %d", name, len(ga), len(gb))
		}
		for k := range ga {
			if ga[k] != gb[k] {
				t.Fatalf("%s: gradient element %d: got bits %#x, want %#x", name, k, ga[k], gb[k])
			}
		}
	}
}

// TestLTCFusedBackwardSingleStepBitExact seeds both Step outputs (the
// output affine and the raw state) with pseudo-random gradients and pins
// every leaf gradient of the fused backward — all 13 parameters, x and h —
// to the legacy graph path, bit for bit. A zero-seed variant exercises the
// +/-0 corners (the single-source sign-bit corner F9-1 included).
func TestLTCFusedBackwardSingleStepBitExact(t *testing.T) {
	for _, tc := range ltcDiffCases() {
		t.Run(tc.name, func(t *testing.T) {
			cell, x, h := tc.build(t)
			extra := map[string]*autograd.Variable{"x": x, "h": h}
			rng := rand.New(rand.NewSource(777))
			// Draw the seeds ONCE: both runs must see identical gradients.
			soT := tensor.Uniform(rng, -1, 1, tc.batch, tc.units)
			svT := tensor.Uniform(rng, -1, 1, tc.batch, tc.units)
			build := func(step func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable)) *autograd.Variable {
				out, v := step(x, h, tc.ts)
				return autograd.Add(
					autograd.SumAll(autograd.Hadamard(out, autograd.Const(soT))),
					autograd.SumAll(autograd.Hadamard(v, autograd.Const(svT))))
			}
			fusedZeroAll(cell, extra)
			build(cell.Step).Backward()
			got := fusedLeafGrads(cell, extra)
			fusedZeroAll(cell, extra)
			build(boundLegacy(cell)).Backward()
			want := fusedLeafGrads(cell, extra)
			fusedCmpGrads(t, got, want)
		})
		t.Run(tc.name+" zero-seed", func(t *testing.T) {
			cell, x, h := tc.build(t)
			extra := map[string]*autograd.Variable{"x": x, "h": h}
			build := func(step func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable)) *autograd.Variable {
				out, v := step(x, h, tc.ts)
				so := autograd.Const(tensor.New(tc.batch, tc.units))
				sv := autograd.Const(tensor.New(tc.batch, tc.units))
				return autograd.Add(
					autograd.SumAll(autograd.Hadamard(out, so)),
					autograd.SumAll(autograd.Hadamard(v, sv)))
			}
			fusedZeroAll(cell, extra)
			build(cell.Step).Backward()
			got := fusedLeafGrads(cell, extra)
			fusedZeroAll(cell, extra)
			build(boundLegacy(cell)).Backward()
			want := fusedLeafGrads(cell, extra)
			fusedCmpGrads(t, got, want)
		})
	}
}

// ltcUnrollLoss builds the seeded scalar loss over a per-step output
// sequence and the final state, in five shapes: ascending, late-before-
// early, out-of-order, final-state-only and even-steps-only. The seeds
// interleave the output-affine and state contributions differently
// (record-high structure), which is exactly what the fused backward's
// state-boundary replay must preserve. Seed tensors are drawn from seeds
// in call order so that two runs over the same pre-drawn slice see
// identical gradients.
func ltcUnrollLoss(kind int, ys []*autograd.Variable, hN *autograd.Variable, seeds []*tensor.Tensor) *autograd.Variable {
	n := len(ys)
	next := 0
	seed := func() *autograd.Variable {
		s := autograd.Const(seeds[next])
		next++
		return s
	}
	term := func(v *autograd.Variable) *autograd.Variable {
		return autograd.SumAll(autograd.Hadamard(v, seed()))
	}
	switch kind {
	case 0: // ascending
		var acc *autograd.Variable
		for i, y := range ys {
			if i == 0 {
				acc = term(y)
			} else {
				acc = autograd.Add(acc, term(y))
			}
		}
		return autograd.Add(acc, term(hN))
	case 1: // late before early
		return autograd.Add(term(ys[n-1]), term(ys[0]))
	case 2: // out of order
		return autograd.Add(autograd.Add(term(ys[n/2]), term(ys[n/4])), term(ys[3*n/4]))
	case 3: // final state only
		return term(hN)
	default: // even steps only
		var acc *autograd.Variable
		for i := 0; i < n; i += 2 {
			if acc == nil {
				acc = term(ys[i])
			} else {
				acc = autograd.Add(acc, term(ys[i]))
			}
		}
		return acc
	}
}

// TestLTCFusedBackwardUnrollBitExact runs a 4-step unroll with the fused
// Step against the legacy Step under every loss shape and pins every leaf
// gradient (parameters, every xs element, h0) bit for bit.
func TestLTCFusedBackwardUnrollBitExact(t *testing.T) {
	const T = 4
	for _, tc := range ltcDiffCases() {
		for kind := 0; kind < 5; kind++ {
			t.Run(tc.name+fmt.Sprintf(" loss=%d", kind), func(t *testing.T) {
				rng := rand.New(rand.NewSource(900))
				var w *Wiring
				if tc.wiring >= 0 {
					w = RandomSparse(tc.inDim, tc.units, tc.wiring, tc.wiring, rng)
				}
				cell := NewLTC(tc.inDim, tc.units, w, tc.unfolds, rng)
				xs := make([]*autograd.Variable, T)
				for i := range xs {
					xs[i] = autograd.Var(tensor.Uniform(rng, -1, 1, tc.batch, tc.inDim))
				}
				h0 := autograd.Var(tensor.Uniform(rng, -1, 1, tc.batch, tc.units))
				extra := map[string]*autograd.Variable{"h0": h0}
				for i, x := range xs {
					extra[fmt.Sprintf("x%d", i)] = x
				}
				// Pre-draw every seed the loss can consume (at most T+1).
				seeds := make([]*tensor.Tensor, T+1)
				for i := range seeds {
					seeds[i] = tensor.Uniform(rng, -1, 1, tc.batch, tc.units)
				}
				run := func(step func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable)) *autograd.Variable {
					h := h0
					ys := make([]*autograd.Variable, T)
					for i, x := range xs {
						ys[i], h = step(x, h, tc.ts)
					}
					return ltcUnrollLoss(kind, ys, h, seeds)
				}
				fusedZeroAll(cell, extra)
				run(cell.Step).Backward()
				got := fusedLeafGrads(cell, extra)
				fusedZeroAll(cell, extra)
				run(boundLegacy(cell)).Backward()
				want := fusedLeafGrads(cell, extra)
				fusedCmpGrads(t, got, want)
			})
		}
	}
}

// TestLTCFusedBackwardIrregularSeed drives the fused backward with a
// manually seeded scalar Grad on the state node (rooted backward): the
// graph path broadcasts-then-reduces such seeds in its reduction
// fallbacks, and the fused backward's broadcast-by-ones must reproduce
// the same values bit for bit. A shape that cannot broadcast must panic
// in both paths (tensor's own broadcast panic).
func TestLTCFusedBackwardIrregularSeed(t *testing.T) {
	rng := rand.New(rand.NewSource(4242))
	cell := NewLTC(3, 4, RandomSparse(3, 4, 0.6, 0.6, rng), 3, rng)
	x := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 3))
	h := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 4))
	extra := map[string]*autograd.Variable{"x": x, "h": h}

	run := func(step func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable), seedBits uint32) {
		fusedZeroAll(cell, extra)
		_, v := step(x, h, 0.1)
		v.Grad = tensor.FromData([]float32{math.Float32frombits(seedBits)}, 1)
		v.Backward()
	}
	for _, seed := range []uint32{0x3f800000, 0xbf800000, 0x80000000, 0x00000000} {
		run(cell.Step, seed)
		got := fusedLeafGrads(cell, extra)
		run(boundLegacy(cell), seed)
		want := fusedLeafGrads(cell, extra)
		fusedCmpGrads(t, got, want)
	}

	// Non-broadcastable seed: both paths must panic (tensor's broadcast
	// panic, surfaced identically).
	panicOf := func(f func()) (msg string) {
		defer func() {
			if r := recover(); r != nil {
				msg = fmt.Sprint(r)
			}
		}()
		f()
		return ""
	}
	mF := panicOf(func() {
		fusedZeroAll(cell, extra)
		_, v := cell.Step(x, h, 0.1)
		v.Grad = tensor.New(3, 3)
		v.Backward()
	})
	mL := panicOf(func() {
		fusedZeroAll(cell, extra)
		_, v := boundLegacy(cell)(x, h, 0.1)
		v.Grad = tensor.New(3, 3)
		v.Backward()
	})
	if mF == "" || mL == "" {
		t.Fatalf("non-broadcastable seed: fused panic %q, legacy panic %q", mF, mL)
	}
}

// TestLTCFusedBackwardIrregularSeedExpansion covers the seed classes
// whose broadcast goes PAST [batch, units] before the graph path's per-op
// reductions pull them back: flat and row seeds over a units == 1 state
// (an outer product against the plane, then SumCols), wide and column
// seeds over a batch == 1 state (then SumRows), and the units==1&&batch==1
// scalar corners. The fused backward's terminal-Div replay must match the
// legacy graph path bit for bit on every leaf.
func TestLTCFusedBackwardIrregularSeedExpansion(t *testing.T) {
	nz := float32(math.Copysign(0, -1))
	cases := []struct {
		name                         string
		inDim, units, unfolds, batch int
		seed                         *tensor.Tensor
	}{
		{"u1 b3 flat[2]", 2, 1, 3, 3, tensor.FromData([]float32{0.5, nz}, 2)},
		{"u1 b3 [1,2]", 2, 1, 3, 3, tensor.FromData([]float32{-0.25, 0.75}, 1, 2)},
		{"u1 b3 [3,2]", 2, 1, 3, 3, tensor.FromData([]float32{0.5, -1, nz, 0.25, -0.5, 1}, 3, 2)},
		{"u1 b1 [1,3] uf1", 2, 1, 1, 1, tensor.FromData([]float32{0.5, nz, -0.25}, 1, 3)},
		{"u1 b1 [3,1] uf1", 2, 1, 1, 1, tensor.FromData([]float32{0.5, nz, -0.25}, 3, 1)},
		{"u1 b1 [1,3] uf3", 2, 1, 3, 1, tensor.FromData([]float32{0.5, nz, -0.25}, 1, 3)},
		{"u1 b1 [3,1] uf3", 2, 1, 3, 1, tensor.FromData([]float32{0.5, nz, -0.25}, 3, 1)},
		{"b1 u3 [2,3]", 2, 3, 3, 1, tensor.FromData([]float32{0.5, -1, nz, 0.25, -0.5, 1}, 2, 3)},
		{"b1 u3 [2,1]", 2, 3, 3, 1, tensor.FromData([]float32{0.5, nz}, 2, 1)},
		{"b2 u3 [1,3]", 2, 3, 3, 2, tensor.FromData([]float32{0.5, nz, -0.25}, 1, 3)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(4242))
			cell := NewLTC(tc.inDim, tc.units, nil, tc.unfolds, rng)
			x := autograd.Var(tensor.Uniform(rng, -1, 1, tc.batch, tc.inDim))
			h := autograd.Var(tensor.Uniform(rng, -1, 1, tc.batch, tc.units))
			extra := map[string]*autograd.Variable{"x": x, "h": h}
			run := func(step func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable)) {
				fusedZeroAll(cell, extra)
				_, v := step(x, h, 0.1)
				v.Grad = tc.seed.Clone()
				v.Backward()
			}
			run(cell.Step)
			got := fusedLeafGrads(cell, extra)
			run(boundLegacy(cell))
			want := fusedLeafGrads(cell, extra)
			fusedCmpGrads(t, got, want)
		})
	}

	// A flat multi-element seed over a [1, 1] state reduces to a 1D [1]
	// numerator gradient, which the graph path's num-fold MatMulTransB
	// rejects before any math: both paths must surface that identical
	// panic, at any unfolds count.
	for _, unfolds := range []int{1, 3} {
		t.Run(fmt.Sprintf("u1 b1 flat[3] uf%d panic", unfolds), func(t *testing.T) {
			rng := rand.New(rand.NewSource(4242))
			cell := NewLTC(2, 1, nil, unfolds, rng)
			x := autograd.Var(tensor.Uniform(rng, -1, 1, 1, 2))
			h := autograd.Var(tensor.Uniform(rng, -1, 1, 1, 1))
			panicOf := func(f func()) (msg string) {
				defer func() {
					if r := recover(); r != nil {
						msg = fmt.Sprint(r)
					}
				}()
				f()
				return ""
			}
			seed := func() *tensor.Tensor { return tensor.FromData([]float32{0.5, -1, 0.25}, 3) }
			mF := panicOf(func() {
				_, v := cell.Step(x, h, 0.1)
				v.Grad = seed()
				v.Backward()
			})
			mL := panicOf(func() {
				_, v := boundLegacy(cell)(x, h, 0.1)
				v.Grad = seed()
				v.Backward()
			})
			if mF == "" || mF != mL {
				t.Fatalf("fused panic %q, legacy panic %q: both must raise the identical MatMulTransB shape panic", mF, mL)
			}
		})
	}
}

// TestLTCFusedLeafGradShapes pins the delivered gradient SHAPES per leaf:
// the fused backward must hand every leaf exactly the shape the legacy
// graph path hands it — including the [1, units] lift gleak/cm carry
// (autograd's documented 1D-lift quirk through the Softplus chain, not
// the parameter's own [units] shape). Shapes are part of the backward
// contract; the bit matrices compare values only.
func TestLTCFusedLeafGradShapes(t *testing.T) {
	for _, tc := range ltcDiffCases() {
		t.Run(tc.name, func(t *testing.T) {
			cell, x, h := tc.build(t)
			extra := map[string]*autograd.Variable{"x": x, "h": h}
			rng := rand.New(rand.NewSource(778))
			soT := tensor.Uniform(rng, -1, 1, tc.batch, tc.units)
			svT := tensor.Uniform(rng, -1, 1, tc.batch, tc.units)
			build := func(step func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable)) *autograd.Variable {
				out, v := step(x, h, tc.ts)
				return autograd.Add(
					autograd.SumAll(autograd.Hadamard(out, autograd.Const(soT))),
					autograd.SumAll(autograd.Hadamard(v, autograd.Const(svT))))
			}
			shapes := func() map[string]string {
				m := make(map[string]string)
				names := []string{"gleak", "vleak", "cm", "mu", "sigma", "w", "sMu", "sSigma", "sW", "inW", "inB", "outW", "outB"}
				for i, p := range cell.Parameters() {
					if p.Grad == nil {
						m[names[i]] = "nil"
					} else {
						m[names[i]] = fmt.Sprint(p.Grad.Shape)
					}
				}
				for name, v := range extra {
					if v.Grad == nil {
						m[name] = "nil"
					} else {
						m[name] = fmt.Sprint(v.Grad.Shape)
					}
				}
				return m
			}
			fusedZeroAll(cell, extra)
			build(cell.Step).Backward()
			got := shapes()
			fusedZeroAll(cell, extra)
			build(boundLegacy(cell)).Backward()
			want := shapes()
			for name, g := range got {
				if g != want[name] {
					t.Fatalf("%s: grad shape %s, legacy %s", name, g, want[name])
				}
			}
		})
	}
}

// TestLTCFusedPenaltyLeafShapePanic pins the consequence of the 1D lift:
// an L2 penalty Hadamard(p, p) over a 1D parameter delivers a [units]
// contribution against the step-side [1, units] one, so the accumulation
// panics with the engine's shape-mismatch message — in BOTH paths, the
// graph path's own quirk replicated, not a fused-kernel divergence.
func TestLTCFusedPenaltyLeafShapePanic(t *testing.T) {
	rng := rand.New(rand.NewSource(5150))
	cell := NewLTC(2, 4, nil, 3, rng)
	x := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 2))
	h := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 4))
	extra := map[string]*autograd.Variable{"x": x, "h": h}
	panicOf := func(f func()) (msg string) {
		defer func() {
			if r := recover(); r != nil {
				msg = fmt.Sprint(r)
			}
		}()
		f()
		return ""
	}
	for _, pi := range []int{0, 2} { // gleak, cm: the 1D-lift leaves
		p := cell.Parameters()[pi]
		build := func(step func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable)) func() {
			return func() {
				fusedZeroAll(cell, extra)
				out, _ := step(x, h, 0.1)
				data := autograd.SumAll(out)
				pen := autograd.Scale(autograd.SumAll(autograd.Hadamard(p, p)), 0.01)
				autograd.Add(data, pen).Backward()
			}
		}
		mF := panicOf(build(cell.Step))
		mL := panicOf(build(boundLegacy(cell)))
		if mF == "" || mF != mL || !strings.Contains(mF, "gradient shape mismatch") {
			t.Fatalf("param %d: fused panic %q, legacy panic %q: both must raise the identical shape-mismatch panic", pi, mF, mL)
		}
	}
}

// TestLTCFusedSignedZeroRowAccumulators pins the signed-zero ownership
// corners of the per-presynaptic row accumulators: with a saturating
// sigma the sigmoid output hits exactly 1, so 1-s is +0 and the block
// gradients carry signed zeros. With batch == 1 the graph path's z and
// SigmoidHadamard reductions ADOPT their buffers (hadamardReduce's and
// negReduce's same-shape arms — no +0-seeded wash), and with units == 1
// the Sub backward's preCol side is a same-shape passthrough; the fused
// kernel must keep the identical -0 sign bits. The u1 b1 case asserts a
// -0 actually flows into mu/sigma/w, so the corner cannot pass vacuously.
func TestLTCFusedSignedZeroRowAccumulators(t *testing.T) {
	for _, tc := range []struct {
		name                         string
		inDim, units, unfolds, batch int
	}{
		{"u1 b1", 3, 1, 3, 1},
		{"u1 b1 uf1", 3, 1, 1, 1},
		{"u1 b2", 3, 1, 3, 2},
		{"u3 b1", 3, 3, 3, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(42))
			cell := NewLTC(tc.inDim, tc.units, nil, tc.unfolds, rng)
			for i := range cell.sigma.Data.Data {
				cell.sigma.Data.Data[i] = 1e37 // saturate: sigmoid hits exactly 1, 1-s == +0
			}
			x := autograd.Var(tensor.Uniform(rng, -1, 1, tc.batch, tc.inDim))
			h := autograd.Var(tensor.Uniform(rng, -1, 1, tc.batch, tc.units))
			so := tensor.Uniform(rng, -1, 1, tc.batch, tc.units)
			sv := tensor.Uniform(rng, -1, 1, tc.batch, tc.units)
			extra := map[string]*autograd.Variable{"x": x, "h": h}
			build := func(step func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable)) *autograd.Variable {
				out, v := step(x, h, 0.1)
				return autograd.Add(
					autograd.SumAll(autograd.Hadamard(out, autograd.Const(so))),
					autograd.SumAll(autograd.Hadamard(v, autograd.Const(sv))))
			}
			fusedZeroAll(cell, extra)
			build(cell.Step).Backward()
			got := fusedLeafGrads(cell, extra)
			fusedZeroAll(cell, extra)
			build(boundLegacy(cell)).Backward()
			want := fusedLeafGrads(cell, extra)
			fusedCmpGrads(t, got, want)
			if tc.name == "u1 b1" {
				negZero := false
				for _, name := range []string{"mu", "sigma", "w"} {
					for _, b := range got[name] {
						if b == 0x80000000 {
							negZero = true
						}
					}
				}
				if !negZero {
					t.Fatalf("saturating sigma put no -0 into mu/sigma/w grads: the sign-bit corner is not exercised")
				}
			}
		})
	}
}

// TestLTCFusedBackwardChainedBitExact drives the chained topology: a
// 2-step unroll where step 1's input is a graph function of step 0's
// OUTPUT, and a 3-step two-layer stack (layer 2's input is layer 1's
// output). The graph path then interleaves the x-chain's contribution to
// the intermediate state gradient BETWEEN the fused region's Col
// scatters and its dense cmT-product buffer; the hvN delivery node must
// reproduce that interleaving bit for bit (a single atomic VJP cannot).
func TestLTCFusedBackwardChainedBitExact(t *testing.T) {
	t.Run("t2 chain terminal loss", func(t *testing.T) {
		rng := rand.New(rand.NewSource(7))
		cell := NewLTC(3, 3, nil, 4, rng)
		x0 := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 3))
		x1 := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 3))
		h0 := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 3))
		c1 := tensor.Uniform(rng, -1, 1, 1, 3)
		c2 := tensor.Uniform(rng, -1, 1, 2, 3)
		extra := map[string]*autograd.Variable{"x0": x0, "x1": x1, "h0": h0}
		run := func(step func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable)) {
			fusedZeroAll(cell, extra)
			y0, h1 := step(x0, h0, 0.1)
			x1c := autograd.Add(autograd.Hadamard(y0, autograd.Const(c1)), autograd.Const(c2))
			_, h2 := step(x1c, h1, 0.1)
			autograd.SumAll(h2).Backward()
		}
		run(cell.Step)
		got := fusedLeafGrads(cell, extra)
		run(boundLegacy(cell))
		want := fusedLeafGrads(cell, extra)
		fusedCmpGrads(t, got, want)
	})

	t.Run("stacked layers seeded", func(t *testing.T) {
		rng := rand.New(rand.NewSource(11))
		l1 := NewLTC(4, 4, nil, 3, rng)
		l2 := NewLTC(4, 4, nil, 3, rng)
		xs := make([]*autograd.Variable, 3)
		for i := range xs {
			xs[i] = autograd.Var(tensor.Uniform(rng, -1, 1, 2, 4))
		}
		h1 := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 4))
		h2 := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 4))
		seeds := make([]*tensor.Tensor, 3)
		for i := range seeds {
			seeds[i] = tensor.Uniform(rng, -1, 1, 2, 4)
		}
		snap := func(tag string, cells ...*LTC) map[string][]uint32 {
			m := make(map[string][]uint32)
			names := []string{"gleak", "vleak", "cm", "mu", "sigma", "w", "sMu", "sSigma", "sW", "inW", "inB", "outW", "outB"}
			for ci, cell := range cells {
				for i, p := range cell.Parameters() {
					g := p.Grad
					if g == nil {
						m[fmt.Sprintf("%s%d.%s", tag, ci, names[i])] = nil
						continue
					}
					bits := make([]uint32, len(g.Data))
					for k, v := range g.Data {
						bits[k] = math.Float32bits(v)
					}
					m[fmt.Sprintf("%s%d.%s", tag, ci, names[i])] = bits
				}
			}
			return m
		}
		extra := map[string]*autograd.Variable{"h1": h1, "h2": h2, "x0": xs[0], "x1": xs[1], "x2": xs[2]}
		zeroAll := func() {
			for _, cell := range []*LTC{l1, l2} {
				for _, p := range cell.Parameters() {
					p.ZeroGrad()
				}
			}
			for _, v := range extra {
				v.ZeroGrad()
			}
		}
		run := func(stepOf func(*LTC) func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable)) map[string][]uint32 {
			zeroAll()
			var acc *autograd.Variable
			a, b := h1, h2
			for i := 0; i < 3; i++ {
				var y1, y2 *autograd.Variable
				y1, a = stepOf(l1)(xs[i], a, 0.1)
				y2, b = stepOf(l2)(y1, b, 0.1)
				term := autograd.Add(
					autograd.SumAll(autograd.Hadamard(y1, autograd.Const(seeds[i]))),
					autograd.SumAll(autograd.Hadamard(y2, autograd.Const(seeds[i]))))
				if acc == nil {
					acc = term
				} else {
					acc = autograd.Add(acc, term)
				}
			}
			acc.Backward()
			m := snap("c", l1, l2)
			for name, v := range extra {
				g := v.Grad
				if g == nil {
					m[name] = nil
					continue
				}
				bits := make([]uint32, len(g.Data))
				for k, x := range g.Data {
					bits[k] = math.Float32bits(x)
				}
				m[name] = bits
			}
			return m
		}
		got := run(func(c *LTC) func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable) { return c.Step })
		want := run(boundLegacy)
		fusedCmpGrads(t, got, want)
	})
}

// TestLTCFusedAdversarialNonFinite drives the fused backward through the
// non-finite operand path of its mul32 replicas: seeds carrying +/-Inf and
// NaN must propagate bit-identically to the legacy graph path (the mul32
// native-multiply arm exists exactly for payload-faithful NaN
// propagation). Also seeds -0 (the sign-bit corner) and covers ts in the
// clamped tiny-step regime.
func TestLTCFusedAdversarialNonFinite(t *testing.T) {
	rng := rand.New(rand.NewSource(31337))
	cell := NewLTC(2, 3, RandomSparse(2, 3, 0.7, 0.7, rng), 2, rng)
	x := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 2))
	h := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 3))
	extra := map[string]*autograd.Variable{"x": x, "h": h}

	seeds := [][]float32{
		{float32(math.Inf(1)), 1, -1, 0.5, float32(math.Inf(-1)), 2},
		{float32(math.NaN()), 1, -1, 0.5, 2, -2},
		{float32(math.Copysign(0, -1)), 1, -1, 0.5, 2, -2},
	}
	for si, sv := range seeds {
		run := func(step func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable)) {
			fusedZeroAll(cell, extra)
			_, v := step(x, h, 0.1)
			v.Grad = tensor.FromData(append([]float32(nil), sv...), 2, 3)
			v.Backward()
		}
		run(cell.Step)
		got := fusedLeafGrads(cell, extra)
		run(boundLegacy(cell))
		want := fusedLeafGrads(cell, extra)
		fusedCmpGrads(t, got, want)
		_ = si
	}

	// Tiny-ts clamped regime: forward and backward stay bit-identical.
	for _, ts := range []float64{1e-40, 1e300} {
		run := func(step func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable)) {
			fusedZeroAll(cell, extra)
			out, v := step(x, h, ts)
			autograd.Add(autograd.SumAll(out), autograd.SumAll(v)).Backward()
		}
		run(cell.Step)
		got := fusedLeafGrads(cell, extra)
		run(boundLegacy(cell))
		want := fusedLeafGrads(cell, extra)
		fusedCmpGrads(t, got, want)
	}
}

// TestLTCFusedStateShapePanic pins the fused kernel's state validation: a
// wrongly shaped h panics before any indexing (the graph path panicked in
// the tensor layer).
func TestLTCFusedStateShapePanic(t *testing.T) {
	rng := rand.New(rand.NewSource(61))
	cell := NewLTC(2, 3, nil, 2, rng)
	x := autograd.Var(tensor.New(2, 2))
	for _, bad := range []*tensor.Tensor{
		tensor.New(2, 4),    // wrong width
		tensor.New(3, 3),    // wrong rows
		tensor.New(6),       // rank 1
		tensor.New(2, 3, 1), // rank 3
	} {
		func() {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("Step with h shape %v did not panic", bad.Shape)
				}
				if msg := fmt.Sprint(r); !strings.Contains(msg, "state shape") {
					t.Fatalf("panic %q should name the state shape", msg)
				}
			}()
			cell.Step(x, autograd.Var(bad), 0.1)
		}()
	}
}

// TestLTCFusedGradShapeMismatchPanic pre-seeds the state leaf's Grad with
// an incompatible shape: delivering the replayed scatter buffers must
// panic with the engine's shape-mismatch message, exactly as the legacy
// Col backward's addGrad does.
func TestLTCFusedGradShapeMismatchPanic(t *testing.T) {
	rng := rand.New(rand.NewSource(67))
	cell := NewLTC(2, 3, nil, 2, rng)
	x := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 2))
	panicOf := func(f func()) (msg string) {
		defer func() {
			if r := recover(); r != nil {
				msg = fmt.Sprint(r)
			}
		}()
		f()
		return ""
	}
	mF := panicOf(func() {
		h := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 3))
		out, _ := cell.Step(x, h, 0.1)
		h.Grad = tensor.New(3, 3)
		autograd.SumAll(out).Backward()
	})
	mL := panicOf(func() {
		h := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 3))
		out, _ := boundLegacy(cell)(x, h, 0.1)
		h.Grad = tensor.New(3, 3)
		autograd.SumAll(out).Backward()
	})
	if !strings.Contains(mF, "gradient shape mismatch") || !strings.Contains(mL, "gradient shape mismatch") {
		t.Fatalf("fused %q, legacy %q: both must raise the shape-mismatch panic", mF, mL)
	}
}

// TestLTCFusedAdversarialFoldOverflow loads absurd recurrent weights so
// the activation blocks overflow the fold to +Inf: the identity MatMul's
// av*0 terms then spread NaN across the row (Inf*0), in BOTH the legacy
// graph path and the fused kernel's literal MatMul calls. Any per-element
// "wash" shortcut would keep the Inf local and diverge here; forward and
// backward must stay Float32bits-identical.
func TestLTCFusedAdversarialFoldOverflow(t *testing.T) {
	rng := rand.New(rand.NewSource(999))
	cell := NewLTC(2, 3, nil, 2, rng)
	for i := range cell.w.Data.Data {
		cell.w.Data.Data[i] = 2e38 // softplus(x) = x for x > 20; two terms overflow the fold
	}
	x := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 2))
	h := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 3))
	extra := map[string]*autograd.Variable{"x": x, "h": h}

	outF, hF := cell.Step(x, h, 0.1)
	outL, hL := legacyLTCStep(cell, x, h, 0.1)
	fusedDiffBits(t, "overflow out", outF.Data, outL.Data)
	fusedDiffBits(t, "overflow state", hF.Data, hL.Data)
	anyNaN := false
	for _, val := range hF.Data.Data {
		if math.IsNaN(float64(val)) {
			anyNaN = true
		}
	}
	if !anyNaN {
		t.Fatalf("expected the overflowed fold to NaN the state, got %v", hF.Data.Data)
	}

	run := func(step func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable)) {
		fusedZeroAll(cell, extra)
		out, v := step(x, h, 0.1)
		autograd.Add(autograd.SumAll(out), autograd.SumAll(v)).Backward()
	}
	run(cell.Step)
	got := fusedLeafGrads(cell, extra)
	run(boundLegacy(cell))
	want := fusedLeafGrads(cell, extra)
	fusedCmpGrads(t, got, want)
}
