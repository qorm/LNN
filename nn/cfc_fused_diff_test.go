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

// Differential acceptance for the fused closed-form CfC step kernel
// (nn/cfc_fused.go): the fused Step must reproduce the former graph path
// BIT FOR BIT — forward outputs and state, and every leaf gradient
// (Float32bits, never tolerance; the double-NaN payload corner documented
// in nn/ltc_fused.go is exempted only in the adversarial non-finite
// tests). legacyCfCStep rebuilds the pre-fusion graph exactly as Step
// built it before 18b (the drive/contract/decayRate/decayFactor methods
// are the same white-box machinery the old Step called). The matrix below
// sweeps dims, batch, wiring density, ts (including the exprel branch
// region), seedings, single-step and unrolled, plus the irregular-seed
// arms of the terminal Add, the adversarial fold overflow, the
// signed-zero corners, chained and stacked topologies, and the mid-flight
// mutation freeze discipline.

// legacyCfCStep is the pre-fusion Step, kept verbatim as the oracle.
func legacyCfCStep(c *CfC, x, h *autograd.Variable, ts float64) (out, hNew *autograd.Variable) {
	if !(ts > 0) || math.IsInf(ts, 0) {
		panic(fmt.Sprintf("nn.CfC.Step: ts must be positive and finite, got %v", ts))
	}
	batch := x.Data.Rows()
	if h == nil {
		h = autograd.Var(tensor.New(batch, c.units))
	}

	// Affine input mapping.
	inputs := autograd.Add(autograd.Hadamard(x, c.inW), c.inB)

	// Positivity-constrained parameters (softplus).
	gleak := autograd.Softplus(c.gleak)
	cm := autograd.Softplus(c.cm)
	wPos := autograd.Softplus(c.w)
	sWPos := autograd.Softplus(c.sW)

	// Synaptic drives: num = sum_j act_j*erev_j, den = sum_j act_j, with the
	// erev signs carried by the erevRows constants (see drive and contract).
	numS, denS := c.drive(inputs, c.sMu, c.sSigma, sWPos, c.erevRowsS, c.wiring.SensoryRow)
	numR, denR := c.drive(h, c.mu, c.sigma, wPos, c.erevRowsR, c.wiring.RecurrentRow)

	epsV := autograd.Const(tensor.FromData([]float32{c.eps}, 1))
	// G = gleak + den, the total conductance [batch, units].
	g := autograd.Add(gleak, autograd.Add(denS, denR))
	// A = (gleak*vleak + num) / (G + eps): the instantaneous reversal state.
	a := autograd.Div(
		autograd.Add(autograd.Hadamard(gleak, c.vleak), autograd.Add(numS, numR)),
		autograd.Add(g, epsV),
	)
	// B = kappa*ts = G/cm * ts, with the overflow/sign cap of decayRate.
	b := c.decayRate(g, cm, epsV, ts)
	// F = 1 - exp(-B), exprel-stabilized (see decayFactor).
	f := c.decayFactor(b)

	// v_new = v + (A - v)*F, the closed-form solution over the span ts.
	vNew := autograd.Add(h, autograd.Hadamard(autograd.Sub(a, h), f))

	out = autograd.Add(autograd.Hadamard(vNew, c.outW), c.outB)
	return out, vNew
}

// boundLegacyCfC returns the legacy Step oracle bound to c, with Step's signature.
func boundLegacyCfC(c *CfC) func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable) {
	return func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable) {
		return legacyCfCStep(c, x, h, ts)
	}
}

// cfcDiffCase is one fused-vs-legacy configuration.
type cfcDiffCase struct {
	name         string
	inDim, units int
	batch        int
	wiring       float32 // sensory/recurrent density; < 0 means fully connected
	ts           float64
}

func (tc cfcDiffCase) build(t *testing.T) (*CfC, *autograd.Variable, *autograd.Variable) {
	t.Helper()
	rng := rand.New(rand.NewSource(500))
	var w *Wiring
	if tc.wiring >= 0 {
		w = RandomSparse(tc.inDim, tc.units, tc.wiring, tc.wiring, rng)
	}
	cell := NewCfC(tc.inDim, tc.units, w, rng)
	x := autograd.Var(tensor.Uniform(rng, -1, 1, tc.batch, tc.inDim))
	h := autograd.Var(tensor.Uniform(rng, -1, 1, tc.batch, tc.units))
	return cell, x, h
}

func cfcDiffCases() []cfcDiffCase {
	return []cfcDiffCase{
		{"base", 4, 16, 8, -1, 0.1},
		{"base ts=1", 4, 16, 8, -1, 1.0},
		{"tiny", 2, 3, 1, -1, 0.1},
		{"units=1", 3, 1, 5, -1, 0.1},
		{"inDim=1", 1, 6, 4, -1, 0.1},
		{"inDim=1 units=1", 1, 1, 3, -1, 1.0},
		{"sparse", 3, 5, 4, 0.4, 0.1},
		{"sparse ts=1", 3, 5, 2, 0.6, 1.0},
		{"zero masks", 2, 3, 2, 0, 0.1},
		{"batch=1", 2, 4, 1, -1, 0.1},
		{"exprel low ts", 3, 4, 3, -1, 1e-3},
		{"exprel mid ts", 2, 5, 2, 0.7, 3e-2},
		{"clamped tiny ts", 2, 3, 2, -1, 1e-40},
		{"clamped huge ts", 2, 3, 2, -1, 1e300},
	}
}

// TestCfCFusedForwardBitExact pins the fused forward: out and hNew of
// Step must be Float32bits-identical to the legacy graph path over the
// whole matrix.
func TestCfCFusedForwardBitExact(t *testing.T) {
	for _, tc := range cfcDiffCases() {
		t.Run(tc.name, func(t *testing.T) {
			cell, x, h := tc.build(t)
			outF, hF := cell.Step(x, h, tc.ts)
			outL, hL := legacyCfCStep(cell, x, h, tc.ts)
			fusedDiffBits(t, "out", outF.Data, outL.Data)
			fusedDiffBits(t, "state", hF.Data, hL.Data)
		})
	}
}

// cfcLeafGrads snapshots the gradients of the cell's 13 parameters plus
// any extra leaves, keyed by a stable label, as Float32bits.
func cfcLeafGrads(cell *CfC, extra map[string]*autograd.Variable) map[string][]uint32 {
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

func cfcZeroAll(cell *CfC, extra map[string]*autograd.Variable) {
	for _, p := range cell.Parameters() {
		p.ZeroGrad()
	}
	for _, v := range extra {
		v.ZeroGrad()
	}
}

func cfcCmpGrads(t *testing.T, a, b map[string][]uint32) {
	t.Helper()
	cfcCmpGradsNaN(t, a, b, false)
}

// cfcCmpGradsNaN compares two gradient snapshots bit for bit; with
// nanExempt a position where both sides are NaN (any payload/sign)
// passes — the documented double-NaN corner — while the NaN position set
// and every finite value must still agree exactly.
func cfcCmpGradsNaN(t *testing.T, a, b map[string][]uint32, nanExempt bool) {
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
			if ga[k] == gb[k] {
				continue
			}
			if nanExempt && fuzzIsNaNBits(ga[k]) && fuzzIsNaNBits(gb[k]) {
				continue
			}
			t.Fatalf("%s: gradient element %d: got bits %#x, want %#x", name, k, ga[k], gb[k])
		}
	}
}

// TestCfCFusedBackwardSingleStepBitExact seeds both Step outputs (the
// output affine and the raw state) with pseudo-random gradients and pins
// every leaf gradient of the fused backward — all 13 parameters, x and h —
// to the legacy graph path, bit for bit. A zero-seed variant exercises the
// +/-0 corners (the single-source sign-bit corner F9-1 included).
func TestCfCFusedBackwardSingleStepBitExact(t *testing.T) {
	for _, tc := range cfcDiffCases() {
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
			cfcZeroAll(cell, extra)
			build(cell.Step).Backward()
			got := cfcLeafGrads(cell, extra)
			cfcZeroAll(cell, extra)
			build(boundLegacyCfC(cell)).Backward()
			want := cfcLeafGrads(cell, extra)
			cfcCmpGrads(t, got, want)
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
			cfcZeroAll(cell, extra)
			build(cell.Step).Backward()
			got := cfcLeafGrads(cell, extra)
			cfcZeroAll(cell, extra)
			build(boundLegacyCfC(cell)).Backward()
			want := cfcLeafGrads(cell, extra)
			cfcCmpGrads(t, got, want)
		})
	}
}

// TestCfCFusedBackwardUnrollBitExact runs a 4-step unroll with the fused
// Step against the legacy Step under every loss shape and pins every leaf
// gradient (parameters, every xs element, h0) bit for bit.
func TestCfCFusedBackwardUnrollBitExact(t *testing.T) {
	const T = 4
	for _, tc := range cfcDiffCases() {
		for kind := 0; kind < 5; kind++ {
			t.Run(tc.name+fmt.Sprintf(" loss=%d", kind), func(t *testing.T) {
				rng := rand.New(rand.NewSource(900))
				var w *Wiring
				if tc.wiring >= 0 {
					w = RandomSparse(tc.inDim, tc.units, tc.wiring, tc.wiring, rng)
				}
				cell := NewCfC(tc.inDim, tc.units, w, rng)
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
				cfcZeroAll(cell, extra)
				run(cell.Step).Backward()
				got := cfcLeafGrads(cell, extra)
				cfcZeroAll(cell, extra)
				run(boundLegacyCfC(cell)).Backward()
				want := cfcLeafGrads(cell, extra)
				cfcCmpGrads(t, got, want)
			})
		}
	}
}

// TestCfCFusedBackwardIrregularSeedPanics pins the terminal-Add reduction
// arms and both panic texts for irregular manually seeded state gradients
// (see nn/cfc_fused.go's irregular-seed section): the general shapes
// raise the SumToShape "cannot reduce" panic in both paths; the
// batch == units == 1 scalar arm delivers the [1] SumAll to h and then
// raises the engine's shape-mismatch panic in both paths.
func TestCfCFusedBackwardIrregularSeedPanics(t *testing.T) {
	panicOf := func(f func()) (msg string) {
		defer func() {
			if r := recover(); r != nil {
				msg = fmt.Sprint(r)
			}
		}()
		f()
		return ""
	}

	// General case: any non-exact shape panics with the identical
	// cannot-reduce message in both paths (the terminal node is an Add —
	// there is no broadcast path).
	rng := rand.New(rand.NewSource(4242))
	cell := NewCfC(2, 3, nil, rng)
	x := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 2))
	for _, sh := range [][]int{{1}, {1, 1}, {1, 3}, {2, 1}, {1, 6}, {6}, {3}, {1, 2}, {2, 2}, {2, 3, 1}} {
		h := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 3))
		seed := tensor.New(sh...)
		mF := panicOf(func() {
			_, v := cell.Step(x, h, 0.1)
			v.Grad = seed.Clone()
			v.Backward()
		})
		mL := panicOf(func() {
			_, v := legacyCfCStep(cell, x, h, 0.1)
			v.Grad = seed.Clone()
			v.Backward()
		})
		want := fmt.Sprintf("tensor.SumToShape: cannot reduce shape %v to %v", sh, []int{2, 3})
		if mF != want || mL != want {
			t.Fatalf("seed %v: fused panic %q, legacy panic %q, want %q", sh, mF, mL, want)
		}
	}

	// The scalar corner (batch == units == 1): every non-exact seed shape
	// takes the SumAll arm — the [1] delivery lands on h, then the Sub
	// side's [1, 1] delivery panics with the engine's mismatch message.
	cell1 := NewCfC(2, 1, nil, rand.New(rand.NewSource(43)))
	x1 := autograd.Var(tensor.Uniform(rand.New(rand.NewSource(44)), -1, 1, 1, 2))
	// A pre-seeded h.Grad of an incompatible shape panics one delivery
	// EARLIER — at the scalar arm's own [1] delivery — with the pre-seed's
	// shape in the message, identically in both paths.
	for _, pre := range [][]int{{3}, {1, 1}} {
		mF := panicOf(func() {
			h := autograd.Var(tensor.New(1, 1))
			h.Grad = tensor.New(pre...)
			_, v := cell1.Step(x1, h, 0.1)
			v.Grad = tensor.New(5)
			v.Backward()
		})
		mL := panicOf(func() {
			h := autograd.Var(tensor.New(1, 1))
			h.Grad = tensor.New(pre...)
			_, v := legacyCfCStep(cell1, x1, h, 0.1)
			v.Grad = tensor.New(5)
			v.Backward()
		})
		want := fmt.Sprintf("autograd: gradient shape mismatch: accumulated %v vs incoming [1]", pre)
		if mF != want || mL != want {
			t.Fatalf("u1b1 pre-seed %v: fused panic %q, legacy panic %q, want %q", pre, mF, mL, want)
		}
	}
	for _, sh := range [][]int{{1}, {3}, {1, 3}, {3, 1}, {2, 2}, {3, 3}} {
		run := func(step func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable)) (string, []uint32) {
			h := autograd.Var(tensor.New(1, 1))
			defer func() { _ = recover() }()
			var msg string
			func() {
				defer func() {
					if r := recover(); r != nil {
						msg = fmt.Sprint(r)
					}
				}()
				_, v := step(x1, h, 0.1)
				v.Grad = tensor.New(sh...)
				v.Backward()
			}()
			var hBits []uint32
			if h.Grad != nil {
				if fmt.Sprint(h.Grad.Shape) != "[1]" {
					t.Fatalf("seed %v: h.Grad shape %v, want [1] after the scalar-arm delivery", sh, h.Grad.Shape)
				}
				hBits = []uint32{math.Float32bits(h.Grad.Data[0])}
			}
			return msg, hBits
		}
		mF, hF := run(cell1.Step)
		mL, hL := run(boundLegacyCfC(cell1))
		if mF == "" || mF != mL || !strings.Contains(mF, "gradient shape mismatch") {
			t.Fatalf("u1b1 seed %v: fused panic %q, legacy panic %q: both must raise the identical mismatch panic", sh, mF, mL)
		}
		if (hF == nil) != (hL == nil) {
			t.Fatalf("u1b1 seed %v: h.Grad nil-ness differs (%v vs %v)", sh, hF == nil, hL == nil)
		}
		if hF != nil && hF[0] != hL[0] {
			t.Fatalf("u1b1 seed %v: delivered [1] gradient bits %#x vs %#x", sh, hF[0], hL[0])
		}
	}
}

// TestCfCFusedBackwardIrregularSeedReductions covers the two reduction
// arms that let an irregular seed FLOW through with defined values:
// batch == 1 with a [k, units] seed (SumRows) and units == 1 with a
// [batch, k] seed (SumCols). After the terminal Add's arms, every
// downstream gradient is regular-shaped, so the fused kernel's regular
// replay must match the legacy graph path bit for bit on every leaf.
func TestCfCFusedBackwardIrregularSeedReductions(t *testing.T) {
	nz := float32(math.Copysign(0, -1))
	cases := []struct {
		name                string
		inDim, units, batch int
		seed                *tensor.Tensor
	}{
		{"b1 u3 [2,3]", 2, 3, 1, tensor.FromData([]float32{0.5, -1, nz, 0.25, -0.5, 1}, 2, 3)},
		{"b1 u3 [4,3]", 2, 3, 1, tensor.FromData([]float32{0.5, -1, nz, 0.25, -0.5, 1, 0.75, -0.25, 1.5, -1.25, 0.125, -0.75}, 4, 3)},
		{"u1 b3 [3,2]", 2, 1, 3, tensor.FromData([]float32{0.5, -1, nz, 0.25, -0.5, 1}, 3, 2)},
		{"u1 b2 [2,4]", 3, 1, 2, tensor.FromData([]float32{0.5, -1, nz, 0.25, -0.5, 1, 0.75, -1.5}, 2, 4)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(5150))
			cell := NewCfC(tc.inDim, tc.units, nil, rng)
			x := autograd.Var(tensor.Uniform(rng, -1, 1, tc.batch, tc.inDim))
			h := autograd.Var(tensor.Uniform(rng, -1, 1, tc.batch, tc.units))
			extra := map[string]*autograd.Variable{"x": x, "h": h}
			run := func(step func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable)) {
				cfcZeroAll(cell, extra)
				_, v := step(x, h, 0.1)
				v.Grad = tc.seed.Clone()
				v.Backward()
			}
			run(cell.Step)
			got := cfcLeafGrads(cell, extra)
			run(boundLegacyCfC(cell))
			want := cfcLeafGrads(cell, extra)
			cfcCmpGrads(t, got, want)
		})
	}
}

// TestCfCFusedAdversarialNonFinite drives the fused backward through the
// non-finite operand path of its mul32 replicas: seeds carrying +/-Inf and
// NaN must propagate bit-identically to the legacy graph path (the mul32
// native-multiply arm exists exactly for payload-faithful NaN
// propagation), modulo the documented double-NaN payload corner. Also
// seeds -0 (the sign-bit corner) and covers ts in the clamped regimes.
func TestCfCFusedAdversarialNonFinite(t *testing.T) {
	rng := rand.New(rand.NewSource(31337))
	cell := NewCfC(2, 3, RandomSparse(2, 3, 0.7, 0.7, rng), rng)
	x := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 2))
	h := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 3))
	extra := map[string]*autograd.Variable{"x": x, "h": h}

	seeds := [][]float32{
		{float32(math.Inf(1)), 1, -1, 0.5, float32(math.Inf(-1)), 2},
		{float32(math.NaN()), 1, -1, 0.5, 2, -2},
		{float32(math.Copysign(0, -1)), 1, -1, 0.5, 2, -2},
	}
	for _, sv := range seeds {
		run := func(step func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable)) {
			cfcZeroAll(cell, extra)
			_, v := step(x, h, 0.1)
			v.Grad = tensor.FromData(append([]float32(nil), sv...), 2, 3)
			v.Backward()
		}
		run(cell.Step)
		got := cfcLeafGrads(cell, extra)
		run(boundLegacyCfC(cell))
		want := cfcLeafGrads(cell, extra)
		cfcCmpGradsNaN(t, got, want, true)
	}

	// Clamped regimes: forward and backward stay bit-identical (strict:
	// these seeds are finite).
	for _, ts := range []float64{1e-40, 1e300, 1e-3, 3e-2} {
		run := func(step func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable)) {
			cfcZeroAll(cell, extra)
			out, v := step(x, h, ts)
			autograd.Add(autograd.SumAll(out), autograd.SumAll(v)).Backward()
		}
		run(cell.Step)
		got := cfcLeafGrads(cell, extra)
		run(boundLegacyCfC(cell))
		want := cfcLeafGrads(cell, extra)
		cfcCmpGrads(t, got, want)
	}
}

// TestCfCFusedAdversarialFoldOverflow loads absurd synaptic weights so
// the activation blocks overflow the folds to +Inf: the identity MatMul's
// av*0 terms then spread NaN across the row (Inf*0), in BOTH the legacy
// graph path and the fused kernel's literal MatMul calls (the
// normFoldIdentity non-finite fallback). Forward and backward must stay
// identical, modulo the double-NaN payload corner.
func TestCfCFusedAdversarialFoldOverflow(t *testing.T) {
	rng := rand.New(rand.NewSource(999))
	cell := NewCfC(2, 3, nil, rng)
	for i := range cell.sigma.Data.Data {
		cell.sigma.Data.Data[i] = 1e37 // saturate: sigmoid hits exactly 1
	}
	for i := range cell.mu.Data.Data {
		cell.mu.Data.Data[i] = -1e37 // ... for every synapse (z = +Inf)
	}
	for i := range cell.w.Data.Data {
		cell.w.Data.Data[i] = 2e38 // softplus(x) = x for x > 20; two terms overflow the fold
	}
	x := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 2))
	h := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 3))
	extra := map[string]*autograd.Variable{"x": x, "h": h}

	outF, hF := cell.Step(x, h, 0.1)
	outL, hL := legacyCfCStep(cell, x, h, 0.1)
	fusedDiffBitsNaN(t, "overflow out", outF.Data, outL.Data)
	fusedDiffBitsNaN(t, "overflow state", hF.Data, hL.Data)
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
		cfcZeroAll(cell, extra)
		out, v := step(x, h, 0.1)
		autograd.Add(autograd.SumAll(out), autograd.SumAll(v)).Backward()
	}
	run(cell.Step)
	got := cfcLeafGrads(cell, extra)
	run(boundLegacyCfC(cell))
	want := cfcLeafGrads(cell, extra)
	cfcCmpGradsNaN(t, got, want, true)
}

// TestCfCFusedSignedZeroRowAccumulators pins the signed-zero ownership
// corners of the per-presynaptic row accumulators: with a saturating
// sigma the sigmoid output hits exactly 1, so 1-s is +0 and the block
// gradients carry signed zeros. With batch == 1 the graph path's
// reductions ADOPT their buffers (hadamardReduce's and negReduce's
// same-shape arms — no +0-seeded wash), and with units == 1 the Sub
// backward's preCol side is a same-shape passthrough; the fused kernel
// must keep the identical -0 sign bits. The u1 b1 case asserts a -0
// actually flows into mu/sigma/w, so the corner cannot pass vacuously.
func TestCfCFusedSignedZeroRowAccumulators(t *testing.T) {
	for _, tc := range []struct {
		name                string
		inDim, units, batch int
	}{
		{"u1 b1", 3, 1, 1},
		{"u1 b2", 3, 1, 2},
		{"u3 b1", 3, 3, 1},
		{"u1 b1 in1", 1, 1, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(42))
			cell := NewCfC(tc.inDim, tc.units, nil, rng)
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
			cfcZeroAll(cell, extra)
			build(cell.Step).Backward()
			got := cfcLeafGrads(cell, extra)
			cfcZeroAll(cell, extra)
			build(boundLegacyCfC(cell)).Backward()
			want := cfcLeafGrads(cell, extra)
			cfcCmpGrads(t, got, want)
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

// TestCfCFusedBackwardChainedBitExact drives the chained topology: a
// 2-step unroll where step 1's input is a graph function of step 0's
// OUTPUT, and a 3-step two-layer stack (layer 2's input is layer 1's
// output). The graph-level input affine keeps the chained contribution's
// interleaving with the fused region's deliveries identical in both paths.
func TestCfCFusedBackwardChainedBitExact(t *testing.T) {
	t.Run("t2 chain terminal loss", func(t *testing.T) {
		rng := rand.New(rand.NewSource(7))
		cell := NewCfC(3, 3, nil, rng)
		x0 := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 3))
		x1 := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 3))
		h0 := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 3))
		c1 := tensor.Uniform(rng, -1, 1, 1, 3)
		c2 := tensor.Uniform(rng, -1, 1, 2, 3)
		extra := map[string]*autograd.Variable{"x0": x0, "x1": x1, "h0": h0}
		run := func(step func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable)) {
			cfcZeroAll(cell, extra)
			y0, h1 := step(x0, h0, 0.1)
			x1c := autograd.Add(autograd.Hadamard(y0, autograd.Const(c1)), autograd.Const(c2))
			_, h2 := step(x1c, h1, 0.1)
			autograd.SumAll(h2).Backward()
		}
		run(cell.Step)
		got := cfcLeafGrads(cell, extra)
		run(boundLegacyCfC(cell))
		want := cfcLeafGrads(cell, extra)
		cfcCmpGrads(t, got, want)
	})

	t.Run("stacked layers seeded", func(t *testing.T) {
		rng := rand.New(rand.NewSource(11))
		l1 := NewCfC(4, 4, nil, rng)
		l2 := NewCfC(4, 4, nil, rng)
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
		snap := func(tag string, cells ...*CfC) map[string][]uint32 {
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
			for _, cell := range []*CfC{l1, l2} {
				for _, p := range cell.Parameters() {
					p.ZeroGrad()
				}
			}
			for _, v := range extra {
				v.ZeroGrad()
			}
		}
		run := func(stepOf func(*CfC) func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable)) map[string][]uint32 {
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
		got := run(func(c *CfC) func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable) { return c.Step })
		want := run(boundLegacyCfC)
		cfcCmpGrads(t, got, want)
	})
}

// TestCfCFusedLeafGradShapes pins the delivered gradient SHAPES per leaf:
// the fused backward must hand every leaf exactly the shape the legacy
// graph path hands it — including the [1, units] lift gleak/cm carry
// (autograd's documented 1D-lift quirk through the Softplus chain).
func TestCfCFusedLeafGradShapes(t *testing.T) {
	for _, tc := range cfcDiffCases() {
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
			cfcZeroAll(cell, extra)
			build(cell.Step).Backward()
			got := shapes()
			cfcZeroAll(cell, extra)
			build(boundLegacyCfC(cell)).Backward()
			want := shapes()
			for name, g := range got {
				if g != want[name] {
					t.Fatalf("%s: grad shape %s, legacy %s", name, g, want[name])
				}
			}
		})
	}
}

// TestCfCFusedPenaltyLeafShapePanic pins the consequence of the 1D lift:
// an L2 penalty Hadamard(p, p) over a 1D parameter delivers a [units]
// contribution against the step-side [1, units] one, so the accumulation
// panics with the engine's shape-mismatch message — in BOTH paths, the
// graph path's own quirk replicated, not a fused-kernel divergence.
func TestCfCFusedPenaltyLeafShapePanic(t *testing.T) {
	rng := rand.New(rand.NewSource(5150))
	cell := NewCfC(2, 4, nil, rng)
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
				cfcZeroAll(cell, extra)
				out, _ := step(x, h, 0.1)
				data := autograd.SumAll(out)
				pen := autograd.Scale(autograd.SumAll(autograd.Hadamard(p, p)), 0.01)
				autograd.Add(data, pen).Backward()
			}
		}
		mF := panicOf(build(cell.Step))
		mL := panicOf(build(boundLegacyCfC(cell)))
		if mF == "" || mF != mL || !strings.Contains(mF, "gradient shape mismatch") {
			t.Fatalf("param %d: fused panic %q, legacy panic %q: both must raise the identical shape-mismatch panic", pi, mF, mL)
		}
	}
}

// TestCfCFusedStateShapePanic pins the fused kernel's state validation:
// every non-exact h shape panics with the "state shape" message. The
// legacy graph path panics in the tensor layer for most of these, but
// SILENTLY broadcasts a broadcast-compatible wrong state ([batch, 1] and
// [1, units]) — inputs the Cell contract forbids; the fused Step is
// deliberately strict there (see nn/cfc_fused.go's header), and this test
// pins both the strict panic and the documented divergence.
func TestCfCFusedStateShapePanic(t *testing.T) {
	rng := rand.New(rand.NewSource(61))
	cell := NewCfC(2, 3, nil, rng)
	x := autograd.Var(tensor.New(2, 2))
	panicOf := func(f func()) (msg string) {
		defer func() {
			if r := recover(); r != nil {
				msg = fmt.Sprint(r)
			}
		}()
		f()
		return ""
	}
	// Shapes the legacy graph path rejects in the tensor layer: the fused
	// kernel rejects them too, with its explicit message.
	for _, bad := range []*tensor.Tensor{
		tensor.New(3, 3),    // wrong rows
		tensor.New(2, 4),    // wrong width (SliceRow out of range on legacy)
		tensor.New(6),       // rank 1
		tensor.New(2, 3, 1), // rank 3
	} {
		mF := panicOf(func() { cell.Step(x, autograd.Var(bad), 0.1) })
		if !strings.Contains(mF, "state shape") {
			t.Fatalf("Step with h shape %v: fused panic %q should name the state shape", bad.Shape, mF)
		}
		mL := panicOf(func() { legacyCfCStep(cell, x, autograd.Var(bad), 0.1) })
		if mL == "" {
			t.Fatalf("Step with h shape %v: legacy did not panic; the strictness note is stale", bad.Shape)
		}
	}
	// Broadcast-compatible wrong shapes: the legacy graph path silently
	// accepts them (the tensor-layer broadcast contract); the fused kernel
	// panics — the deliberate strictness the header documents.
	for _, bad := range []*tensor.Tensor{
		tensor.New(2, 1), // [batch, 1]
		tensor.New(1, 3), // [1, units]
		tensor.New(1, 1), // broadcast scalar
	} {
		mL := panicOf(func() { legacyCfCStep(cell, x, autograd.Var(bad), 0.1) })
		if mL != "" {
			t.Fatalf("legacy silently-broadcast corner %v now panics (%q); update the strictness note", bad.Shape, mL)
		}
		mF := panicOf(func() { cell.Step(x, autograd.Var(bad), 0.1) })
		if !strings.Contains(mF, "state shape") {
			t.Fatalf("Step with h shape %v: fused panic %q should name the state shape", bad.Shape, mF)
		}
	}
}

// TestCfCFusedGradShapeMismatchPanic pre-seeds the state leaf's Grad with
// an incompatible shape: delivering the replayed passthrough must panic
// with the engine's shape-mismatch message, exactly as the legacy Add
// backward's addGrad does on its first delivery.
func TestCfCFusedGradShapeMismatchPanic(t *testing.T) {
	rng := rand.New(rand.NewSource(67))
	cell := NewCfC(2, 3, nil, rng)
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
		out, _ := legacyCfCStep(cell, x, h, 0.1)
		h.Grad = tensor.New(3, 3)
		autograd.SumAll(out).Backward()
	})
	if !strings.Contains(mF, "gradient shape mismatch") || !strings.Contains(mL, "gradient shape mismatch") {
		t.Fatalf("fused %q, legacy %q: both must raise the shape-mismatch panic", mF, mL)
	}
}

// TestCfCFusedReversalFlipBitExact rewrites the erev/sErev storage in
// place (the Load path's polarity flip): the kernel's live aliases must
// observe the flip exactly as the graph path's shared-storage erevRows
// constants do — flipped outputs differ from the originals, and both
// paths stay bit-identical, forward and backward.
func TestCfCFusedReversalFlipBitExact(t *testing.T) {
	rng := rand.New(rand.NewSource(71))
	cell := NewCfC(3, 4, nil, rng)
	x := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 3))
	h := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 4))
	extra := map[string]*autograd.Variable{"x": x, "h": h}

	run := func(step func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable)) (map[string][]uint32, []uint32, []uint32) {
		cfcZeroAll(cell, extra)
		out, v := step(x, h, 0.1)
		autograd.Add(autograd.SumAll(out), autograd.SumAll(v)).Backward()
		return cfcLeafGrads(cell, extra), dataBits(out.Data), dataBits(v.Data)
	}
	gradsF0, outF0, hF0 := run(cell.Step)
	gradsL0, _, _ := run(boundLegacyCfC(cell))
	cfcCmpGrads(t, gradsF0, gradsL0)

	for i := range cell.erev.Data {
		cell.erev.Data[i] = -cell.erev.Data[i]
	}
	for i := range cell.sErev.Data {
		cell.sErev.Data[i] = -cell.sErev.Data[i]
	}
	gradsF1, outF1, hF1 := run(cell.Step)
	gradsL1, outL1, hL1 := run(boundLegacyCfC(cell))
	cfcCmpGrads(t, gradsF1, gradsL1)
	for k := range outF1 {
		if outF1[k] != outL1[k] || hF1[k] != hL1[k] {
			t.Fatalf("flipped forward element %d: fused %#x/%#x, legacy %#x/%#x", k, outF1[k], hF1[k], outL1[k], hL1[k])
		}
	}
	changed := false
	for k := range outF1 {
		if outF1[k] != outF0[k] || hF1[k] != hF0[k] {
			changed = true
		}
	}
	if !changed {
		t.Fatal("the polarity flip changed no output: the test would prove nothing")
	}
}

// TestCfCFusedMidFlightMutationDiscipline mutates parameters between the
// forward and the backward: mu/sigma (frozen by the graph's SliceRow
// nodes, and by the kernel's forward copies) must keep the forward-time
// values in BOTH paths, while vleak (read live by the graph's Hadamard
// backward, and by the kernel) must show the mutated value in BOTH paths
// — identical behavior, not necessarily identical to the unmutated run.
func TestCfCFusedMidFlightMutationDiscipline(t *testing.T) {
	rng := rand.New(rand.NewSource(73))
	cell := NewCfC(3, 4, nil, rng)
	x := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 3))
	h := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 4))
	extra := map[string]*autograd.Variable{"x": x, "h": h}
	so := tensor.Uniform(rng, -1, 1, 2, 4)

	run := func(step func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable), mutate func()) map[string][]uint32 {
		cfcZeroAll(cell, extra)
		out, v := step(x, h, 0.1)
		mutate()
		autograd.Add(
			autograd.SumAll(autograd.Hadamard(out, autograd.Const(so))),
			autograd.SumAll(autograd.Hadamard(v, autograd.Const(so)))).Backward()
		return cfcLeafGrads(cell, extra)
	}
	noop := func() {}
	mutMu := func() {
		for i := range cell.mu.Data.Data {
			cell.mu.Data.Data[i] = 100 + float32(i)
		}
	}
	mutVleak := func() {
		for i := range cell.vleak.Data.Data {
			cell.vleak.Data.Data[i] = -7 - float32(i)
		}
	}

	// Frozen leaves: mutating mu mid-flight changes nothing in either path
	// (both replay from the forward-frozen rows), so the mutated runs match
	// each other AND the unmutated baseline bit for bit. The mutations are
	// absolute assignments, but every run must start from the same values:
	// restore the mutated storage after each run.
	muSave := append([]float32(nil), cell.mu.Data.Data...)
	vleakSave := append([]float32(nil), cell.vleak.Data.Data...)
	restore := func() {
		copy(cell.mu.Data.Data, muSave)
		copy(cell.vleak.Data.Data, vleakSave)
	}
	baseF := run(cell.Step, noop)
	baseL := run(boundLegacyCfC(cell), noop)
	cfcCmpGrads(t, baseF, baseL)
	mutF := run(cell.Step, mutMu)
	restore()
	mutL := run(boundLegacyCfC(cell), mutMu)
	restore()
	cfcCmpGrads(t, mutF, mutL)
	cfcCmpGrads(t, mutF, baseF)

	// Live leaf: mutating vleak mid-flight moves the gradient in BOTH
	// paths identically (the graph's Hadamard backward reads the live
	// leaf; the kernel does the same).
	liveF := run(cell.Step, mutVleak)
	restore()
	liveL := run(boundLegacyCfC(cell), mutVleak)
	restore()
	cfcCmpGrads(t, liveF, liveL)
	// The live read is observable in GLEAK's gradient (vleak's own
	// gradient is gLV⊙gleak — it does not contain vleak's value): the
	// gleakSp-side contribution gLV⊙vleak must move identically in both
	// paths when vleak mutates mid-flight.
	if fmt.Sprint(liveF["gleak"]) == fmt.Sprint(baseF["gleak"]) {
		t.Fatal("the mid-flight vleak mutation moved no gleak gradient: the live-read contract is not exercised")
	}
}

// TestCfCFusedNodeAccount pins the fusion's structural win: a fused CfC
// Step records 9 op nodes + 15 leaves = 24 graph nodes at any dims
// (against the graph path's 66 + 14*(inDim+units)), with h the fused
// node's FIRST parent (the remat spine-freedom invariant).
func TestCfCFusedNodeAccount(t *testing.T) {
	for _, dims := range [][2]int{{1, 8}, {4, 16}, {2, 3}, {1, 1}, {3, 1}, {2, 32}} {
		inDim, units := dims[0], dims[1]
		rng := rand.New(rand.NewSource(3))
		cell := NewCfC(inDim, units, nil, rng)
		x := autograd.Var(tensor.Uniform(rng, -1, 1, 5, inDim))
		h := autograd.Var(tensor.Uniform(rng, -1, 1, 5, units))
		out, hN := cell.Step(x, h, 0.1)
		seen := make(map[*autograd.Variable]bool)
		ops, leaves := 0, 0
		for _, root := range []*autograd.Variable{out, hN} {
			for _, v := range autograd.TopoOrder(root) {
				if seen[v] {
					continue
				}
				seen[v] = true
				if len(v.Parents()) == 0 {
					leaves++
				} else {
					ops++
				}
			}
		}
		if ops != 9 || leaves != 15 {
			t.Fatalf("in=%d u=%d: fused step graph = %d ops + %d leaves, want 9 + 15", inDim, units, ops, leaves)
		}
		// h must be the fused node's first parent (the DFS descent hits
		// the state leaf before any parameter chain — no spine class).
		fused := hN
		if len(fused.Parents()) == 0 {
			t.Fatalf("in=%d u=%d: hNew is a leaf, expected the fused node", inDim, units)
		}
		if fused.Parents()[0] != h {
			t.Fatalf("in=%d u=%d: fused node's first parent is not h", inDim, units)
		}
	}
}

// TestSoftplus32MatchesTensor pins the kernel's softplus replica to
// tensor.Softplus bit for bit over a grid spanning both branches (the
// x > 20 passthrough included) and the non-finite inputs — the same
// formula, so strict Float32bits with no exemption.
func TestSoftplus32MatchesTensor(t *testing.T) {
	xs := []float32{
		float32(math.Inf(-1)), -1e38, -100, -25, -20.000002, -20, -19.999998,
		-1, -1e-8, float32(math.Copysign(0, -1)), 0, 1e-8, 1, 19, 20,
		20.000002, 21, 25, 100, 1e38, float32(math.Inf(1)), float32(math.NaN()),
	}
	in := tensor.FromData(append([]float32(nil), xs...), len(xs))
	want := tensor.Softplus(in)
	for i, x := range xs {
		got := softplus32(x)
		if math.Float32bits(got) != math.Float32bits(want.Data[i]) {
			t.Fatalf("softplus32(%v) = %v (bits %#x), tensor.Softplus %v (bits %#x)",
				x, got, math.Float32bits(got), want.Data[i], math.Float32bits(want.Data[i]))
		}
	}
}

// TestCfCFusedDecayRateCapEngaged drives the decayRate cap chain into its
// engaged regime — cm loaded so softplus(cm) underflows to 0 (cmEps = eps)
// and w loaded so kappa = G/eps (~3e35) overshoots the cap headroom
// hi = MaxFloat32/ts/1.0001 (~3.4e34 at ts = 1e4) while staying FINITE
// (the cap cannot engage at ts = 0.1: there hi = MaxFloat32 and an
// overshooting kappa is already +Inf — NaN in both paths). Then
// cap(k) = k - softplus(k - hi) = hi exactly (softplus32's x > 20
// passthrough arm), B = hi*ts stays finite, and both paths must agree
// bit for bit, forward and backward.
func TestCfCFusedDecayRateCapEngaged(t *testing.T) {
	rng := rand.New(rand.NewSource(1213))
	cell := NewCfC(2, 3, nil, rng)
	for i := range cell.cm.Data.Data {
		cell.cm.Data.Data[i] = -1e30 // softplus underflows to exactly 0
	}
	for i := range cell.w.Data.Data {
		cell.w.Data.Data[i] = 1e27 // G ~ 3e27, kappa ~ 3e35 > hi ~ 3.4e34, finite
	}
	x := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 2))
	h := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 3))
	extra := map[string]*autograd.Variable{"x": x, "h": h}

	outF, hF := cell.Step(x, h, 1e4)
	outL, hL := legacyCfCStep(cell, x, h, 1e4)
	fusedDiffBits(t, "capped out", outF.Data, outL.Data)
	fusedDiffBits(t, "capped state", hF.Data, hL.Data)
	assertFinite(t, "capped state", hF)
	// The cap must actually be engaged: with B = hi*ts the decay factor
	// saturates at 1 and the state relaxes fully to A (not the ts -> 0
	// fixed point).
	moved := false
	for k := range hF.Data.Data {
		if hF.Data.Data[k] != h.Data.Data[k] {
			moved = true
		}
	}
	if !moved {
		t.Fatal("the engaged cap froze the state; the scenario does not exercise the cap")
	}

	run := func(step func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable)) {
		cfcZeroAll(cell, extra)
		out, v := step(x, h, 1e4)
		autograd.Add(autograd.SumAll(out), autograd.SumAll(v)).Backward()
	}
	run(cell.Step)
	got := cfcLeafGrads(cell, extra)
	run(boundLegacyCfC(cell))
	want := cfcLeafGrads(cell, extra)
	cfcCmpGrads(t, got, want)
}
