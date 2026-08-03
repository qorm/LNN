package optimizer

import (
	"math"
	"math/rand"
	"testing"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/tensor"
)

// Compile-time interface check.
var _ Optimizer = (*AdEMAMix)(nil)

// --- Update-rule exactness (hand-computed, bit-exact) ---

func TestAdEMAMixStepExactConstantGradient(t *testing.T) {
	// Paper-style hand derivation pinning TWO AdEMAMix-specific behaviors
	// bit for bit: the slow EMA m2 carries NO bias correction, and it
	// enters the update scaled by Alpha. With a constant gradient g the
	// fast EMA's bias correction makes m1hat = g and vhat = g^2 exactly at
	// every step, while m2 fills as (1-Beta3^t)*g, so the update is
	// LR*(g + Alpha*(1-Beta3^t)*g)/(|g| + Eps). Beta1 = 0.5, Beta2 = 0.25,
	// Beta3 = 0.5, Alpha = 2, LR = 1 keep every value exactly
	// representable, and Eps = 1e-8 is absorbed by rounding at
	// sqrt(vhat) = 4 (far below half an ulp), so the denominator is
	// exactly 4.
	//
	// Step 1: m1 = 2, m2 = 2, v = 12; m1hat = 4, vhat = 16
	//         update = (4 + 2*2)/4 = 2     p = 10 - 2  = 8
	// Step 2: m1 = 3, m2 = 3, v = 15; m1hat = 4, vhat = 16
	//         update = (4 + 2*3)/4 = 2.5   p = 8 - 2.5 = 5.5
	// Step 3: m1 = 3.5, m2 = 3.5, v = 15.75; m1hat = 4, vhat = 16
	//         update = (4 + 2*3.5)/4 = 2.75 p = 5.5 - 2.75 = 2.75
	//
	// A bias-corrected m2 would give m2hat = 4 at step 1 and p = 7, so
	// this test fails on the classic AdEMAMix mistranscription.
	p := param([]float32{10}, []float32{4})
	o := NewAdEMAMix(1, 0.5, 0.25, 0.5, 2, 0, 1e-8)
	params := []*autograd.Variable{p}

	for _, want := range []float32{8, 5.5, 2.75} {
		o.Step(params)
		expectData(t, p, []float32{want})
		setGrad(p, []float32{4})
	}
}

// --- Reference-formula comparison (float64, tolerance + discrimination) ---

// ademamixReference holds a float64 simulation of the paper's equations
// (arXiv:2409.03137, the "AdEMAMix" display and Algorithm 1), independent
// of the implementation under test. tamper selects deliberate
// mistranscriptions for the discrimination gate.
type ademamixReference struct {
	m1, m2, v []float64
	t         int
}

// ademamixRefStep applies one reference update to d (kept in float64).
// tamper: 0 = faithful; 1 = slow EMA bias-corrected (the paper's m2
// carries none); 2 = alpha taken constant (no warmup ramp); 3 = beta3
// warmed up LINEARLY in beta3 instead of in the EMA half-life.
func ademamixRefStep(d []float64, st *ademamixReference, g []float32, lr, b1, b2, b3, alpha float32, warmup int, eps float32, tamper int) {
	st.t++
	fb1, fb2 := float64(b1), float64(b2)
	alphaT, beta3T := float64(alpha), float64(b3)
	if warmup > 0 && st.t < warmup {
		switch tamper {
		case 2:
			// alpha stays at its final value from the first step.
		case 3:
			alphaT = float64(st.t) / float64(warmup) * float64(alpha)
			beta3T = float64(b1) + float64(st.t)/float64(warmup)*(float64(b3)-float64(b1))
		default:
			alphaT = float64(st.t) / float64(warmup) * float64(alpha)
			// The paper's closed form f_beta3 (half-life interpolation),
			// independent of the implementation's f/f_inv form.
			lns, ln3 := math.Log(fb1), math.Log(float64(b3))
			a := float64(st.t) / float64(warmup)
			beta3T = math.Min(math.Exp(lns*ln3/((1-a)*ln3+a*lns)), float64(b3))
		}
	}
	bc1 := 1 - math.Pow(fb1, float64(st.t))
	bc2 := 1 - math.Pow(fb2, float64(st.t))
	for i := range d {
		gi := float64(g[i])
		st.m1[i] = fb1*st.m1[i] + (1-fb1)*gi
		st.m2[i] = beta3T*st.m2[i] + (1-beta3T)*gi
		st.v[i] = fb2*st.v[i] + (1-fb2)*gi*gi
		m2use := st.m2[i]
		if tamper == 1 {
			m2use = st.m2[i] / (1 - math.Pow(beta3T, float64(st.t)))
		}
		d[i] -= float64(lr) * (st.m1[i]/bc1 + alphaT*m2use) / (math.Sqrt(st.v[i]/bc2) + float64(eps))
	}
}

// relDiff returns the max elementwise relative difference between the
// float32 parameter and the float64 reference (relative to |ref|, floored
// at 1 to stay meaningful near zero).
func relDiff(got []float32, ref []float64) float64 {
	worst := 0.0
	for i := range ref {
		d := math.Abs(float64(got[i]) - ref[i])
		if r := d / math.Max(1, math.Abs(ref[i])); r > worst {
			worst = r
		}
	}
	return worst
}

func TestAdEMAMixStepMatchesReferenceFormula(t *testing.T) {
	// 30 steps over two parameters with the alpha/beta3 warmup schedulers
	// ACTIVE for the first 12 (so the boundary at t = 12 is crossed) and a
	// non-degenerate gradient sequence. The float64 reference is written
	// from the paper's equations; the tolerance absorbs float32-vs-float64
	// rounding (the library defines its own float32 op order; there is no
	// cross-implementation bit-exactness obligation).
	const steps, warmup = 30, 12
	lr, b1, b2, b3, alpha, eps := float32(0.02), float32(0.9), float32(0.99), float32(0.999), float32(2.5), float32(1e-8)
	rng := rand.New(rand.NewSource(7))
	grads := make([][][]float32, steps)
	for s := range grads {
		grads[s] = [][]float32{
			{rng.Float32()*4 - 2, rng.Float32()*4 - 2},
			{rng.Float32()*4 - 2, rng.Float32()*4 - 2, rng.Float32()*4 - 2},
		}
	}

	run := func(tamper int) (diff0, diff1 float64) {
		p0 := param([]float32{0.75, -1.5}, nil)
		p1 := param([]float32{-0.5, 2, 0.25}, nil)
		o := NewAdEMAMix(lr, b1, b2, b3, alpha, warmup, eps)
		d0 := []float64{0.75, -1.5}
		d1 := []float64{-0.5, 2, 0.25}
		r0 := &ademamixReference{m1: make([]float64, 2), m2: make([]float64, 2), v: make([]float64, 2)}
		r1 := &ademamixReference{m1: make([]float64, 3), m2: make([]float64, 3), v: make([]float64, 3)}
		for s := 0; s < steps; s++ {
			setGrad(p0, grads[s][0])
			setGrad(p1, grads[s][1])
			o.Step([]*autograd.Variable{p0, p1})
			ademamixRefStep(d0, r0, grads[s][0], lr, b1, b2, b3, alpha, warmup, eps, tamper)
			ademamixRefStep(d1, r1, grads[s][1], lr, b1, b2, b3, alpha, warmup, eps, tamper)
		}
		// Parameter trajectories and the internal moment buffers alike.
		diff0 = relDiff(p0.Data.Data, d0)
		diff1 = relDiff(p1.Data.Data, d1)
		st0, st1 := o.state[p0], o.state[p1]
		for i := range d0 {
			for _, pair := range [][2]float64{
				{float64(st0.m1[i]), r0.m1[i]}, {float64(st0.m2[i]), r0.m2[i]}, {float64(st0.v[i]), r0.v[i]},
			} {
				if d := math.Abs(pair[0]-pair[1]) / math.Max(1, math.Abs(pair[1])); d > diff0 {
					diff0 = d
				}
			}
		}
		for i := range d1 {
			for _, pair := range [][2]float64{
				{float64(st1.m1[i]), r1.m1[i]}, {float64(st1.m2[i]), r1.m2[i]}, {float64(st1.v[i]), r1.v[i]},
			} {
				if d := math.Abs(pair[0]-pair[1]) / math.Max(1, math.Abs(pair[1])); d > diff1 {
					diff1 = d
				}
			}
		}
		return diff0, diff1
	}

	const tol = 1e-5
	d0, d1 := run(0)
	if d0 > tol || d1 > tol {
		t.Fatalf("faithful run diverges from the paper reference: p0 %g, p1 %g (tol %g)", d0, d1, tol)
	}
	t.Logf("faithful run max rel diff vs float64 reference: p0 %g, p1 %g", d0, d1)

	// Discrimination gate: the tolerance must still reject the classic
	// mistranscriptions, each by a wide margin.
	for _, tc := range []struct {
		tamper int
		name   string
	}{
		{1, "bias-corrected slow EMA"},
		{2, "alpha without warmup"},
		{3, "beta3 linear (not half-life) warmup"},
	} {
		d0, d1 := run(tc.tamper)
		if worst := math.Max(d0, d1); worst < 1e-2 {
			t.Errorf("%s: tampered reference stays within %g of the implementation — the tolerance gate has no teeth", tc.name, worst)
		} else {
			t.Logf("%s: correctly rejected, max rel diff %g", tc.name, worst)
		}
	}
}

// --- Warmup schedulers against the official formulas ---

func TestAdEMAMixSchedulersMatchOfficialFormulas(t *testing.T) {
	// The official apple/ml-ademamix schedulers, transcribed independently:
	//   linear_warmup_scheduler:      a = t/T; (1-a)*0 + a*alpha
	//   linear_hl_warmup_scheduler:   f(b) = ln(0.5)/ln(b+1e-8) - 1,
	//                                 fInv(x) = 0.5^(1/(x+1)),
	//                                 fInv((1-a)*f(beta1) + a*f(beta3))
	// and the paper's closed form for beta3:
	//   min(exp(ln(beta1)*ln(beta3) / ((1-a)*ln(beta3) + a*ln(beta1))), beta3)
	const warmup = 250
	b1, b3, alpha := float32(0.9), float32(0.9999), float32(5)
	for _, tt := range []int{1, 2, 7, 100, warmup - 1} {
		a := float64(tt) / float64(warmup)

		gotA := ademamixAlpha(tt, warmup, alpha)
		wantA := float32(a * float64(alpha)) // (1-a)*0 + a*alpha
		if gotA != wantA {
			t.Errorf("alpha(%d): got %v, want official %v", tt, gotA, wantA)
		}

		f := func(beta float64) float64 { return math.Log(0.5)/math.Log(beta+1e-8) - 1 }
		fInv := func(x float64) float64 { return math.Pow(0.5, 1/(x+1)) }
		gotB := ademamixBeta3(tt, warmup, b1, b3)
		wantB := float32(fInv((1-a)*f(float64(b1)) + a*f(float64(b3))))
		if gotB != wantB {
			t.Errorf("beta3(%d): got %v, want official %v", tt, gotB, wantB)
		}

		// The paper's closed form must agree with the official f/f_inv form
		// up to the official's +1e-8 inside the logarithm (the paper writes
		// ln(beta) where the repo writes ln(beta+1e-8)): algebraically the
		// same scheduler, systematically ~1e-8 relative apart in the log,
		// which the half-life inversion amplifies to ~1e-6 here.
		lns, ln3 := math.Log(float64(b1)), math.Log(float64(b3))
		paper := math.Min(math.Exp(lns*ln3/((1-a)*ln3+a*lns)), float64(b3))
		if d := math.Abs(float64(gotB) - paper); d > 1e-5 {
			t.Errorf("beta3(%d): implementation %v vs paper closed form %v differ by %g", tt, gotB, paper, d)
		}
	}
	// Monotonicity sanity along the whole ramp: alpha increases linearly,
	// beta3 increases from Beta1 toward Beta3 and ends near it.
	prevA, prevB := float32(0), float32(0)
	for tt := 1; tt < warmup; tt++ {
		av, bv := ademamixAlpha(tt, warmup, alpha), ademamixBeta3(tt, warmup, b1, b3)
		if av < prevA || bv < prevB {
			t.Fatalf("schedulers not monotone at t=%d: alpha %v<%v, beta3 %v<%v", tt, av, prevA, bv, prevB)
		}
		prevA, prevB = av, bv
	}
	// beta3 at the last scheduled step is one ramp increment short of the
	// target; Step uses the final value from t >= warmup on (crossed by the
	// reference test above, whose 30 steps pass t = 12).
}

func TestAdEMAMixWarmupOneIsConstant(t *testing.T) {
	// Warmup = 1 has no scheduled step (t < 1 never holds for t >= 1), so
	// it must behave exactly like Warmup = 0.
	newPair := func() (*AdEMAMix, *autograd.Variable) {
		return NewAdEMAMix(0.02, 0.9, 0.99, 0.999, 2.5, 0, 1e-8), param([]float32{0.75, -1.5}, []float32{1, -2})
	}
	o0, p0 := newPair()
	o1, p1 := newPair()
	o1.Warmup = 1
	for s := 0; s < 5; s++ {
		g := []float32{float32(s) - 2, 0.5}
		setGrad(p0, g)
		setGrad(p1, g)
		o0.Step([]*autograd.Variable{p0})
		o1.Step([]*autograd.Variable{p1})
	}
	if !sameBits(p0.Data.Data, p1.Data.Data) {
		t.Errorf("Warmup=1 differs from Warmup=0: %v vs %v", p1.Data.Data, p0.Data.Data)
	}
}

// --- Convergence ---

func TestAdEMAMixConvergesOnLinearFit(t *testing.T) {
	// Same linear-fit harness as the other optimizers. Two AdEMAMix-specific
	// notes: (i) the normalized update keeps the step size near
	// LR*(1+Alpha) even at the optimum — like Adam, the exported LR is
	// annealed mid-training (a supported pattern) to settle the residual
	// orbit; (ii) Beta3 = 0.99, not the paper's 0.9999: the paper's
	// slow-EMA memory horizon (~10k steps) exceeds this whole toy run, so
	// the toy uses a horizon matched to it. Faithfulness to the paper's
	// equations — including Beta3 = 0.9999 arithmetic — is pinned by the
	// exact and reference tests, not by this one.
	o := NewAdEMAMix(0.05, 0.9, 0.999, 0.99, 2, 100, 1e-8)
	w, b, loss := linearFit(t, o, 1100, func(epoch int) {
		switch epoch {
		case 300:
			o.LR = 0.01
		case 600:
			o.LR = 0.002
		case 900:
			o.LR = 0.0005
		}
	})
	if math.Abs(float64(w-2)) > 5e-3 || math.Abs(float64(b-1)) > 5e-3 {
		t.Errorf("did not converge: w=%v b=%v", w, b)
	}
	if loss > 1e-4 {
		t.Errorf("final loss %v, want < 1e-4", loss)
	}
}

// --- State semantics ---

func TestAdEMAMixStateIsolatedBetweenParameters(t *testing.T) {
	// Warm up parameter A, then introduce B: B's first update under the
	// used optimizer must equal a fresh optimizer's first update.
	used, fresh := NewAdEMAMixDefault(0.01, 0), NewAdEMAMixDefault(0.01, 0)
	pA := param([]float32{1}, []float32{3})
	for i := 0; i < 3; i++ {
		used.Step([]*autograd.Variable{pA})
		setGrad(pA, []float32{3})
	}
	pB := param([]float32{1}, []float32{3})
	used.Step([]*autograd.Variable{pA, pB})

	pWant := param([]float32{1}, []float32{3})
	fresh.Step([]*autograd.Variable{pWant})
	if pB.Data.Data[0] != pWant.Data.Data[0] {
		t.Errorf("pB = %v, want fresh first step %v", pB.Data.Data[0], pWant.Data.Data[0])
	}
}

func TestAdEMAMixStateKeyedByPointerIdentity(t *testing.T) {
	// Re-pointing Data at a new same-sized tensor keeps the moments: the
	// optimizer sees the same parameter. Compare against an equivalent
	// run whose parameter was never re-pointed but whose Data was set to
	// the same value before the second step.
	mk := func() (*AdEMAMix, *autograd.Variable) {
		return NewAdEMAMixDefault(0.01, 0), param([]float32{2}, []float32{4})
	}
	o1, p1 := mk()
	o1.Step([]*autograd.Variable{p1})
	p1.Data = p1.Data.Clone() // same parameter, new storage, same values
	setGrad(p1, []float32{2})
	o1.Step([]*autograd.Variable{p1})

	o2, p2 := mk()
	o2.Step([]*autograd.Variable{p2})
	setGrad(p2, []float32{2})
	o2.Step([]*autograd.Variable{p2})
	if !sameBits(p1.Data.Data, p2.Data.Data) {
		t.Errorf("re-pointed parameter stepped differently: %v vs %v", p1.Data.Data, p2.Data.Data)
	}
	if got := o1.state[p1].t; got != 2 {
		t.Errorf("update count after re-point: got %d, want 2", got)
	}
}

func TestAdEMAMixStepPanicsOnParameterResize(t *testing.T) {
	o := NewAdEMAMixDefault(0.01, 0)
	p := param([]float32{1, 2}, []float32{1, 1})
	o.Step([]*autograd.Variable{p})
	p.Data = tensor.FromData([]float32{1, 2, 3}, 3)
	if !panics(func() { o.Step([]*autograd.Variable{p}) }) {
		t.Error("resizing a parameter between steps did not panic")
	}
}

func TestAdEMAMixSkippedParameterKeepsFreshState(t *testing.T) {
	// A parameter skipped (Grad nil) while others step must not have its
	// update count advanced: its first real update matches a brand-new
	// optimizer's first step, bias corrections and scheduler lookups
	// alike.
	o := NewAdEMAMixDefault(0.01, 40)
	skipped := param([]float32{1}, nil)
	other := param([]float32{1}, []float32{2})
	for i := 0; i < 4; i++ {
		o.Step([]*autograd.Variable{skipped, other})
		setGrad(other, []float32{2})
	}
	setGrad(skipped, []float32{2})
	o.Step([]*autograd.Variable{skipped})

	fresh := NewAdEMAMixDefault(0.01, 40)
	pWant := param([]float32{1}, []float32{2})
	fresh.Step([]*autograd.Variable{pWant})
	if skipped.Data.Data[0] != pWant.Data.Data[0] {
		t.Errorf("skipped parameter's first real update = %v, want fresh %v",
			skipped.Data.Data[0], pWant.Data.Data[0])
	}
	if got := o.state[skipped].t; got != 1 {
		t.Errorf("update count for skipped parameter: got %d, want 1", got)
	}
}

// --- Constructor validation ---

func TestAdEMAMixConstructorsRejectInvalidHyperparameters(t *testing.T) {
	nan := float32(math.NaN())
	cases := []struct {
		name string
		new  func()
	}{
		{"NewAdEMAMix(0, ...)", func() { NewAdEMAMix(0, 0.9, 0.999, 0.9999, 5, 10, 1e-8) }},
		{"NewAdEMAMix(NaN, ...)", func() { NewAdEMAMix(nan, 0.9, 0.999, 0.9999, 5, 10, 1e-8) }},
		{"NewAdEMAMix beta1 -0.1", func() { NewAdEMAMix(0.1, -0.1, 0.999, 0.9999, 5, 10, 1e-8) }},
		{"NewAdEMAMix beta1 1", func() { NewAdEMAMix(0.1, 1, 0.999, 0.9999, 5, 10, 1e-8) }},
		{"NewAdEMAMix beta1 NaN", func() { NewAdEMAMix(0.1, nan, 0.999, 0.9999, 5, 10, 1e-8) }},
		{"NewAdEMAMix beta2 -0.1", func() { NewAdEMAMix(0.1, 0.9, -0.1, 0.9999, 5, 10, 1e-8) }},
		{"NewAdEMAMix beta2 1", func() { NewAdEMAMix(0.1, 0.9, 1, 0.9999, 5, 10, 1e-8) }},
		{"NewAdEMAMix beta2 NaN", func() { NewAdEMAMix(0.1, 0.9, nan, 0.9999, 5, 10, 1e-8) }},
		{"NewAdEMAMix beta3 -0.1", func() { NewAdEMAMix(0.1, 0.9, 0.999, -0.1, 5, 10, 1e-8) }},
		{"NewAdEMAMix beta3 1", func() { NewAdEMAMix(0.1, 0.9, 0.999, 1, 5, 10, 1e-8) }},
		{"NewAdEMAMix beta3 NaN", func() { NewAdEMAMix(0.1, 0.9, 0.999, nan, 5, 10, 1e-8) }},
		{"NewAdEMAMix alpha -1", func() { NewAdEMAMix(0.1, 0.9, 0.999, 0.9999, -1, 10, 1e-8) }},
		{"NewAdEMAMix alpha NaN", func() { NewAdEMAMix(0.1, 0.9, 0.999, 0.9999, nan, 10, 1e-8) }},
		{"NewAdEMAMix warmup -1", func() { NewAdEMAMix(0.1, 0.9, 0.999, 0.9999, 5, -1, 1e-8) }},
		{"NewAdEMAMix eps 0", func() { NewAdEMAMix(0.1, 0.9, 0.999, 0.9999, 5, 10, 0) }},
		{"NewAdEMAMix eps NaN", func() { NewAdEMAMix(0.1, 0.9, 0.999, 0.9999, 5, 10, nan) }},
		{"NewAdEMAMixDefault(-1)", func() { NewAdEMAMixDefault(-1, 10) }},
		{"NewAdEMAMixDefault warmup -1", func() { NewAdEMAMixDefault(0.1, -1) }},
	}
	for _, c := range cases {
		if !panics(c.new) {
			t.Errorf("%s did not panic", c.name)
		}
	}
}
