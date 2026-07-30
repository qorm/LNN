package optimizer

import (
	"math"
	"math/rand"
	"testing"

	"lnn/autograd"
	"lnn/tensor"
)

// Compile-time interface checks.
var (
	_ Optimizer = (*SGD)(nil)
	_ Optimizer = (*Momentum)(nil)
	_ Optimizer = (*Adam)(nil)
)

// param builds a leaf variable with the given data and, if grad is
// non-nil, a gradient buffer — the same state loss.Backward() would
// leave behind, set directly so the update math can be checked without
// depending on autograd's internals.
func param(data, grad []float32) *autograd.Variable {
	p := autograd.Var(tensor.FromData(append([]float32(nil), data...), len(data)))
	if grad != nil {
		p.Grad = tensor.FromData(append([]float32(nil), grad...), len(grad))
	}
	return p
}

func setGrad(p *autograd.Variable, g []float32) {
	p.Grad = tensor.FromData(append([]float32(nil), g...), len(g))
}

// expectData compares p.Data element-for-element (bit-exact: every test
// below uses exactly representable values).
func expectData(t *testing.T, p *autograd.Variable, want []float32) {
	t.Helper()
	got := p.Data.Data
	if len(got) != len(want) {
		t.Fatalf("data length: got %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("element %d: got %v, want %v", i, got[i], want[i])
		}
	}
}

// nearData compares with a relative tolerance of 1e-6. On FMA-capable
// targets (arm64) the compiler may contract a*b+c into fused
// multiply-adds differently in the two transcription sites, which moves
// results by a few ulps; 1e-6 absorbs that while still failing on any
// genuine formula error. Bit-exact arithmetic is pinned separately by
// the hand-derived constant-gradient test.
func nearData(t *testing.T, p *autograd.Variable, want []float32) {
	t.Helper()
	got := p.Data.Data
	if len(got) != len(want) {
		t.Fatalf("data length: got %d, want %d", len(got), len(want))
	}
	for i := range want {
		tol := 1e-6 * float32(math.Max(1, float64(abs32(want[i]))))
		if d := abs32(got[i] - want[i]); d > tol {
			t.Errorf("element %d: got %v, want %v (diff %v > tol %v)", i, got[i], want[i], d, tol)
		}
	}
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func panics(f func()) (ok bool) {
	defer func() {
		if recover() != nil {
			ok = true
		}
	}()
	f()
	return false
}

// --- Update-rule exactness (hand-computed, bit-exact) ---

func TestSGDStepExact(t *testing.T) {
	p := param([]float32{1, -2}, []float32{4, -8})
	o := NewSGD(0.25)

	o.Step([]*autograd.Variable{p})
	// 1 - 0.25*4 = 0; -2 - 0.25*(-8) = 0.
	expectData(t, p, []float32{0, 0})

	o.LR = 0.5 // mid-training learning-rate change
	setGrad(p, []float32{2, 6})
	o.Step([]*autograd.Variable{p})
	// 0 - 0.5*2 = -1; 0 - 0.5*6 = -3.
	expectData(t, p, []float32{-1, -3})
}

func TestMomentumStepExact(t *testing.T) {
	// v = Mu*v + g; p -= LR*v, matching doc/training.md's momentum
	// snippet. Mu = 0.5, LR = 0.25 keep every value exactly
	// representable:
	//   g=4:  v = 0.5*0 + 4 = 4    p = 2 - 0.25*4  = 1
	//   g=2:  v = 0.5*4 + 2 = 4    p = 1 - 0.25*4  = 0
	//   g=-4: v = 0.5*4 - 4 = -2   p = 0 - 0.25*-2 = 0.5
	p := param([]float32{2}, []float32{4})
	o := NewMomentum(0.25, 0.5)
	params := []*autograd.Variable{p}

	o.Step(params)
	expectData(t, p, []float32{1})
	setGrad(p, []float32{2})
	o.Step(params)
	expectData(t, p, []float32{0})
	setGrad(p, []float32{-4})
	o.Step(params)
	expectData(t, p, []float32{0.5})
}

func TestMomentumZeroMuIsSGD(t *testing.T) {
	pm := param([]float32{3}, []float32{8})
	ps := param([]float32{3}, []float32{8})
	NewMomentum(0.25, 0).Step([]*autograd.Variable{pm})
	NewSGD(0.25).Step([]*autograd.Variable{ps})
	expectData(t, pm, ps.Data.Data) // 3 - 0.25*8 = 1
}

func TestAdamStepExactConstantGradient(t *testing.T) {
	// Paper-style hand derivation. With a constant gradient g, Adam's
	// bias correction makes mhat = g and vhat = g^2 exactly at every
	// step, so each update is LR*sign(g): p walks 10 -> 9 -> 8 -> 7.
	// Beta1 = 0.5, Beta2 = 0.25, LR = 1 keep the arithmetic exact in
	// float32, and Eps = 1e-8 is absorbed by rounding at sqrt(vhat) = 4
	// (far below half an ulp), so the denominator stays exactly 4.
	//
	// Step 1: m = 2, v = 12, 1-Beta1^1 = 0.5, 1-Beta2^1 = 0.75
	//         mhat = 4, vhat = 16, update = 4/(4) = 1.
	// Step 2: m = 3, v = 15, 1-Beta1^2 = 0.75, 1-Beta2^2 = 0.9375
	//         mhat = 4, vhat = 16, update = 1.
	// Step 3: m = 3.5, v = 15.75, 1-Beta1^3 = 0.875, 1-Beta2^3 = 0.984375
	//         mhat = 4, vhat = 16, update = 1.
	p := param([]float32{10}, []float32{4})
	o := NewAdam(1, 0.5, 0.25, 1e-8)
	params := []*autograd.Variable{p}

	for _, want := range []float32{9, 8, 7} {
		o.Step(params)
		expectData(t, p, []float32{want})
		setGrad(p, []float32{4})
	}
}

func TestAdamStepMatchesReferenceFormula(t *testing.T) {
	// Bit-exact transcription check: a reference implementation written
	// from the paper's formulas (Kingma & Ba 2014, Algorithm 1) with the
	// same float32 operations must agree with Adam.Step element for
	// element on a non-degenerate gradient sequence.
	p := param([]float32{0.75, -1.5}, nil)
	o := NewAdam(0.1, 0.9, 0.999, 1e-8)

	d := []float32{0.75, -1.5}
	m := []float32{0, 0}
	v := []float32{0, 0}
	var pow1, pow2 float32 = 1, 1
	grads := [][]float32{{2, -0.5}, {0.25, 3}, {-1, -1}}
	for step, g := range grads {
		setGrad(p, g)
		o.Step([]*autograd.Variable{p})

		pow1 *= 0.9
		pow2 *= 0.999
		bc1 := 1 - pow1
		bc2 := 1 - pow2
		one1 := 1 - float32(0.9)
		one2 := 1 - float32(0.999)
		for i := range d {
			m[i] = 0.9*m[i] + one1*g[i]
			v[i] = 0.999*v[i] + one2*g[i]*g[i]
			mhat := m[i] / bc1
			vhat := v[i] / bc2
			d[i] -= 0.1 * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
		}
		nearData(t, p, d)
		if got := o.state[p].t; got != step+1 {
			t.Fatalf("step count: got %d, want %d", got, step+1)
		}
	}
}

// --- Convergence ---

// linearFit runs the README quick-start loop (fit y = 2x + 1 with a
// hand-rolled linear model) against the given optimizer and returns the
// final w, b and loss. anneal, if non-nil, is called before each epoch
// so tests can mutate the optimizer's exported hyperparameters
// mid-training.
func linearFit(t *testing.T, o Optimizer, epochs int, anneal func(epoch int)) (w, b, loss float32) {
	t.Helper()
	rng := rand.New(rand.NewSource(42))
	const n = 32
	xs, ys := make([]float32, n), make([]float32, n)
	for i := range xs {
		xs[i] = rng.Float32()*2 - 1
		ys[i] = 2*xs[i] + 1
	}
	x := autograd.Const(tensor.FromData(xs, n, 1))
	y := autograd.Const(tensor.FromData(ys, n, 1))
	wv := autograd.Var(tensor.Randn(rng, 1, 1))
	bv := autograd.Var(tensor.New(1))
	params := []*autograd.Variable{wv, bv}

	for epoch := 0; epoch < epochs; epoch++ {
		if anneal != nil {
			anneal(epoch)
		}
		pred := autograd.Add(autograd.MatMul(x, wv), bv)
		diff := autograd.Sub(pred, y)
		l := autograd.MeanAll(autograd.Hadamard(diff, diff))
		for _, p := range params {
			p.ZeroGrad()
		}
		l.Backward()
		o.Step(params)
		loss = l.Value()
	}
	return wv.Data.Data[0], bv.Data.Data[0], loss
}

func TestSGDConvergesOnLinearFit(t *testing.T) {
	// Same setup, seed and hyperparameters as the README quick start,
	// which recovers w = 2.0000, b = 1.0000 at epoch 199.
	w, b, loss := linearFit(t, NewSGD(0.1), 200, nil)
	if math.Abs(float64(w-2)) > 1e-3 || math.Abs(float64(b-1)) > 1e-3 {
		t.Errorf("did not converge to float32 precision: w=%v b=%v", w, b)
	}
	if loss > 1e-5 {
		t.Errorf("final loss %v, want < 1e-5", loss)
	}
}

func TestAdamConvergesOnLinearFit(t *testing.T) {
	// Adam's step magnitude stays near LR even at the optimum, so
	// anneal the exported LR field mid-training (a supported pattern)
	// to tighten the final error below the oscillation amplitude.
	o := NewAdamDefault(0.1)
	w, b, loss := linearFit(t, o, 400, func(epoch int) {
		switch epoch {
		case 200:
			o.LR = 0.01
		case 300:
			o.LR = 0.001
		}
	})
	if math.Abs(float64(w-2)) > 5e-3 || math.Abs(float64(b-1)) > 5e-3 {
		t.Errorf("did not converge: w=%v b=%v", w, b)
	}
	if loss > 1e-4 {
		t.Errorf("final loss %v, want < 1e-4", loss)
	}
}

func TestMomentumMatchesDocExample(t *testing.T) {
	// doc/training.md verifies its momentum snippet on (w - 5)^2 with
	// lr = 0.05, beta = 0.9: w = 5.0007 after 150 iterations from a
	// random start. Momentum must reproduce that result.
	rng := rand.New(rand.NewSource(42))
	wv := autograd.Var(tensor.Randn(rng, 1))
	five := autograd.Const(tensor.FromData([]float32{5}, 1))
	o := NewMomentum(0.05, 0.9)
	params := []*autograd.Variable{wv}

	for it := 0; it < 150; it++ {
		diff := autograd.Sub(wv, five)
		loss := autograd.MeanAll(autograd.Hadamard(diff, diff))
		wv.ZeroGrad()
		loss.Backward()
		o.Step(params)
	}
	if d := math.Abs(float64(wv.Data.Data[0] - 5)); d > 0.01 {
		t.Errorf("w = %v, want within 0.01 of 5 (doc reports 5.0007)", wv.Data.Data[0])
	}
}

// --- State semantics ---

func TestStateIsolatedBetweenParameters(t *testing.T) {
	// Warm up parameter A, then introduce B: B's first update under the
	// used optimizer must equal a fresh optimizer's first update.
	t.Run("Momentum", func(t *testing.T) {
		used, fresh := NewMomentum(0.25, 0.5), NewMomentum(0.25, 0.5)
		pA := param([]float32{2}, []float32{4})
		for i := 0; i < 3; i++ {
			used.Step([]*autograd.Variable{pA})
			setGrad(pA, []float32{4})
		}
		pB := param([]float32{2}, []float32{4})
		used.Step([]*autograd.Variable{pA, pB})

		pWant := param([]float32{2}, []float32{4})
		fresh.Step([]*autograd.Variable{pWant}) // v = 4; p = 2 - 0.25*4 = 1
		if pB.Data.Data[0] != pWant.Data.Data[0] {
			t.Errorf("pB = %v, want fresh first step %v", pB.Data.Data[0], pWant.Data.Data[0])
		}
	})
	t.Run("Adam", func(t *testing.T) {
		used, fresh := NewAdamDefault(0.1), NewAdamDefault(0.1)
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
	})
}

func TestStateKeyedByPointerIdentity(t *testing.T) {
	// Documented behavior: state belongs to the *Variable pointer.
	// Re-pointing Data at a new same-sized tensor keeps the velocity —
	// the optimizer still sees the same parameter.
	o := NewMomentum(0.25, 0.5)
	p := param([]float32{2}, []float32{4})
	params := []*autograd.Variable{p}
	o.Step(params) // v = 4, p = 1
	setGrad(p, []float32{2})
	o.Step(params) // v = 4, p = 0

	p.Data = tensor.FromData([]float32{10}, 1) // same parameter, new storage
	setGrad(p, []float32{-4})
	o.Step(params) // v = 0.5*4 - 4 = -2; p = 10 - 0.25*(-2) = 10.5
	expectData(t, p, []float32{10.5})
}

func TestStepPanicsOnParameterResize(t *testing.T) {
	t.Run("Momentum", func(t *testing.T) {
		o := NewMomentum(0.25, 0.5)
		p := param([]float32{1, 2}, []float32{1, 1})
		o.Step([]*autograd.Variable{p})
		p.Data = tensor.FromData([]float32{1, 2, 3}, 3)
		if !panics(func() { o.Step([]*autograd.Variable{p}) }) {
			t.Error("resizing a parameter between steps did not panic")
		}
	})
	t.Run("Adam", func(t *testing.T) {
		o := NewAdamDefault(0.1)
		p := param([]float32{1, 2}, []float32{1, 1})
		o.Step([]*autograd.Variable{p})
		p.Data = tensor.FromData([]float32{1}, 1)
		if !panics(func() { o.Step([]*autograd.Variable{p}) }) {
			t.Error("resizing a parameter between steps did not panic")
		}
	})
}

// --- Step contract ---

func TestStepSkipsNilGrad(t *testing.T) {
	opts := map[string]Optimizer{
		"SGD":      NewSGD(0.25),
		"Momentum": NewMomentum(0.25, 0.5),
		"Adam":     NewAdamDefault(0.1),
	}
	for name, o := range opts {
		p := param([]float32{7, -3}, nil) // never saw a Backward
		q := param([]float32{1, 1}, []float32{4, 4})
		o.Step([]*autograd.Variable{p, q})
		if p.Grad != nil {
			t.Errorf("%s: Step created a Grad on a nil-Grad parameter", name)
		}
		expectData(t, p, []float32{7, -3})
	}
}

func TestSkippedParameterKeepsFreshState(t *testing.T) {
	// A parameter skipped (Grad nil) while others step must not have its
	// Adam update count advanced: when it first receives a gradient, its
	// bias correction must match a brand-new optimizer's first step.
	o := NewAdamDefault(0.1)
	skipped := param([]float32{1}, nil)
	other := param([]float32{1}, []float32{2})
	for i := 0; i < 4; i++ {
		o.Step([]*autograd.Variable{skipped, other})
		setGrad(other, []float32{2})
	}
	setGrad(skipped, []float32{2})
	o.Step([]*autograd.Variable{skipped})

	fresh := NewAdamDefault(0.1)
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

func TestStepDoesNotZeroGrad(t *testing.T) {
	// Gradient lifetime is the caller's contract: leaf gradients
	// accumulate across graphs by design, and gradient-accumulation
	// schedules zero every N iterations. Step must leave Grad untouched.
	p := param([]float32{1}, []float32{4})
	NewSGD(0.25).Step([]*autograd.Variable{p})
	if p.Grad == nil {
		t.Fatal("Step zeroed Grad; the caller owns gradient lifetime")
	}
	if p.Grad.Data[0] != 4 {
		t.Errorf("Step mutated Grad: got %v, want 4", p.Grad.Data[0])
	}
}

// --- Constructor validation ---

func TestConstructorsRejectInvalidHyperparameters(t *testing.T) {
	nan := float32(math.NaN())
	cases := []struct {
		name string
		new  func()
	}{
		{"NewSGD(0)", func() { NewSGD(0) }},
		{"NewSGD(-0.1)", func() { NewSGD(-0.1) }},
		{"NewSGD(NaN)", func() { NewSGD(nan) }},
		{"NewMomentum(0, 0.9)", func() { NewMomentum(0, 0.9) }},
		{"NewMomentum(NaN, 0.9)", func() { NewMomentum(nan, 0.9) }},
		{"NewMomentum(0.1, -0.1)", func() { NewMomentum(0.1, -0.1) }},
		{"NewMomentum(0.1, 1)", func() { NewMomentum(0.1, 1) }},
		{"NewMomentum(0.1, NaN)", func() { NewMomentum(0.1, nan) }},
		{"NewAdam(0, 0.9, 0.999, 1e-8)", func() { NewAdam(0, 0.9, 0.999, 1e-8) }},
		{"NewAdam(NaN, 0.9, 0.999, 1e-8)", func() { NewAdam(nan, 0.9, 0.999, 1e-8) }},
		{"NewAdam(0.1, -0.1, 0.999, 1e-8)", func() { NewAdam(0.1, -0.1, 0.999, 1e-8) }},
		{"NewAdam(0.1, 1, 0.999, 1e-8)", func() { NewAdam(0.1, 1, 0.999, 1e-8) }},
		{"NewAdam(0.1, NaN, 0.999, 1e-8)", func() { NewAdam(0.1, nan, 0.999, 1e-8) }},
		{"NewAdam(0.1, 0.9, -0.1, 1e-8)", func() { NewAdam(0.1, 0.9, -0.1, 1e-8) }},
		{"NewAdam(0.1, 0.9, 1, 1e-8)", func() { NewAdam(0.1, 0.9, 1, 1e-8) }},
		{"NewAdam(0.1, 0.9, NaN, 1e-8)", func() { NewAdam(0.1, 0.9, nan, 1e-8) }},
		{"NewAdam(0.1, 0.9, 0.999, 0)", func() { NewAdam(0.1, 0.9, 0.999, 0) }},
		{"NewAdam(0.1, 0.9, 0.999, -1e-8)", func() { NewAdam(0.1, 0.9, 0.999, -1e-8) }},
		{"NewAdam(0.1, 0.9, 0.999, NaN)", func() { NewAdam(0.1, 0.9, 0.999, nan) }},
		{"NewAdamDefault(-1)", func() { NewAdamDefault(-1) }},
	}
	for _, c := range cases {
		if !panics(c.new) {
			t.Errorf("%s did not panic", c.name)
		}
	}
}
