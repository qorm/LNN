package optimizer

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/tensor"
)

// Compile-time interface check.
var _ Optimizer = (*ScheduleFreeAdamW)(nil)

// --- Update-rule exactness (hand-computed, bit-exact) ---

func TestScheduleFreeStepExactConstantGradient(t *testing.T) {
	// Hand derivation pinning the y-in-place update form of the official
	// implementation: on the first step c = 1 so y snaps to z; with a
	// constant gradient the normalized gradient is exactly sign(g).
	// LR = 1, Beta1 = 0.5, Beta2 = 0.25, g = 4 keep the arithmetic exact,
	// and Eps = 1e-8 is absorbed by rounding at sqrt(vhat) = 4.
	//
	// Step 1 (k=0): v = 12, vhat = 16, gn = 1; weight = 1, wsum = 1, c = 1
	//         y = 10 + 1*(10-10) + 1*(0.5*0-1)*1 = 9;  z = 10 - 1 = 9
	// Step 2 (k=1): v = 15, vhat = 16, gn = 1; wsum = 2, c = 0.5
	//         y = 9 + 0.5*(9-9) + 1*(0.5*0.5-1)*1 = 8.25;  z = 9 - 1 = 8
	p := param([]float32{10}, []float32{4})
	o := NewScheduleFreeAdamW(1, 0.5, 0.25, 1e-8)
	params := []*autograd.Variable{p}

	o.Step(params)
	expectData(t, p, []float32{9})
	if z := o.state[p].z[0]; z != 9 {
		t.Fatalf("z after step 1: got %v, want 9", z)
	}
	setGrad(p, []float32{4})
	o.Step(params)
	expectData(t, p, []float32{8.25})
	if z := o.state[p].z[0]; z != 8 {
		t.Fatalf("z after step 2: got %v, want 8", z)
	}
	// Eval converts y to the averaged sequence x: with a constant LR the
	// average weights are uniform, so x is the running mean of the z
	// history — (9 + 8)/2 = 8.5 exactly. This is the x-semantics pin,
	// hand-derived.
	o.Eval(params)
	expectData(t, p, []float32{8.5})
}

// --- Reference-formula comparison (float64, tolerance + discrimination) ---

// scheduleFreeReference holds a float64 simulation of the official
// facebookresearch/schedule_free AdamWScheduleFree update (current, v1.3+
// bias correction), independent of the implementation under test. d holds
// the y sequence; x is never materialized, exactly as in the official
// in-place form. tamper selects deliberate mistranscriptions for the
// discrimination gate.
type scheduleFreeReference struct {
	z, v             []float64
	k                int
	weightSum, lrMax float64
}

// scheduleFreeRefStep applies one reference update. tamper: 0 = faithful;
// 1 = y-lerp omitted (c effectively 0); 2 = weight decay applied at z
// instead of y; 3 = average weight off by one (c = 1/(k+2)).
func scheduleFreeRefStep(d []float64, st *scheduleFreeReference, g []float32, lr, b1, b2, eps, wd float32, warmup int, tamper int) {
	fb1, fb2 := float64(b1), float64(b2)
	sched := 1.0
	if st.k < warmup {
		sched = float64(st.k+1) / float64(warmup)
	}
	bc2 := 1 - math.Pow(fb2, float64(st.k+1))
	flr := float64(lr) * sched
	if flr > st.lrMax {
		st.lrMax = flr
	}
	w := st.lrMax * st.lrMax
	st.weightSum += w
	c := w / st.weightSum
	for i := range d {
		gi := float64(g[i])
		st.v[i] = fb2*st.v[i] + (1-fb2)*gi*gi
		gn := gi / (math.Sqrt(st.v[i]/bc2) + float64(eps))
		if tamper == 2 {
			gn += float64(wd) * st.z[i] // decay at z: the official applies it at y
		} else {
			gn += float64(wd) * d[i]
		}
		zi := st.z[i]
		switch tamper {
		case 1:
			d[i] += flr * (fb1*(1-0) - 1) * gn
		case 3:
			cc := 1 / float64(st.k+2)
			d[i] = d[i] + cc*(zi-d[i]) + flr*(fb1*(1-cc)-1)*gn
		default:
			d[i] = d[i] + c*(zi-d[i]) + flr*(fb1*(1-c)-1)*gn
		}
		st.z[i] = zi - flr*gn
	}
	st.k++
}

func TestScheduleFreeStepMatchesReferenceFormula(t *testing.T) {
	// 30 steps over two parameters with lr warmup (5 steps, boundary
	// crossed) and non-zero weight decay, against the float64 reference
	// written from the official implementation's equations. The tolerance
	// absorbs float32-vs-float64 rounding; the library defines its own
	// float32 op order.
	const steps, warmup = 30, 5
	lr, b1, b2, eps, wd := float32(0.05), float32(0.9), float32(0.99), float32(1e-8), float32(0.3)
	rng := rand.New(rand.NewSource(11))
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
		o := NewScheduleFreeAdamW(lr, b1, b2, eps)
		o.WeightDecay, o.WarmupSteps = wd, warmup
		d0 := []float64{0.75, -1.5}
		d1 := []float64{-0.5, 2, 0.25}
		r0 := &scheduleFreeReference{z: []float64{0.75, -1.5}, v: make([]float64, 2), lrMax: -1}
		r1 := &scheduleFreeReference{z: []float64{-0.5, 2, 0.25}, v: make([]float64, 3), lrMax: -1}
		for s := 0; s < steps; s++ {
			setGrad(p0, grads[s][0])
			setGrad(p1, grads[s][1])
			o.Step([]*autograd.Variable{p0, p1})
			scheduleFreeRefStep(d0, r0, grads[s][0], lr, b1, b2, eps, wd, warmup, tamper)
			scheduleFreeRefStep(d1, r1, grads[s][1], lr, b1, b2, eps, wd, warmup, tamper)
		}
		// Parameters (y) and the internal z/v buffers alike.
		diff0, diff1 = relDiff(p0.Data.Data, d0), relDiff(p1.Data.Data, d1)
		for i := range d0 {
			for _, pair := range [][2]float64{
				{float64(o.state[p0].z[i]), r0.z[i]}, {float64(o.state[p0].v[i]), r0.v[i]},
			} {
				if d := math.Abs(pair[0]-pair[1]) / math.Max(1, math.Abs(pair[1])); d > diff0 {
					diff0 = d
				}
			}
		}
		for i := range d1 {
			for _, pair := range [][2]float64{
				{float64(o.state[p1].z[i]), r1.z[i]}, {float64(o.state[p1].v[i]), r1.v[i]},
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
		t.Fatalf("faithful run diverges from the official reference: p0 %g, p1 %g (tol %g)", d0, d1, tol)
	}
	t.Logf("faithful run max rel diff vs float64 reference: p0 %g, p1 %g", d0, d1)

	// Discrimination gate: the tolerance must still reject the classic
	// mistranscriptions, each by a wide margin.
	for _, tc := range []struct {
		tamper int
		name   string
	}{
		{1, "y-lerp omitted"},
		{2, "weight decay at z"},
		{3, "average weight off by one"},
	} {
		d0, d1 := run(tc.tamper)
		if worst := math.Max(d0, d1); worst < 1e-2 {
			t.Errorf("%s: tampered reference stays within %g of the implementation — the tolerance gate has no teeth", tc.name, worst)
		} else {
			t.Logf("%s: correctly rejected, max rel diff %g", tc.name, worst)
		}
	}
}

// --- The y/x (train/eval) contract ---

// panicMessage runs f and reports the recovered panic value, if any.
func panicMessage(f func()) (msg string, ok bool) {
	defer func() {
		if r := recover(); r != nil {
			msg, ok = fmt.Sprint(r), true
		}
	}()
	f()
	return "", false
}

func TestScheduleFreeTrainEvalContract(t *testing.T) {
	const b1 = 0.5
	stepN := func(o *ScheduleFreeAdamW, p *autograd.Variable, g float32, n int) {
		for i := 0; i < n; i++ {
			setGrad(p, []float32{g})
			o.Step([]*autograd.Variable{p})
		}
	}

	t.Run("FreshOptimizerStartsInTrainMode", func(t *testing.T) {
		// Unlike the official implementation (train_mode = False until
		// .train()), a fresh optimizer steps immediately: at construction
		// x = y = z, so the initial train() converts nothing.
		p := param([]float32{1}, []float32{2})
		NewScheduleFreeAdamWDefault(0.01).Step([]*autograd.Variable{p})
		if p.Data.Data[0] == 1 {
			t.Error("first Step of a fresh optimizer did not update the parameter")
		}
	})

	t.Run("StepPanicsInEvalMode", func(t *testing.T) {
		o := NewScheduleFreeAdamW(0.05, b1, 0.99, 1e-8)
		p := param([]float32{1}, nil)
		stepN(o, p, 2, 3)
		o.Eval([]*autograd.Variable{p})
		if !panics(func() { o.Step([]*autograd.Variable{p}) }) {
			t.Error("Step in eval mode did not panic: training at x must be caught")
		}
		o.Train([]*autograd.Variable{p})
		stepN(o, p, 2, 1) // works again after Train
	})

	t.Run("FreshEvalBlocksStepUntilTrain", func(t *testing.T) {
		// Eval on a never-stepped optimizer has no state to convert but
		// still flips the mode — matching the official .eval() — so the
		// next Step panics until Train.
		o := NewScheduleFreeAdamWDefault(0.01)
		p := param([]float32{3}, []float32{1})
		o.Eval([]*autograd.Variable{p})
		expectData(t, p, []float32{3}) // nothing to convert: x = y = z
		if !panics(func() { o.Step([]*autograd.Variable{p}) }) {
			t.Error("Step after Eval on a fresh optimizer did not panic")
		}
		o.Train([]*autograd.Variable{p})
		o.Step([]*autograd.Variable{p})
		if p.Data.Data[0] == 3 {
			t.Error("Step after Train did not update the parameter")
		}
	})

	t.Run("EvalTrainIdempotent", func(t *testing.T) {
		o := NewScheduleFreeAdamW(0.05, b1, 0.99, 1e-8)
		p := param([]float32{1}, nil)
		stepN(o, p, 2, 5)
		o.Eval([]*autograd.Variable{p})
		x := append([]float32(nil), p.Data.Data...)
		o.Eval([]*autograd.Variable{p}) // second Eval is a no-op
		expectData(t, p, x)
		o.Train([]*autograd.Variable{p})
		y := append([]float32(nil), p.Data.Data...)
		o.Train([]*autograd.Variable{p}) // second Train is a no-op
		expectData(t, p, y)
	})

	t.Run("UnsteppedParameterUntouched", func(t *testing.T) {
		o := NewScheduleFreeAdamW(0.05, b1, 0.99, 1e-8)
		stepped := param([]float32{1}, nil)
		fresh := param([]float32{7}, nil)
		params := []*autograd.Variable{stepped, fresh}
		stepN(o, stepped, 2, 3)
		o.Eval(params)
		expectData(t, fresh, []float32{7}) // no state: x = y = z, nothing converts
		if _, ok := o.state[fresh]; ok {
			t.Error("Eval fabricated state for a never-stepped parameter")
		}
		o.Train(params)
		expectData(t, fresh, []float32{7})
		if _, ok := o.state[fresh]; ok {
			t.Error("Train fabricated state for a never-stepped parameter")
		}
	})

	t.Run("NilParameterPanics", func(t *testing.T) {
		o := NewScheduleFreeAdamWDefault(0.01)
		if !panics(func() { o.Train([]*autograd.Variable{nil}) }) {
			t.Error("Train(nil) did not panic")
		}
		oe := NewScheduleFreeAdamWDefault(0.01)
		p := param([]float32{1}, []float32{1})
		oe.Step([]*autograd.Variable{p})
		if !panics(func() { oe.Eval([]*autograd.Variable{nil}) }) {
			t.Error("Eval(nil) did not panic")
		}
	})

	t.Run("XIsTheWeightedAverageOfZ", func(t *testing.T) {
		// THE semantic pin of the eval sequence: with a constant LR the
		// average weights are uniform, so after Eval the parameters must
		// equal the running mean of the z sequence (paper eq. 5 with
		// c = 1/(k+1)), computed independently in float64 from the
		// optimizer's own z snapshots.
		o := NewScheduleFreeAdamW(0.05, 0.9, 0.99, 1e-8)
		p0 := param([]float32{0.75, -1.5}, nil)
		p1 := param([]float32{-0.5}, nil)
		params := []*autograd.Variable{p0, p1}
		rng := rand.New(rand.NewSource(3))
		var zHist0, zHist1 [][]float32
		const steps = 12
		for s := 0; s < steps; s++ {
			setGrad(p0, []float32{rng.Float32()*4 - 2, rng.Float32()*4 - 2})
			setGrad(p1, []float32{rng.Float32()*4 - 2})
			o.Step(params)
			zHist0 = append(zHist0, append([]float32(nil), o.state[p0].z...))
			zHist1 = append(zHist1, append([]float32(nil), o.state[p1].z...))
		}
		o.Eval(params)
		mean := func(hist [][]float32, i int) float64 {
			acc := 0.0
			for _, z := range hist {
				acc += float64(z[i])
			}
			return acc / float64(len(hist))
		}
		got := []float64{float64(p0.Data.Data[0]), float64(p0.Data.Data[1]), float64(p1.Data.Data[0])}
		want := []float64{mean(zHist0, 0), mean(zHist0, 1), mean(zHist1, 0)}
		for i := range want {
			if d := math.Abs(got[i] - want[i]); d > 1e-6*math.Max(1, math.Abs(want[i])) {
				t.Errorf("x[%d] = %v, want the uniform average of z %v (diff %g)", i, got[i], want[i], d)
			}
		}
	})

	t.Run("YRoundTripIsWithinRounding", func(t *testing.T) {
		// The y -> x -> y conversion round trip is NOT bit-exact (each
		// conversion rounds), but it must stay within a few ulps: after
		// Eval + Train the parameters are back at y within float32
		// rounding of the pre-Eval y.
		o := NewScheduleFreeAdamW(0.05, 0.9, 0.99, 1e-8)
		p := param([]float32{0.75}, nil)
		params := []*autograd.Variable{p}
		stepN(o, p, 0.5, 8)
		setGrad(p, []float32{-1.25})
		o.Step(params)
		yBefore := append([]float32(nil), p.Data.Data...)
		o.Eval(params)
		o.Train(params)
		for i := range yBefore {
			d := math.Abs(float64(p.Data.Data[i] - yBefore[i]))
			if d > 1e-6*math.Max(1, math.Abs(float64(yBefore[i]))) {
				t.Errorf("y after Eval+Train = %v, want %v within rounding (diff %g)",
					p.Data.Data[i], yBefore[i], d)
			}
		}
	})
}

// --- Subset desynchronization: the per-parameter mode bits ---

func TestScheduleFreeSubsetDesyncPanics(t *testing.T) {
	// The Medium-3 trap: Eval(all) then Train(subset) used to flip one
	// global flag and leave the unconverted parameters silently training
	// at x (red-team measured a 8e-4 relative divergence over 20 steps).
	// Per-parameter mode bits turn it into a panic naming the first
	// unconverted parameter.
	stepN := func(o *ScheduleFreeAdamW, params []*autograd.Variable, n int) {
		for i := 0; i < n; i++ {
			for j, p := range params {
				setGrad(p, []float32{0.1*float32(i+1) + 0.01*float32(j)})
			}
			o.Step(params)
		}
	}
	newPair := func() (*ScheduleFreeAdamW, []*autograd.Variable) {
		return NewScheduleFreeAdamW(0.05, 0.9, 0.99, 1e-8),
			[]*autograd.Variable{param([]float32{0.5}, nil), param([]float32{-0.25}, nil), param([]float32{1.5}, nil)}
	}

	t.Run("TrainSubset", func(t *testing.T) {
		o, params := newPair()
		stepN(o, params, 5)
		o.Eval(params)
		o.Train(params[:1]) // the misuse: only parameter 0 converted back
		msg, ok := panicMessage(func() { o.Step(params) })
		if !ok {
			t.Fatal("Step after Eval(all)+Train(subset) did not panic")
		}
		if !strings.Contains(msg, "parameter 1") || !strings.Contains(msg, "eval mode") {
			t.Errorf("panic %q does not name the first unconverted parameter (index 1)", msg)
		}
		// Recovering is exact: convert the rest, then both optimizers
		// agree bit for bit with a never-desynced reference.
		o.Train(params[1:])
		oc, paramsC := newPair()
		stepN(oc, paramsC, 5)
		oc.Eval(paramsC)
		oc.Train(paramsC)
		for j, p := range params {
			setGrad(p, []float32{0.7})
			setGrad(paramsC[j], []float32{0.7})
		}
		o.Step(params)
		oc.Step(paramsC)
		for j := range params {
			if !sameBits(params[j].Data.Data, paramsC[j].Data.Data) {
				t.Errorf("parameter %d diverged after recover: %v vs %v", j, params[j].Data.Data, paramsC[j].Data.Data)
			}
		}
	})

	t.Run("EvalSubset", func(t *testing.T) {
		// The mirror image: only parameter 1 was converted to x. Step
		// must panic on it even though the other parameters are in train
		// mode — and parameter 1 must be named.
		o, params := newPair()
		stepN(o, params, 5)
		o.Eval(params[1:2])
		msg, ok := panicMessage(func() { o.Step(params) })
		if !ok {
			t.Fatal("Step after Eval(subset) did not panic")
		}
		if !strings.Contains(msg, "parameter 1") {
			t.Errorf("panic %q does not name the eval parameter (index 1)", msg)
		}
	})

	t.Run("UnsteppedParameterFollowsOptimizerLevelFlag", func(t *testing.T) {
		// A parameter introduced after Eval carries no state: it is
		// gated by the optimizer-level flag, so Step panics on it until
		// Train — while already-train-mode stateful parameters are
		// unaffected by that flag.
		o, params := newPair()
		stepN(o, params[:1], 5)
		o.Eval(params[:1])
		o.Train(params[:1]) // back to train everywhere
		o.Eval(params[:1])  // and eval again, this time leaving it there
		late := param([]float32{2}, []float32{1})
		msg, ok := panicMessage(func() { o.Step([]*autograd.Variable{late}) })
		if !ok || !strings.Contains(msg, "parameter 0") || !strings.Contains(msg, "carries no state") {
			t.Errorf("Step on a stateless parameter in eval mode: (%v, %q)", ok, msg)
		}
	})
}

// --- Warmup schedule and weight decay (internals, exact) ---

func TestScheduleFreeWarmupScheduleExact(t *testing.T) {
	// LR = 0.5 and WarmupSteps = 4 keep every scheduled lr and average
	// weight exactly representable: lr ramps 0.125, 0.25, 0.375, 0.5 and
	// the average weights are the squared running maxima.
	o := NewScheduleFreeAdamW(0.5, 0.9, 0.99, 1e-8)
	o.WarmupSteps = 4
	p := param([]float32{1}, []float32{0.5})
	wantLrMax := []float32{0.125, 0.25, 0.375, 0.5, 0.5}
	wantSum := []float64{0.015625, 0.078125, 0.21875, 0.46875, 0.71875}
	for s := 0; s < 5; s++ {
		setGrad(p, []float32{0.5})
		o.Step([]*autograd.Variable{p})
		st := o.state[p]
		if st.lrMax != wantLrMax[s] {
			t.Errorf("step %d: lrMax = %v, want %v", s+1, st.lrMax, wantLrMax[s])
		}
		if st.weightSum != wantSum[s] {
			t.Errorf("step %d: weightSum = %v, want %v", s+1, st.weightSum, wantSum[s])
		}
		if st.k != s+1 {
			t.Errorf("step %d: k = %d", s+1, st.k)
		}
	}
}

func TestScheduleFreeWeightDecayAppliedAtY(t *testing.T) {
	// One step with decoupled weight decay: gn = g/denom + wd*y where y is
	// the value the parameter holds at Step entry. LR = 1, Beta2 = 0.25,
	// g = 0.5 give denom = 0.5 exactly (Eps absorbed), so gn = 1 + 0.25*2
	// = 1.5; c = 1 snaps y to z and both end at 2 - 1.5 = 0.5. Decay at z
	// (the post-update 0.5) would give gn = 1.125 and y = 0.875 instead.
	p := param([]float32{2}, []float32{0.5})
	o := NewScheduleFreeAdamW(1, 0.5, 0.25, 1e-8)
	o.WeightDecay = 0.25
	o.Step([]*autograd.Variable{p})
	expectData(t, p, []float32{0.5})
	if z := o.state[p].z[0]; z != 0.5 {
		t.Fatalf("z after the decay step: got %v, want 0.5", z)
	}
}

// --- Convergence: x is the deployable sequence ---

func TestScheduleFreeConvergesOnLinearFit(t *testing.T) {
	// The selling point of the method: the x sequence converges without
	// any learning-rate annealing. Train in (default) train mode on the
	// same linear-fit data as the other optimizers, then Eval and check
	// the recovered model AND a freshly evaluated loss at x.
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
	o := NewScheduleFreeAdamWDefault(0.05)

	for epoch := 0; epoch < 400; epoch++ {
		pred := autograd.Add(autograd.MatMul(x, wv), bv)
		diff := autograd.Sub(pred, y)
		l := autograd.MeanAll(autograd.Hadamard(diff, diff))
		for _, p := range params {
			p.ZeroGrad()
		}
		l.Backward()
		o.Step(params)
	}
	o.Eval(params)
	w, b := wv.Data.Data[0], bv.Data.Data[0]
	if math.Abs(float64(w-2)) > 5e-3 || math.Abs(float64(b-1)) > 5e-3 {
		t.Errorf("x did not converge: w=%v b=%v", w, b)
	}
	// Fresh graph at x for the loss (params now hold x).
	pred := autograd.Add(autograd.MatMul(x, wv), bv)
	diff := autograd.Sub(pred, y)
	if loss := autograd.MeanAll(autograd.Hadamard(diff, diff)).Value(); loss > 1e-4 {
		t.Errorf("loss at x %v, want < 1e-4", loss)
	}
}

// --- State semantics ---

func TestScheduleFreeStateIsolatedBetweenParameters(t *testing.T) {
	// Warm up parameter A, then introduce B: B's first update under the
	// used optimizer must equal a fresh optimizer's first update.
	used, fresh := NewScheduleFreeAdamWDefault(0.05), NewScheduleFreeAdamWDefault(0.05)
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

func TestScheduleFreeSkippedParameterKeepsFreshState(t *testing.T) {
	// A parameter skipped (Grad nil) while others step must not advance k,
	// pow2, lrMax or weightSum: its first real update matches a brand-new
	// optimizer's first step.
	o := NewScheduleFreeAdamWDefault(0.05)
	skipped := param([]float32{1}, nil)
	other := param([]float32{1}, []float32{2})
	for i := 0; i < 4; i++ {
		o.Step([]*autograd.Variable{skipped, other})
		setGrad(other, []float32{2})
	}
	setGrad(skipped, []float32{2})
	o.Step([]*autograd.Variable{skipped})

	fresh := NewScheduleFreeAdamWDefault(0.05)
	pWant := param([]float32{1}, []float32{2})
	fresh.Step([]*autograd.Variable{pWant})
	if skipped.Data.Data[0] != pWant.Data.Data[0] {
		t.Errorf("skipped parameter's first real update = %v, want fresh %v",
			skipped.Data.Data[0], pWant.Data.Data[0])
	}
	if got := o.state[skipped].k; got != 1 {
		t.Errorf("update count for skipped parameter: got %d, want 1", got)
	}
}

func TestScheduleFreePanicsOnParameterResize(t *testing.T) {
	resize := func(p *autograd.Variable) { p.Data = tensor.FromData([]float32{1, 2, 3}, 3) }

	t.Run("Step", func(t *testing.T) {
		o := NewScheduleFreeAdamWDefault(0.05)
		p := param([]float32{1, 2}, []float32{1, 1})
		o.Step([]*autograd.Variable{p})
		resize(p)
		if !panics(func() { o.Step([]*autograd.Variable{p}) }) {
			t.Error("resizing a parameter between steps did not panic")
		}
	})
	t.Run("Eval", func(t *testing.T) {
		o := NewScheduleFreeAdamWDefault(0.05)
		p := param([]float32{1, 2}, []float32{1, 1})
		o.Step([]*autograd.Variable{p})
		resize(p)
		if !panics(func() { o.Eval([]*autograd.Variable{p}) }) {
			t.Error("resizing a parameter before Eval did not panic")
		}
	})
	t.Run("Train", func(t *testing.T) {
		o := NewScheduleFreeAdamWDefault(0.05)
		p := param([]float32{1, 2}, []float32{1, 1})
		o.Step([]*autograd.Variable{p})
		o.Eval([]*autograd.Variable{p})
		resize(p)
		if !panics(func() { o.Train([]*autograd.Variable{p}) }) {
			t.Error("resizing a parameter before Train did not panic")
		}
	})
}

// --- Constructor validation ---

func TestScheduleFreeConstructorsRejectInvalidHyperparameters(t *testing.T) {
	nan := float32(math.NaN())
	cases := []struct {
		name string
		new  func()
	}{
		{"NewScheduleFreeAdamW(0, ...)", func() { NewScheduleFreeAdamW(0, 0.9, 0.999, 1e-8) }},
		{"NewScheduleFreeAdamW(NaN, ...)", func() { NewScheduleFreeAdamW(nan, 0.9, 0.999, 1e-8) }},
		{"beta1 -0.1", func() { NewScheduleFreeAdamW(0.1, -0.1, 0.999, 1e-8) }},
		{"beta1 0 (Polyak-Ruppert, needs 1/beta1)", func() { NewScheduleFreeAdamW(0.1, 0, 0.999, 1e-8) }},
		{"beta1 1", func() { NewScheduleFreeAdamW(0.1, 1, 0.999, 1e-8) }},
		{"beta1 NaN", func() { NewScheduleFreeAdamW(0.1, nan, 0.999, 1e-8) }},
		{"beta2 -0.1", func() { NewScheduleFreeAdamW(0.1, 0.9, -0.1, 1e-8) }},
		{"beta2 1", func() { NewScheduleFreeAdamW(0.1, 0.9, 1, 1e-8) }},
		{"beta2 NaN", func() { NewScheduleFreeAdamW(0.1, 0.9, nan, 1e-8) }},
		{"eps 0", func() { NewScheduleFreeAdamW(0.1, 0.9, 0.999, 0) }},
		{"eps -1e-8", func() { NewScheduleFreeAdamW(0.1, 0.9, 0.999, -1e-8) }},
		{"eps NaN", func() { NewScheduleFreeAdamW(0.1, 0.9, 0.999, nan) }},
		{"NewScheduleFreeAdamWDefault(-1)", func() { NewScheduleFreeAdamWDefault(-1) }},
	}
	for _, c := range cases {
		if !panics(c.new) {
			t.Errorf("%s did not panic", c.name)
		}
	}
}
