package nn

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"strings"
	"testing"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/tensor"
)

// TestCfCForwardSmoke is the forward-pass regression: the closed-form cell
// must run, produce correctly shaped outputs, and stay finite, including a
// second step from the returned state and a Linear readout.
func TestCfCForwardSmoke(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	cell := NewCfC(2, 4, nil, rng)
	if cell.StateSize() != 4 {
		t.Fatalf("StateSize = %d, want 4", cell.StateSize())
	}
	x := autograd.Var(tensor.Uniform(rng, -1, 1, 3, 2))
	out, h := cell.Step(x, nil, 0.1)
	if out.Data.Rows() != 3 || out.Data.Cols() != 4 {
		t.Fatalf("out shape %v, want [3 4]", out.Data.Shape)
	}
	if h.Data.Rows() != 3 || h.Data.Cols() != 4 {
		t.Fatalf("h shape %v, want [3 4]", h.Data.Shape)
	}
	assertFinite(t, "Step output/state", out, h)

	out2, h2 := cell.Step(x, h, 0.1)
	assertFinite(t, "second step", out2, h2)
	fc := NewLinear(4, 2, rng)
	y := fc.Forward(out2)
	if y.Data.Rows() != 3 || y.Data.Cols() != 2 {
		t.Fatalf("readout shape %v, want [3 2]", y.Data.Shape)
	}
	assertFinite(t, "readout", y)
}

// TestCfCZeroMasksPureLeakClosedForm checks the degenerate case: with every
// sensory and recurrent synapse masked out, num = den = 0 and the closed-form
// update reduces to the pure leaky relaxation
//
//	kappa = softplus(gleak) / (softplus(cm) + eps)
//	A     = softplus(gleak)*vleak / (softplus(gleak) + eps)
//	v_new = h + (A - h) * (1 - exp(-kappa*ts)),
//
// exactly as Step implements it (B = kappa*ts lands on the direct branch for
// this ts). This is the hand-derived closed form of cm*dv/dt = -gleak*(v-l).
func TestCfCZeroMasksPureLeakClosedForm(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const inDim, units = 2, 3
	const ts = 0.5
	cell := NewCfC(inDim, units, RandomSparse(inDim, units, 0, 0, rng), rng)

	h0 := autograd.Var(tensor.FromData([]float32{0.3, -0.2, 0.1}, 1, units))
	x := autograd.Var(tensor.FromData([]float32{0.7, -0.4}, 1, inDim))
	_, hNew := cell.Step(x, h0, ts)

	softplus := func(v float64) float64 { return math.Log1p(math.Exp(v)) }
	for j := 0; j < units; j++ {
		g := softplus(float64(cell.gleak.Data.Data[j]))
		cmv := softplus(float64(cell.cm.Data.Data[j]))
		l := float64(cell.vleak.Data.Data[j])
		kappa := g / (cmv + float64(cell.eps))
		a := g * l / (g + float64(cell.eps))
		v := float64(h0.Data.Data[j])
		v += (a - v) * (1 - math.Exp(-kappa*ts))
		if got := float64(hNew.Data.Data[j]); math.Abs(got-v) > 1e-4 {
			t.Fatalf("unit %d: zero-mask state = %v, closed form %v (|diff| %g)", j, got, v, math.Abs(got-v))
		}
	}
}

// TestCfCDecayFactorExprelStability exercises the exprel stabilization of
// F(B) = 1 - exp(-B) directly: across the full non-negative float32 range,
// including the B -> 0 trap, values must be finite, match the float64
// reference 1 - exp(-B), and increase monotonically.
func TestCfCDecayFactorExprelStability(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	cell := NewCfC(2, 2, nil, rng)

	bs := []float32{0, 1e-30, 1e-12, 1e-8, 1e-6, 1e-4, 3e-3, 9.9e-3, 1.01e-2, 2e-2, 0.5, 5, 100, 1e3, 3e38}
	b := autograd.Var(tensor.FromData(bs, len(bs)))
	f := cell.decayFactor(b)
	assertFinite(t, "decayFactor over B grid", f)

	for i, bv := range bs {
		got := float64(f.Data.Data[i])
		// float64 expm1 is the cancellation-free reference for 1 - exp(-B);
		// the naive 1-exp form would itself cancel to 0 for tiny B.
		want := -math.Expm1(-float64(bv))
		if bv < cfcExprelThreshold {
			// Taylor branch: full float32 relative precision even as B -> 0,
			// where the naive 1-exp(-B) loses everything to cancellation.
			if math.Abs(got-want) > 1e-5*math.Abs(want)+1e-45 {
				t.Fatalf("B=%g: Taylor branch F = %v, want %v (rel diff %g)", bv, got, want, math.Abs(got-want)/math.Abs(want))
			}
			if bv > 0 && got <= 0 {
				t.Fatalf("B=%g: Taylor branch collapsed to %v; the decay factor (and its gradient) must stay alive as B -> 0", bv, got)
			}
		} else {
			if math.Abs(got-want) > 1e-5*math.Max(1, math.Abs(want)) {
				t.Fatalf("B=%g: direct branch F = %v, want %v (|diff| %g)", bv, got, want, math.Abs(got-want))
			}
		}
		if i > 0 && f.Data.Data[i] < f.Data.Data[i-1] {
			t.Fatalf("F not monotone at B=%g: %v < %v", bv, f.Data.Data[i], f.Data.Data[i-1])
		}
	}
	// The limit behaviors explicitly: F(0) == 0 and F(huge) == 1.
	if f.Data.Data[0] != 0 {
		t.Fatalf("F(0) = %v, want exactly 0", f.Data.Data[0])
	}
	if f.Data.Data[len(bs)-1] != 1 {
		t.Fatalf("F(3e38) = %v, want exactly 1", f.Data.Data[len(bs)-1])
	}
}

// TestCfCExprelBoundaryContinuity verifies that the Taylor/direct branch
// switch at B = cfcExprelThreshold introduces no jump. With zero masks the
// decay rate is kappa_j = softplus(gleak_j)/(softplus(cm_j)+eps) per neuron,
// so ts* = threshold/kappa_0 puts neuron 0 exactly on the boundary; stepping
// just below and just above ts* must move the state by far less than the ts
// perturbation scale. The log-spaced sweep additionally requires every state
// element to evolve monotonically in ts (it is a convex interpolation between
// h0 and A), which any branch discontinuity would break.
func TestCfCExprelBoundaryContinuity(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	const inDim, units = 2, 3
	cell := NewCfC(inDim, units, RandomSparse(inDim, units, 0, 0, rng), rng)
	h0 := autograd.Var(tensor.FromData([]float32{0.3, -0.2, 0.1}, 1, units))
	x := autograd.Var(tensor.FromData([]float32{0.7, -0.4}, 1, inDim))

	softplus := func(v float64) float64 { return math.Log1p(math.Exp(v)) }
	kappa0 := softplus(float64(cell.gleak.Data.Data[0])) /
		(softplus(float64(cell.cm.Data.Data[0])) + float64(cell.eps))
	tsStar := cfcExprelThreshold / kappa0

	_, below := cell.Step(x, h0, tsStar*(1-1e-4))
	_, above := cell.Step(x, h0, tsStar*(1+1e-4))
	for i := range below.Data.Data {
		if d := math.Abs(float64(above.Data.Data[i] - below.Data.Data[i])); d > 1e-4 {
			t.Fatalf("branch boundary jump at element %d: |diff| = %g across ts = %g*(1+-1e-4)", i, d, tsStar)
		}
	}

	// Log-spaced sweep straddling the threshold region: v_new(ts) is a convex
	// interpolation between h0 and A, so every state element must evolve
	// monotonically in ts; a branch discontinuity would break monotonicity.
	tsVals := []float64{3e-4, 6e-4, 1.2e-3, 2.4e-3, 4.8e-3, 9.6e-3, 1.92e-2, 3.84e-2, 7.68e-2}
	trace := make([][]float32, len(tsVals))
	for step, ts := range tsVals {
		_, h := cell.Step(x, h0, ts)
		assertFinite(t, fmt.Sprintf("sweep ts=%g", ts), h)
		trace[step] = append([]float32(nil), h.Data.Data...)
	}
	for j := 0; j < units; j++ {
		first, last := trace[0][j], trace[len(tsVals)-1][j]
		dir := float32(1)
		if last < first {
			dir = -1
		}
		if last == first {
			continue // element happens not to move; nothing to check
		}
		for step := 1; step < len(tsVals); step++ {
			if delta := (trace[step][j] - trace[step-1][j]) * dir; delta < -1e-6 {
				t.Fatalf("unit %d non-monotone in ts at step %d: delta %g against direction %g (ts sequence crosses the exprel branch)", j, step, delta, dir)
			}
		}
	}
}

// TestCfCStepTinyTsFixedPoint checks the dt -> 0 semantics of the closed-form
// solution: an infinitesimal time span must leave the state (essentially)
// where it was, and extreme spans on both ends must stay finite.
func TestCfCStepTinyTsFixedPoint(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	cell := NewCfC(2, 4, nil, rng)
	x := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 2))
	h0 := autograd.Var(tensor.FromData([]float32{0.3, -0.2, 0.1, 0.5, -0.1, 0.4, 0.2, -0.3}, 2, 4))

	out, h := cell.Step(x, h0, 1e-40)
	assertFinite(t, "ts=1e-40", out, h)
	for i := range h.Data.Data {
		if d := math.Abs(float64(h.Data.Data[i] - h0.Data.Data[i])); d > 1e-6 {
			t.Fatalf("ts=1e-40 moved element %d by %g; closed-form dt->0 must fix the state", i, d)
		}
	}
	out, h = cell.Step(x, nil, 1e-300)
	assertFinite(t, "ts=1e-300 from zero state", out, h)
	out, h = cell.Step(x, h, 1e300)
	assertFinite(t, "ts=1e300", out, h)
	out, h = cell.Step(x, h, 0.1)
	assertFinite(t, "ts=0.1", out, h)
}

// TestCfCStepRejectsBadTs: NaN (which slips past a naive `ts <= 0` check),
// zero, negative and both infinities must panic, with the ts value in the
// message, while sane spans stay unaffected.
func TestCfCStepRejectsBadTs(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	cell := NewCfC(2, 4, nil, rng)
	x := autograd.Var(tensor.New(2, 2))
	for _, ts := range []float64{math.NaN(), 0, -0.1, math.Inf(-1), math.Inf(1)} {
		ts := ts
		t.Run(fmt.Sprint(ts), func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("Step with ts=%v did not panic", ts)
				}
				if msg := fmt.Sprint(r); !strings.Contains(msg, fmt.Sprint(ts)) {
					t.Fatalf("panic message %q should carry the ts value %v", msg, ts)
				}
			}()
			cell.Step(x, nil, ts)
		})
	}
	out, h := cell.Step(x, nil, 1e-40)
	assertFinite(t, "ts=1e-40 under validation", out, h)
	out, h = cell.Step(x, nil, 0.1)
	assertFinite(t, "ts=0.1 under validation", out, h)
}

// TestCfCDeterministicSameSeed: two cells built from the same seed and fed
// identical inputs must produce bitwise-identical outputs.
func TestCfCDeterministicSameSeed(t *testing.T) {
	build := func() (*CfC, *autograd.Variable) {
		rng := rand.New(rand.NewSource(23))
		cell := NewCfC(2, 4, RandomSparse(2, 4, 0.7, 0.5, rng), rng)
		x := autograd.Var(tensor.Uniform(rng, -1, 1, 3, 2))
		return cell, x
	}
	c1, x1 := build()
	c2, x2 := build()

	out1, h1 := c1.Step(x1, nil, 0.1)
	out2, h2 := c2.Step(x2, nil, 0.1)
	for i := range out1.Data.Data {
		if out1.Data.Data[i] != out2.Data.Data[i] || h1.Data.Data[i] != h2.Data.Data[i] {
			t.Fatalf("output differs at %d under identical seeds: %v vs %v", i, out1.Data.Data[i], out2.Data.Data[i])
		}
	}
}

// TestCfCGradientsFiniteAcrossUnroll runs BPTT over a multi-step sequence and
// requires every parameter gradient to exist and be finite.
func TestCfCGradientsFiniteAcrossUnroll(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	cell := NewCfC(2, 4, nil, rng) // fully connected wiring
	readout := NewLinear(4, 1, rng)
	params := ParametersOf(cell, readout)

	const steps = 5
	xs := make([]*autograd.Variable, steps)
	for i := range xs {
		xs[i] = autograd.Var(tensor.Uniform(rng, -1, 1, 3, 2))
	}
	ys, hN := Unroll(cell, xs, nil, 0.2)
	if hN == nil || len(ys) != steps {
		t.Fatalf("Unroll returned %d outputs, want %d", len(ys), steps)
	}
	var acc *autograd.Variable
	for i, y := range ys {
		target := autograd.Var(tensor.Uniform(rng, -1, 1, 3, 1))
		diff := autograd.Sub(readout.Forward(y), target)
		sq := autograd.Hadamard(diff, diff)
		if i == 0 {
			acc = sq
		} else {
			acc = autograd.Add(acc, sq)
		}
	}
	loss := autograd.MeanAll(acc)
	for _, p := range params {
		p.ZeroGrad()
	}
	loss.Backward()

	for i, p := range params {
		if p.Grad == nil {
			t.Fatalf("parameter %d has nil gradient after Backward", i)
		}
		assertFinite(t, "parameter gradient", autograd.Var(p.Grad))
	}
	assertFinite(t, "loss", loss)
}

// TestCfCGradcheckAllParameters finite-difference-checks every trainable
// parameter over a 2-step Unroll. float32 central differences carry ~eps/2h
// absolute roundoff, so the relative error uses a 1e-2 denominator floor:
// gradients above the floor get a true relative check, gradients below it an
// absolute ~1e-4 check. The measured maxima are logged.
func TestCfCGradcheckAllParameters(t *testing.T) {
	rng := rand.New(rand.NewSource(41))
	cell := NewCfC(2, 4, RandomSparse(2, 4, 0.9, 0.8, rng), rng)
	readout := NewLinear(4, 1, rng)
	params := ParametersOf(cell, readout)

	const steps = 2
	xs := make([]*autograd.Variable, steps)
	for i := range xs {
		xs[i] = autograd.Var(tensor.Uniform(rng, -1, 1, 3, 2))
	}
	targets := make([]*autograd.Variable, steps)
	for i := range targets {
		targets[i] = autograd.Var(tensor.Uniform(rng, -1, 1, 3, 1))
	}
	lossFn := func() *autograd.Variable {
		ys, _ := Unroll(cell, xs, nil, 0.3)
		var acc *autograd.Variable
		for i, y := range ys {
			diff := autograd.Sub(readout.Forward(y), targets[i])
			sq := autograd.Hadamard(diff, diff)
			if i == 0 {
				acc = sq
			} else {
				acc = autograd.Add(acc, sq)
			}
		}
		return autograd.MeanAll(acc)
	}

	loss := lossFn()
	assertFinite(t, "gradcheck loss", loss)
	for _, p := range params {
		p.ZeroGrad()
	}
	loss.Backward()

	const h = 1e-3
	const floor = 1e-2
	maxRel, maxAbs := 0.0, 0.0
	for pi, p := range params {
		if p.Grad == nil {
			t.Fatalf("parameter %d has nil gradient", pi)
		}
		for i := range p.Data.Data {
			orig := p.Data.Data[i]
			p.Data.Data[i] = orig + h
			l1 := lossFn().Value()
			p.Data.Data[i] = orig - h
			l2 := lossFn().Value()
			p.Data.Data[i] = orig
			num := float64(l1-l2) / (2 * h)
			ana := float64(p.Grad.Data[i])
			d := math.Abs(num - ana)
			if d > maxAbs {
				maxAbs = d
			}
			rel := d / math.Max(math.Max(math.Abs(num), math.Abs(ana)), floor)
			if rel > maxRel {
				maxRel = rel
			}
		}
	}
	t.Logf("gradcheck over %d parameter elements: max rel err %.3e (denom floor %g), max abs err %.3e",
		paramCount(params), maxRel, floor, maxAbs)
	if maxRel > 1e-2 {
		t.Fatalf("gradcheck max relative error %.3e exceeds 1e-2", maxRel)
	}
}

func paramCount(params []*autograd.Variable) int {
	n := 0
	for _, p := range params {
		n += len(p.Data.Data)
	}
	return n
}

// TestCfCParametersExcludeErev checks that reversal potentials are fixed
// +/-1 constants absent from the trainable parameter set. Since the #10 fix
// they are plain *tensor.Tensor fields (see TestCfCReversalPotentialsCarryNoGradient),
// so leaking into the []*autograd.Variable Parameters() is a type-level
// impossibility; the count and the +/-1 value assertions stay as before.
func TestCfCParametersExcludeErev(t *testing.T) {
	rng := rand.New(rand.NewSource(19))
	cell := NewCfC(2, 4, nil, rng)

	params := cell.Parameters()
	if len(params) != 13 {
		t.Fatalf("Parameters() has %d entries, want 13 (erev/sErev excluded)", len(params))
	}
	for name, e := range map[string]*tensor.Tensor{"erev": cell.erev, "sErev": cell.sErev} {
		for _, v := range e.Data {
			if v != 1 && v != -1 {
				t.Fatalf("%s entry %v is not +/-1", name, v)
			}
		}
	}
}

// TestCfCReversalPotentialsCarryNoGradient pins archived finding #10: the
// reversal potentials used to enter the graph as Var leaves and accumulated
// dead gradients that nothing ever read (red team measured
// max|dL/dsErev| ~ 9e-3). They are now plain *tensor.Tensor fields baked
// into the numReduce indicators, so a backward pass over a multi-step Unroll
// can leave no gradient on them — verified structurally by reading the
// fields via reflection: they are the gradient-free data type, and the 13
// trainable parameters still receive finite gradients.
func TestCfCReversalPotentialsCarryNoGradient(t *testing.T) {
	rng := rand.New(rand.NewSource(61))
	cell := NewCfC(2, 4, nil, rng)
	readout := NewLinear(4, 1, rng)
	params := ParametersOf(cell, readout)

	xs := make([]*autograd.Variable, 3)
	for i := range xs {
		xs[i] = autograd.Var(tensor.Uniform(rng, -1, 1, 3, 2))
	}
	ys, _ := Unroll(cell, xs, nil, 0.2)
	var acc *autograd.Variable
	for i, y := range ys {
		target := autograd.Var(tensor.Uniform(rng, -1, 1, 3, 1))
		diff := autograd.Sub(readout.Forward(y), target)
		sq := autograd.Hadamard(diff, diff)
		if i == 0 {
			acc = sq
		} else {
			acc = autograd.Add(acc, sq)
		}
	}
	for _, p := range params {
		p.ZeroGrad()
	}
	autograd.MeanAll(acc).Backward()

	// The direct evidence: the reversal-potential fields are plain tensors.
	// *tensor.Tensor has no Grad field, so gradient accumulation on them is
	// not merely zero but structurally impossible.
	rv := reflect.ValueOf(cell).Elem()
	tensorType := reflect.TypeOf((*tensor.Tensor)(nil))
	for _, name := range []string{"erev", "sErev"} {
		f := rv.FieldByName(name)
		if !f.IsValid() {
			t.Fatalf("cell has no %s field", name)
		}
		if f.Type() != tensorType {
			t.Fatalf("%s field is %v, want *tensor.Tensor (gradient-free data); a graph node here would revive the #10 dead gradient", name, f.Type())
		}
	}

	// The bake removed dead weight only: every trainable parameter still
	// receives a finite gradient.
	for i, p := range params {
		if p.Grad == nil {
			t.Fatalf("parameter %d has nil gradient after Backward", i)
		}
		assertFinite(t, "parameter gradient", autograd.Var(p.Grad))
	}
}

// legacyCfCDrive replicates the pre-#10 drive(): the reversal potentials
// enter the graph as Var leaves and the presynaptic axis is contracted by an
// Add chain of masked outer products. It is the white-box oracle for the
// baked-indicator contraction the current drive() runs.
func legacyCfCDrive(
	pre, mu, sigma, w, erev *autograd.Variable,
	maskRow func(i int) *tensor.Tensor,
) (num, den *autograd.Variable) {
	n := pre.Data.Cols()
	for i := 0; i < n; i++ {
		muR := autograd.SliceRow(mu, i)
		sigR := autograd.SliceRow(sigma, i)
		wR := autograd.SliceRow(w, i)
		erevR := autograd.SliceRow(erev, i)
		preCol := autograd.Col(pre, i)
		act := autograd.Sigmoid(autograd.Hadamard(sigR, autograd.Sub(preCol, muR)))
		act = autograd.Hadamard(act, wR)
		act = autograd.Hadamard(act, autograd.Const(maskRow(i)))
		rev := autograd.Hadamard(act, erevR)
		if i == 0 {
			num, den = rev, act
		} else {
			num = autograd.Add(num, rev)
			den = autograd.Add(den, act)
		}
	}
	return num, den
}

// TestCfCDriveBakeMatchesLegacyBitExact proves the baked-indicator
// contraction is not merely equivalent but bit-identical to the old
// Add-of-Hadamards drive: same forward bits for num and den on both paths
// (sparse wiring included, so masked synapses are exercised), and same
// backward bits for every parameter gradient. The oracle rebuilds erev as a
// Var leaf, which simultaneously reproduces the old design's dead gradient —
// asserted non-nil so the oracle is pinned as the genuine legacy code path.
func TestCfCDriveBakeMatchesLegacyBitExact(t *testing.T) {
	rng := rand.New(rand.NewSource(57))
	cell := NewCfC(3, 5, RandomSparse(3, 5, 0.8, 0.7, rng), rng)
	x := autograd.Var(tensor.Uniform(rng, -1, 1, 4, 3))
	h := autograd.Var(tensor.Uniform(rng, -1, 1, 4, 5))

	inputs := autograd.Add(autograd.Hadamard(x, cell.inW), cell.inB)
	sWPos := autograd.Softplus(cell.sW)
	wPos := autograd.Softplus(cell.w)

	// Baked-indicator path (the current drive).
	numS, denS := cell.drive(inputs, cell.sMu, cell.sSigma, sWPos, cell.denReduceS, cell.numReduceS, cell.wiring.SensoryRow)
	numR, denR := cell.drive(h, cell.mu, cell.sigma, wPos, cell.denReduceR, cell.numReduceR, cell.wiring.RecurrentRow)

	// Legacy Add-chain path, with erev re-wrapped as Var leaves (sharing the
	// same data, which the forward pass only reads).
	erevVar := autograd.Var(cell.erev)
	sErevVar := autograd.Var(cell.sErev)
	lNumS, lDenS := legacyCfCDrive(inputs, cell.sMu, cell.sSigma, sWPos, sErevVar, cell.wiring.SensoryRow)
	lNumR, lDenR := legacyCfCDrive(h, cell.mu, cell.sigma, wPos, erevVar, cell.wiring.RecurrentRow)

	for name, pair := range map[string][2]*autograd.Variable{
		"numS": {numS, lNumS}, "denS": {denS, lDenS},
		"numR": {numR, lNumR}, "denR": {denR, lDenR},
	} {
		if !sameBitsT(pair[0].Data, pair[1].Data) {
			t.Fatalf("%s: baked-indicator contraction differs from the legacy Add chain", name)
		}
	}

	// Backward through equal-valued roots: every parameter gradient must
	// agree bit for bit between the two contractions.
	newLoss := autograd.MeanAll(autograd.Add(autograd.Add(numS, numR), autograd.Add(denS, denR)))
	legacyLoss := autograd.MeanAll(autograd.Add(autograd.Add(lNumS, lNumR), autograd.Add(lDenS, lDenR)))
	params := cell.Parameters()
	for _, p := range params {
		p.ZeroGrad()
	}
	newLoss.Backward()
	newGrads := make([]*tensor.Tensor, len(params))
	for i, p := range params {
		if p.Grad != nil {
			newGrads[i] = p.Grad.Clone()
		}
	}
	for _, p := range params {
		p.ZeroGrad()
	}
	legacyLoss.Backward()
	// gleak/vleak/cm/outW/outB feed the membrane algebra downstream of drive,
	// not drive itself, so they stay gradient-free under both losses; every
	// parameter that does participate must agree bit for bit.
	touched := 0
	for i, p := range params {
		if newGrads[i] == nil && p.Grad == nil {
			continue
		}
		touched++
		if newGrads[i] == nil || p.Grad == nil || !sameBitsT(newGrads[i], p.Grad) {
			t.Fatalf("parameter %d gradient differs between baked and legacy contractions", i)
		}
	}
	if touched != 8 { // inW, inB (via inputs) + sMu, sSigma, sW, mu, sigma, w
		t.Fatalf("drive touched %d parameters, want 8; the oracle graph is not the drive subgraph", touched)
	}

	// The oracle genuinely is the legacy design: its erev leaves accumulated
	// the dead gradients that #10 removes (the red team measured ~9e-3 here).
	if erevVar.Grad == nil || sErevVar.Grad == nil {
		t.Fatal("legacy oracle did not accumulate the dead erev gradient; the oracle is not the old code path")
	}
	var maxDead float32
	for _, g := range []*tensor.Tensor{erevVar.Grad, sErevVar.Grad} {
		for _, v := range g.Data {
			if a := float32(math.Abs(float64(v))); a > maxDead {
				maxDead = a
			}
		}
	}
	if maxDead == 0 {
		t.Fatal("legacy dead gradient is identically zero; the test would prove nothing")
	}
	t.Logf("legacy dead gradient max|dL/derev| = %.3e (now structurally impossible: erev is plain data)", maxDead)
}

// sameBitsT reports whether a and b have identical shapes and bit-identical
// float32 payloads (named apart from save_test.go's saveSameBits to keep
// this oracle self-describing).
func sameBitsT(a, b *tensor.Tensor) bool {
	if len(a.Shape) != len(b.Shape) || len(a.Data) != len(b.Data) {
		return false
	}
	for i := range a.Shape {
		if a.Shape[i] != b.Shape[i] {
			return false
		}
	}
	for i := range a.Data {
		if math.Float32bits(a.Data[i]) != math.Float32bits(b.Data[i]) {
			return false
		}
	}
	return true
}

// TestNewCfCValidation covers constructor argument checks.
func TestNewCfCValidation(t *testing.T) {
	rng := rand.New(rand.NewSource(29))
	cases := []func(){
		func() { NewCfC(0, 4, nil, rng) },
		func() { NewCfC(2, 0, nil, rng) },
		func() { NewCfC(2, 4, FullyConnected(3, 4), rng) }, // sensory shape mismatch
		func() { NewCfC(2, 4, FullyConnected(2, 5), rng) }, // recurrent shape mismatch
	}
	for i, f := range cases {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatalf("case %d did not panic", i)
				}
			}()
			f()
		}()
	}
}
