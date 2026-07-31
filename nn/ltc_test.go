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

func assertFinite(t *testing.T, name string, vs ...*autograd.Variable) {
	t.Helper()
	for _, v := range vs {
		for i, x := range v.Data.Data {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				t.Fatalf("%s: non-finite value %v at element %d", name, x, i)
			}
		}
	}
}

// TestLTCForwardSmoke is the regression for V-01: the LTC forward pass must
// actually run (it never did before the synapses() convention was unified),
// produce correctly shaped outputs, and stay finite.
func TestLTCForwardSmoke(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	cell := NewLTC(2, 4, nil, 6, rng)
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

	// Step again from the returned state, then through a Linear readout.
	out2, h2 := cell.Step(x, h, 0.1)
	assertFinite(t, "second step", out2, h2)
	fc := NewLinear(4, 2, rng)
	y := fc.Forward(out2)
	if y.Data.Rows() != 3 || y.Data.Cols() != 2 {
		t.Fatalf("readout shape %v, want [3 2]", y.Data.Shape)
	}
	assertFinite(t, "readout", y)
}

// TestLTCZeroMasksLeakyIntegrator checks the closed-form degenerate case:
// with every sensory and recurrent synapse masked out, each ODE unfold is
// the pure leaky update
//
//	v_{k+1} = (a*v_k + b*l) / (a + b + eps) = r*v_k + f,
//	a = softplus(cm)*unfolds/ts, b = softplus(gleak), l = vleak,
//
// exactly as implemented by the semi-implicit Euler in Step.
func TestLTCZeroMasksLeakyIntegrator(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const inDim, units, unfolds = 2, 3, 4
	const ts = 0.5
	cell := NewLTC(inDim, units, RandomSparse(inDim, units, 0, 0, rng), unfolds, rng)

	h0 := autograd.Var(tensor.FromData([]float32{0.3, -0.2, 0.1}, 1, units))
	x := autograd.Var(tensor.FromData([]float32{0.7, -0.4}, 1, inDim))
	_, hNew := cell.Step(x, h0, ts)

	softplus := func(v float64) float64 { return math.Log1p(math.Exp(v)) }
	scale := float64(unfolds) / ts
	for j := 0; j < units; j++ {
		a := softplus(float64(cell.cm.Data.Data[j])) * scale
		b := softplus(float64(cell.gleak.Data.Data[j]))
		l := float64(cell.vleak.Data.Data[j])
		r := a / (a + b + float64(cell.eps))
		f := b * l / (a + b + float64(cell.eps))
		v := float64(h0.Data.Data[j])
		for k := 0; k < unfolds; k++ {
			v = r*v + f
		}
		if got := float64(hNew.Data.Data[j]); math.Abs(got-v) > 1e-4 {
			t.Fatalf("unit %d: zero-mask state = %v, closed form %v (|diff| %g)", j, got, v, math.Abs(got-v))
		}
	}
}

// TestLTCGradientsFiniteAcrossUnfolds runs BPTT over a multi-step sequence
// and requires every parameter gradient to exist and be finite.
func TestLTCGradientsFiniteAcrossUnfolds(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	cell := NewLTC(2, 4, nil, 4, rng) // fully connected wiring
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
		diff := autograd.Sub(readout.Forward(y), target) // [batch, 1]
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

// TestLTCStepTinyTsStaysFinite is the V-02 regression: ts = 1e-40 must not
// overflow cm/dt into NaN/Inf outputs.
func TestLTCStepTinyTsStaysFinite(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	cell := NewLTC(2, 4, nil, 6, rng)
	x := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 2))

	out, h := cell.Step(x, nil, 1e-40)
	assertFinite(t, "ts=1e-40 from zero state", out, h)

	// Again from a nonzero state, and with an even more extreme ts.
	out, h = cell.Step(x, h, 1e-40)
	assertFinite(t, "ts=1e-40 continued", out, h)
	out, h = cell.Step(x, h, 1e-300)
	assertFinite(t, "ts=1e-300", out, h)
	// A huge ts (scale -> 0) must stay finite as well.
	out, h = cell.Step(x, h, 1e300)
	assertFinite(t, "ts=1e300", out, h)
}

// TestLTCStepRejectsBadTs is the V-03 regression: NaN (which slipped past
// the old `ts <= 0` check), zero, negative and -Inf ts must panic.
func TestLTCStepRejectsBadTs(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	cell := NewLTC(2, 4, nil, 6, rng)
	x := autograd.Var(tensor.New(2, 2))
	for _, ts := range []float64{math.NaN(), 0, -0.1, math.Inf(-1)} {
		ts := ts
		t.Run("", func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatalf("Step with ts=%v did not panic", ts)
				}
			}()
			cell.Step(x, nil, ts)
		})
	}
}

// TestLTCStepRejectsInfTs is the F3 cleanup: +Inf passed the old positivity
// check (Inf > 0 is true) and silently integrated over an infinite time span
// (the "infinite-time steady state"), which callers rarely intend. Both
// infinities must panic now, on the same path as NaN/0/negative ts, and the
// panic message must carry the offending ts value.
func TestLTCStepRejectsInfTs(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	cell := NewLTC(2, 4, nil, 6, rng)
	x := autograd.Var(tensor.New(2, 2))
	for _, ts := range []float64{math.Inf(1), math.Inf(-1)} {
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
	// Sane ts values, including the tiny-ts finiteness regime, are unaffected.
	out, h := cell.Step(x, nil, 1e-40)
	assertFinite(t, "ts=1e-40 under new validation", out, h)
	out, h = cell.Step(x, nil, 0.1)
	assertFinite(t, "ts=0.1 under new validation", out, h)
}

// TestLTCParametersExcludeErev checks that reversal potentials are fixed
// +/-1 constants absent from the trainable parameter set.
func TestLTCParametersExcludeErev(t *testing.T) {
	rng := rand.New(rand.NewSource(19))
	cell := NewLTC(2, 4, nil, 6, rng)

	params := cell.Parameters()
	if len(params) != 13 {
		t.Fatalf("Parameters() has %d entries, want 13 (erev/sErev excluded)", len(params))
	}
	for _, p := range params {
		if p == cell.erev || p == cell.sErev {
			t.Fatalf("reversal potential %p leaked into Parameters()", p)
		}
	}
	for name, e := range map[string]*autograd.Variable{"erev": cell.erev, "sErev": cell.sErev} {
		for _, v := range e.Data.Data {
			if v != 1 && v != -1 {
				t.Fatalf("%s entry %v is not +/-1", name, v)
			}
		}
	}
}

// TestLTCDeterministicSameSeed: two cells built from the same seed and fed
// identical inputs must produce bitwise-identical outputs.
func TestLTCDeterministicSameSeed(t *testing.T) {
	build := func() (*LTC, *autograd.Variable) {
		rng := rand.New(rand.NewSource(23))
		cell := NewLTC(2, 4, RandomSparse(2, 4, 0.7, 0.5, rng), 4, rng)
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

// scalarSynapses reproduces the pre-vectorization per-synapse-pair loop as a
// reference oracle: row i of each parameter matrix is sliced inside the loop,
// the mask applied per synapse via maskRow(i), and the currents accumulated
// with one Add per presynaptic neuron. Kept verbatim from the old
// implementation to pin the vectorized synapses() to identical numerics.
func scalarSynapses(
	pre, mu, sigma, w, erev *autograd.Variable,
	maskRow func(i int) *tensor.Tensor,
) (num, den *autograd.Variable) {
	n := pre.Data.Cols()
	for i := 0; i < n; i++ {
		muR := autograd.SliceRow(mu, i)
		sigR := autograd.SliceRow(sigma, i)
		wR := autograd.SliceRow(w, i)
		erevR := autograd.SliceRow(erev, i)
		preCol := autograd.Col(pre, i) // [batch, 1]
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

// TestLTCSynapsesVectorizedEquivalence checks the vectorized synapses()
// against the old per-synapse loop element by element, on both the recurrent
// and the sensory path and under a sparse wiring mask. The new path must
// reproduce the old currents exactly — Go's == treats +0 and -0 as equal, so
// only genuine rounding differences can fail the test. The sparse
// contraction changes the graph structure, not the floating-point
// accumulation order: the fold runs left-to-right over presynaptic neurons
// exactly like the old Add chain, and mask ∈ {0, 1} makes w·mask exactly w
// or +0.
func TestLTCSynapsesVectorizedEquivalence(t *testing.T) {
	rng := rand.New(rand.NewSource(41))
	const inDim, units = 3, 5
	cell := NewLTC(inDim, units, RandomSparse(inDim, units, 0.6, 0.6, rng), 2, rng)
	preR := autograd.Var(tensor.Uniform(rng, -1, 1, 4, units))
	preS := autograd.Var(tensor.Uniform(rng, -1, 1, 4, inDim))

	wM := autograd.Hadamard(autograd.Softplus(cell.w), cell.maskR)
	sWm := autograd.Hadamard(autograd.Softplus(cell.sW), cell.maskS)
	newRNum, newRDen := cell.synapses(preR, cell.mu, cell.sigma, wM, cell.erevRowsR)
	oldRNum, oldRDen := scalarSynapses(preR, cell.mu, cell.sigma,
		autograd.Softplus(cell.w), cell.erev, cell.wiring.RecurrentRow)
	newSNum, newSDen := cell.synapses(preS, cell.sMu, cell.sSigma, sWm, cell.erevRowsS)
	oldSNum, oldSDen := scalarSynapses(preS, cell.sMu, cell.sSigma,
		autograd.Softplus(cell.sW), cell.sErev, cell.wiring.SensoryRow)

	exact := func(name string, got, want *autograd.Variable) {
		t.Helper()
		if len(got.Data.Data) != len(want.Data.Data) {
			t.Fatalf("%s: size %d, want %d", name, len(got.Data.Data), len(want.Data.Data))
		}
		for i := range got.Data.Data {
			if got.Data.Data[i] != want.Data.Data[i] {
				t.Fatalf("%s: element %d = %v, want exactly %v",
					name, i, got.Data.Data[i], want.Data.Data[i])
			}
		}
	}
	exact("recurrent num", newRNum, oldRNum)
	exact("recurrent den", newRDen, oldRDen)
	exact("sensory num", newSNum, oldSNum)
	exact("sensory den", newSDen, oldSDen)

	// The single-presynaptic-source path (inDim == 1) skips the den MatMul
	// (identity contraction); it must stay exact as well.
	rng1 := rand.New(rand.NewSource(43))
	cell1 := NewLTC(1, units, RandomSparse(1, units, 0.6, 0.6, rng1), 2, rng1)
	pre1 := autograd.Var(tensor.Uniform(rng1, -1, 1, 4, 1))
	sWm1 := autograd.Hadamard(autograd.Softplus(cell1.sW), cell1.maskS)
	new1Num, new1Den := cell1.synapses(pre1, cell1.sMu, cell1.sSigma, sWm1, cell1.erevRowsS)
	old1Num, old1Den := scalarSynapses(pre1, cell1.sMu, cell1.sSigma,
		autograd.Softplus(cell1.sW), cell1.sErev, cell1.wiring.SensoryRow)
	exact("single-source num", new1Num, old1Num)
	exact("single-source den", new1Den, old1Den)
}

// TestLTCGradCheckAllParameters validates the analytic Backward gradients of
// a whole unrolled LTC training step against central finite differences, for
// every trainable parameter of both the cell and the readout, under a sparse
// wiring mask. The tolerance (2e-2, scaled by the gradient magnitude) is
// generous because float32 differencing is noisy; the test guards the
// mathematical equivalence of the vectorized synapse graph and its
// per-synapse reductions.
func TestLTCGradCheckAllParameters(t *testing.T) {
	rng := rand.New(rand.NewSource(101))
	const inDim, units, unfolds, batch, seqLen = 2, 4, 2, 3, 2
	const h = 1e-3
	cell := NewLTC(inDim, units, RandomSparse(inDim, units, 0.7, 0.7, rng), unfolds, rng)
	readout := NewLinear(units, 1, rng)
	params := ParametersOf(cell, readout)

	xs := make([]*autograd.Variable, seqLen)
	targets := make([]*autograd.Variable, seqLen)
	for i := range xs {
		xs[i] = autograd.Var(tensor.Uniform(rng, -1, 1, batch, inDim))
		targets[i] = autograd.Var(tensor.Uniform(rng, -1, 1, batch, 1))
	}

	buildLoss := func() *autograd.Variable {
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

	// Analytic gradients from one Backward.
	for _, p := range params {
		p.ZeroGrad()
	}
	buildLoss().Backward()

	var maxRel float64
	for pi, p := range params {
		if p.Grad == nil {
			t.Fatalf("parameter %d has nil analytic gradient", pi)
		}
		data := p.Data.Data
		grad := p.Grad.Data
		for k := range data {
			orig := data[k]
			data[k] = orig + h
			lp := float64(buildLoss().Value())
			data[k] = orig - h
			lm := float64(buildLoss().Value())
			data[k] = orig
			num := (lp - lm) / (2 * h)
			an := float64(grad[k])
			scale := math.Max(1, math.Abs(num))
			rel := math.Abs(an-num) / scale
			if rel > maxRel {
				maxRel = rel
			}
			if math.Abs(an-num) > 2e-2*scale {
				t.Fatalf("parameter %d element %d: analytic %v vs numerical %v (|diff| %g)",
					pi, k, an, num, math.Abs(an-num))
			}
		}
	}
	t.Logf("gradcheck max relative error %g (tolerance 2e-2)", maxRel)
}

// TestNewLTCValidation covers constructor argument checks.
func TestNewLTCValidation(t *testing.T) {
	rng := rand.New(rand.NewSource(29))
	cases := []func(){
		func() { NewLTC(0, 4, nil, 4, rng) },
		func() { NewLTC(2, 0, nil, 4, rng) },
		func() { NewLTC(2, 4, nil, 0, rng) },
		func() { NewLTC(2, 4, FullyConnected(3, 4), 4, rng) }, // sensory shape mismatch
		func() { NewLTC(2, 4, FullyConnected(2, 5), 4, rng) }, // recurrent shape mismatch
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

// TestLTCSparseContractionLargeCellMemoryGate pins the item-#14 root-cause
// fix: the constructor must never materialize the former [units^2, units]
// reduction indicators. NewLTC(2, 1024, FullyConnected, 1, ...) used to
// attempt 2*1024^3*4B = 8 GiB of dense indicators (and OOM the caller); the
// sparse contraction's persistent state is (mu/sigma/w + erev + recurrent
// mask + ident) = 6*units^2 float32 plus the wiring plan's units^2 int32,
// i.e. 7*units^2*4B = 28 MiB at units=1024 (plus small sensory terms and
// mask copies, ~32 MiB total). The gate sits at 128 MiB: 4x headroom over
// the derivation, 64x below the old cliff.
func TestLTCSparseContractionLargeCellMemoryGate(t *testing.T) {
	const units = 1024
	build := func() {
		NewLTC(2, units, FullyConnected(2, units), 1, rand.New(rand.NewSource(7)))
	}
	if b := allocBytesPerRun(3, build); b >= 128<<20 {
		t.Fatalf("NewLTC(2, %d, FullyConnected, ...) allocated %d bytes (%.0f MiB); want < 128 MiB (dense indicators would need ~8 GiB)", units, b, float64(b)/(1<<20))
	}

	cell := NewLTC(2, units, FullyConnected(2, units), 1, rand.New(rand.NewSource(7)))
	// Fully-wired plans carry exactly pre*units terms per side.
	if got := cell.planR.terms(); got != units*units {
		t.Errorf("recurrent plan terms = %d, want %d (units^2)", got, units*units)
	}
	if got := cell.planS.terms(); got != 2*units {
		t.Errorf("sensory plan terms = %d, want %d (inDim*units)", got, 2*units)
	}
	// One step must run and stay finite (functionality, not performance).
	x := autograd.Var(tensor.Uniform(rand.New(rand.NewSource(8)), -1, 1, 1, 2))
	out, hNew := cell.Step(x, nil, 0.1)
	assertFinite(t, "large LTC step", out, hNew)
}

// TestLTCSparsePlanCountsMatchWiringMask proves the per-postsynaptic term
// lists are exactly the wiring mask's nonzeros: under RandomSparse(p=0.3)
// the total term count equals the mask nnz, and every column's list holds
// its wired presynaptic indices in ascending order.
func TestLTCSparsePlanCountsMatchWiringMask(t *testing.T) {
	rng := rand.New(rand.NewSource(73))
	const inDim, units = 3, 8
	w := RandomSparse(inDim, units, 0.3, 0.3, rng)
	cell := NewLTC(inDim, units, w, 2, rng)

	nnz := func(m *tensor.Tensor) int {
		n := 0
		for _, v := range m.Data {
			if v == 1 {
				n++
			}
		}
		return n
	}
	colNNZ := func(m *tensor.Tensor, j int) int {
		n := 0
		for i := 0; i < m.Rows(); i++ {
			if m.Data[i*m.Cols()+j] == 1 {
				n++
			}
		}
		return n
	}
	maskR, maskS := w.Recurrent(), w.Sensory()
	if got, want := cell.planR.terms(), nnz(maskR); got != want {
		t.Fatalf("recurrent plan terms = %d, want mask nnz %d", got, want)
	}
	if got, want := cell.planS.terms(), nnz(maskS); got != want {
		t.Fatalf("sensory plan terms = %d, want mask nnz %d", got, want)
	}
	for _, tc := range []struct {
		name string
		plan *synapsePlan
		mask *tensor.Tensor
	}{{"recurrent", cell.planR, maskR}, {"sensory", cell.planS, maskS}} {
		for j := 0; j < units; j++ {
			list := tc.plan.cols[j]
			if len(list) != colNNZ(tc.mask, j) {
				t.Fatalf("%s column %d: %d terms, want %d", tc.name, j, len(list), colNNZ(tc.mask, j))
			}
			for k, i := range list {
				if tc.mask.Data[int(i)*units+j] != 1 {
					t.Fatalf("%s column %d term %d: presynaptic %d is not wired", tc.name, j, k, i)
				}
				if k > 0 && i <= list[k-1] {
					t.Fatalf("%s column %d: presynaptic indices not ascending at %d", tc.name, j, k)
				}
			}
		}
	}
}

// TestLTCSparseContractionFullyMaskedColumnZeroSign pins the F-RT1 corner
// on the sparse contraction: a postsynaptic column whose every synapse is
// masked, with reversal potentials -1, must contract to +0 (0x00000000) —
// not -0 (0x80000000). This is the tensor.MatMul av==0 zero-skip behavior
// the fold replicates (contract constraints 3 and 4): the pre-#14 dense
// MatMul produced +0 here, the even older per-synapse Add chain produced
// -0, and the backward leaves +0 gradients on the masked weights either
// way.
func TestLTCSparseContractionFullyMaskedColumnZeroSign(t *testing.T) {
	const inDim, units = 2, 4
	const maskedCol = 1
	sensory := tensor.New(inDim, units).OnesLike()
	recurrent := tensor.New(units, units).OnesLike()
	for i := 0; i < inDim; i++ {
		sensory.Data[i*units+maskedCol] = 0
	}
	for i := 0; i < units; i++ {
		recurrent.Data[i*units+maskedCol] = 0
	}
	cell := NewLTC(inDim, units, &Wiring{sensoryMask: sensory, recurrentMask: recurrent}, 2, rand.New(rand.NewSource(71)))
	// Force every reversal potential onto the masked column to -1: the
	// maximally adversarial sign for the -0 product. The erevRows views
	// share this storage, so the contraction sees the flip immediately.
	for i := 0; i < units; i++ {
		cell.erev.Data.Data[i*units+maskedCol] = -1
	}

	rng := rand.New(rand.NewSource(72))
	pre := autograd.Var(tensor.Uniform(rng, -1, 1, 3, units))
	wM := autograd.Hadamard(autograd.Softplus(cell.w), cell.maskR)
	num, den := cell.synapses(pre, cell.mu, cell.sigma, wM, cell.erevRowsR)
	for b := 0; b < 3; b++ {
		if bits := math.Float32bits(num.Data.Data[b*units+maskedCol]); bits != 0 {
			t.Fatalf("fully-masked num column batch %d = %#x (-0 would be 0x80000000); want +0", b, bits)
		}
		if bits := math.Float32bits(den.Data.Data[b*units+maskedCol]); bits != 0 {
			t.Fatalf("fully-masked den column batch %d = %#x; want +0", b, bits)
		}
	}
	// Backward: the masked weights' gradients are +0, as under the dense
	// indicator MatMul (F-RT1: "反向零梯度同").
	autograd.SumAll(autograd.Add(num, den)).Backward()
	for i := 0; i < units; i++ {
		if bits := math.Float32bits(cell.w.Grad.Data[i*units+maskedCol]); bits != 0 {
			t.Fatalf("masked w[%d, %d] gradient = %#x; want +0", i, maskedCol, bits)
		}
	}
}
