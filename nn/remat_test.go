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

// This file pins the rematerialized chunked-BPTT contract of UnrollRemat:
// for any cell, sequence and loss, its leaf gradients and outputs must be
// BIT-IDENTICAL (Float32bits — never tolerance) to the reference pipeline
// "Unroll, build the same loss, loss.Backward()", at a fraction of the
// peak graph memory. TestStepDoesNotMutateInputs is the empirical premise
// that makes the zero-copy Detach underneath safe; the differential matrix
// is the acceptance gate; the edge and determinism tests close out the
// contract.

// rematCase is one differential configuration.
type rematCase struct {
	name      string
	cfc       bool // false: LTC
	inDim     int
	units     int
	unfolds   int // LTC only
	batch     int
	T         int
	chunkSize int
	h0        bool // use a non-nil initial state
	ts        float64
	loss      int // rematLoss kind
}

// rematLoss builds the loss closure of the given kind over the readout,
// targets and regularization leaf (any may be unused by a kind). The kinds
// deliberately stress the seed plumbing: full-sequence losses, losses
// blind to most steps (nil seeds), a step consumed twice, and a loss
// closing over a cell parameter.
func rematLoss(kind int, readout *Linear, targets []*autograd.Variable, reg *autograd.Variable, T int) func(ys []*autograd.Variable) *autograd.Variable {
	switch kind {
	case 0: // per-step readout MSE over the whole sequence (the benchmark's loss)
		return func(ys []*autograd.Variable) *autograd.Variable {
			var acc *autograd.Variable
			for t, y := range ys {
				diff := autograd.Sub(readout.Forward(y), targets[t])
				sq := autograd.Hadamard(diff, diff)
				if t == 0 {
					acc = sq
				} else {
					acc = autograd.Add(acc, sq)
				}
			}
			return autograd.Scale(autograd.MeanAll(acc), 1/float32(T))
		}
	case 1: // final step only: every other step keeps a nil seed
		return func(ys []*autograd.Variable) *autograd.Variable {
			y := ys[len(ys)-1]
			return autograd.MeanAll(autograd.Hadamard(y, y))
		}
	case 2: // even steps only, step-weighted
		return func(ys []*autograd.Variable) *autograd.Variable {
			var acc *autograd.Variable
			for i := 0; i < len(ys); i += 2 {
				term := autograd.SumAll(autograd.Scale(ys[i], float32(i+1)))
				if acc == nil {
					acc = term
				} else {
					acc = autograd.Add(acc, term)
				}
			}
			return autograd.Scale(acc, 0.5)
		}
	case 3: // final step plus an L2 regularizer closing over a cell parameter
		return func(ys []*autograd.Variable) *autograd.Variable {
			y := ys[len(ys)-1]
			data := autograd.MeanAll(autograd.Hadamard(y, y))
			pen := autograd.Scale(autograd.SumAll(autograd.Hadamard(reg, reg)), 0.01)
			return autograd.Add(data, pen)
		}
	case 4: // the final output consumed by two loss-graph branches
		return func(ys []*autograd.Variable) *autograd.Variable {
			y := ys[len(ys)-1]
			return autograd.MeanAll(autograd.Add(autograd.Hadamard(y, y), autograd.Scale(y, 3)))
		}
	case 6: // late before early: the loss DFS visits ys[len-1] first, then ys[0]
		return func(ys []*autograd.Variable) *autograd.Variable {
			late := autograd.SumAll(autograd.Scale(ys[len(ys)-1], 1))
			early := autograd.SumAll(autograd.Scale(ys[0], 2))
			return autograd.Add(late, early)
		}
	case 7: // a middle segment only [n/3, 2n/3], visited ascending
		return func(ys []*autograd.Variable) *autograd.Variable {
			lo, hi := len(ys)/3, 2*len(ys)/3
			var acc *autograd.Variable
			for i := lo; i <= hi; i++ {
				term := autograd.SumAll(autograd.Scale(ys[i], float32(i-lo+1)))
				if acc == nil {
					acc = term
				} else {
					acc = autograd.Add(acc, term)
				}
			}
			return autograd.Scale(acc, 0.25)
		}
	case 8: // out-of-order visits: n/2, then n/4, then 3n/4
		return func(ys []*autograd.Variable) *autograd.Variable {
			n := len(ys)
			a := autograd.SumAll(autograd.Scale(ys[n/2], 1))
			b := autograd.SumAll(autograd.Scale(ys[n/4], 2))
			c := autograd.SumAll(autograd.Scale(ys[3*n/4], 3))
			return autograd.Add(autograd.Add(a, b), c)
		}
	case 9: // adversarial: one high visit, then a run of consecutive lower
		// visits — non-record-high seeds that forbid unit boundaries and
		// force a recompute unit longer than chunkSize
		return func(ys []*autograd.Variable) *autograd.Variable {
			n := len(ys)
			loss := autograd.SumAll(autograd.Scale(ys[n-2], 1))
			for i := 2; i <= 5 && i < n; i++ {
				loss = autograd.Add(loss, autograd.SumAll(autograd.Scale(ys[i], float32(i))))
			}
			return loss
		}
	default: // 5: first step only
		return func(ys []*autograd.Variable) *autograd.Variable {
			return autograd.SumAll(ys[0])
		}
	}
}

// rematSetup builds the cell, readout, inputs, targets and initial state
// for a case, deterministically under the seed, plus the aggregated
// parameter list (cell first, then readout — the order gradients are
// compared in).
func rematSetup(tc rematCase, seed int64) (cell Cell, params []*autograd.Variable, readout *Linear, xs, targets []*autograd.Variable, h0 *autograd.Variable) {
	rng := rand.New(rand.NewSource(seed))
	var module Module
	if tc.cfc {
		c := NewCfC(tc.inDim, tc.units, nil, rng)
		cell, module = c, c
	} else {
		c := NewLTC(tc.inDim, tc.units, nil, tc.unfolds, rng)
		cell, module = c, c
	}
	readout = NewLinear(tc.units, 1, rng)
	params = ParametersOf(module, readout)
	xs = make([]*autograd.Variable, tc.T)
	targets = make([]*autograd.Variable, tc.T)
	for i := range xs {
		xs[i] = autograd.Var(tensor.Uniform(rng, -1, 1, tc.batch, tc.inDim))
		targets[i] = autograd.Var(tensor.Uniform(rng, -0.5, 0.5, tc.batch, 1))
	}
	if tc.h0 {
		h0 = autograd.Var(tensor.Uniform(rng, -0.5, 0.5, tc.batch, tc.units))
	}
	return cell, params, readout, xs, targets, h0
}

// gradBits snapshots a variable's gradient: nil-ness is part of the
// contract (a leaf the loss never touches must stay nil, not zero).
func gradBits(v *autograd.Variable) (nilGrad bool, bits []uint32) {
	if v.Grad == nil {
		return true, nil
	}
	bits = make([]uint32, len(v.Grad.Data))
	for i, x := range v.Grad.Data {
		bits[i] = math.Float32bits(x)
	}
	return false, bits
}

func dataBits(t *tensor.Tensor) []uint32 {
	bits := make([]uint32, len(t.Data))
	for i, x := range t.Data {
		bits[i] = math.Float32bits(x)
	}
	return bits
}

func diffBits(t *testing.T, label string, got, want []uint32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", label, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s: element %d = %#08x (%v), want %#08x (%v)", label, i,
				got[i], math.Float32frombits(got[i]), want[i], math.Float32frombits(want[i]))
		}
	}
}

// runRematReference runs the reference pipeline and snapshots every
// observable: per-step output values, the final state, the loss value and
// every leaf gradient.
func runRematReference(cell Cell, params []*autograd.Variable, readout *Linear, xs, targets []*autograd.Variable, h0 *autograd.Variable, tc rematCase) map[string][]uint32 {
	zeroRematLeaves(params, xs, h0)
	lossFn := rematLoss(tc.loss, readout, targets, params[3], tc.T)
	ys, hN := Unroll(cell, xs, h0, tc.ts)
	loss := lossFn(ys)
	loss.Backward()
	return rematSnapshot(params, xs, h0, ys, hN, loss)
}

// runRematCandidate runs UnrollRemat and snapshots the same observables.
func runRematCandidate(cell Cell, params []*autograd.Variable, readout *Linear, xs, targets []*autograd.Variable, h0 *autograd.Variable, tc rematCase) map[string][]uint32 {
	zeroRematLeaves(params, xs, h0)
	lossFn := rematLoss(tc.loss, readout, targets, params[3], tc.T)
	ys, hN, loss := UnrollRemat(cell, params, xs, h0, tc.ts, tc.chunkSize, lossFn)
	return rematSnapshot(params, xs, h0, ys, hN, loss)
}

func zeroRematLeaves(params, xs []*autograd.Variable, h0 *autograd.Variable) {
	for _, p := range params {
		p.ZeroGrad()
	}
	for _, x := range xs {
		x.ZeroGrad()
	}
	if h0 != nil {
		h0.ZeroGrad()
	}
}

// rematSnapshot flattens outputs and gradients into a name-keyed bit map;
// nil gradients are recorded under a "<name>:nil" marker key instead.
func rematSnapshot(params, xs []*autograd.Variable, h0 *autograd.Variable, ys []*autograd.Variable, hN, loss *autograd.Variable) map[string][]uint32 {
	snap := map[string][]uint32{}
	put := func(name string, v *autograd.Variable) {
		nilGrad, bits := gradBits(v)
		if nilGrad {
			snap[name+":nil"] = []uint32{1}
			return
		}
		snap[name+":nil"] = []uint32{0}
		snap[name] = bits
	}
	for i, p := range params {
		put("param"+itoa(i), p)
	}
	for i, x := range xs {
		put("x"+itoa(i), x)
	}
	if h0 != nil {
		put("h0", h0)
	}
	for i, y := range ys {
		snap["y"+itoa(i)] = dataBits(y.Data)
	}
	if hN != nil {
		snap["hN"] = dataBits(hN.Data)
	}
	snap["loss"] = dataBits(loss.Data)
	return snap
}

// itoa avoids importing strconv for three call sites in a test file.
func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var buf [8]byte
	p := len(buf)
	for i > 0 {
		p--
		buf[p] = byte('0' + i%10)
		i /= 10
	}
	return string(buf[p:])
}

// TestUnrollRematDifferential is the acceptance gate: across cells (LTC
// and CfC), dimensions, sequence lengths, chunk sizes (1, non-divisors,
// exactly T, beyond T), initial states, time spans and six loss shapes,
// UnrollRemat must reproduce the reference pipeline's outputs and every
// leaf gradient bit for bit.
func TestUnrollRematDifferential(t *testing.T) {
	cases := []rematCase{
		// The single-source corners (inDim=1 / units=1) exercise the
		// signed-zero backward paths the contraction documents.
		{name: "ltc-t1-c1-single", inDim: 1, units: 1, unfolds: 1, batch: 1, T: 1, chunkSize: 1, h0: false, ts: 1.0, loss: 0},
		{name: "ltc-t2-cGeT", inDim: 2, units: 3, unfolds: 4, batch: 2, T: 2, chunkSize: 3, h0: true, ts: 0.1, loss: 0},
		{name: "ltc-t17-c3", inDim: 4, units: 8, unfolds: 6, batch: 3, T: 17, chunkSize: 3, h0: true, ts: 1.0, loss: 0},
		{name: "ltc-t17-c4-last", inDim: 4, units: 8, unfolds: 4, batch: 2, T: 17, chunkSize: 4, h0: false, ts: 0.1, loss: 1},
		{name: "ltc-t17-cEqT-even", inDim: 3, units: 5, unfolds: 2, batch: 1, T: 17, chunkSize: 17, h0: true, ts: 1.0, loss: 2},
		{name: "ltc-t64-c8", inDim: 2, units: 4, unfolds: 3, batch: 2, T: 64, chunkSize: 8, h0: true, ts: 0.1, loss: 0},
		{name: "ltc-t64-c1-last", inDim: 1, units: 4, unfolds: 4, batch: 1, T: 64, chunkSize: 1, h0: false, ts: 1.0, loss: 1},
		{name: "ltc-t17-c100-reg", inDim: 5, units: 7, unfolds: 5, batch: 2, T: 17, chunkSize: 100, h0: true, ts: 1.0, loss: 3},
		{name: "ltc-t5-c2-even", inDim: 2, units: 3, unfolds: 1, batch: 2, T: 5, chunkSize: 2, h0: false, ts: 0.1, loss: 2},
		{name: "ltc-t9-c3-twice", inDim: 3, units: 4, unfolds: 3, batch: 2, T: 9, chunkSize: 3, h0: true, ts: 1.0, loss: 4},
		{name: "ltc-t16-c5-first", inDim: 2, units: 5, unfolds: 2, batch: 1, T: 16, chunkSize: 5, h0: true, ts: 0.1, loss: 5},
		{name: "ltc-t17-c3-lateearly", inDim: 2, units: 4, unfolds: 3, batch: 2, T: 17, chunkSize: 3, h0: true, ts: 1.0, loss: 6},
		{name: "ltc-t17-c4-midseg", inDim: 3, units: 5, unfolds: 2, batch: 1, T: 17, chunkSize: 4, h0: false, ts: 0.1, loss: 7},
		{name: "ltc-t64-c8-midseg", inDim: 2, units: 4, unfolds: 3, batch: 2, T: 64, chunkSize: 8, h0: true, ts: 1.0, loss: 7},
		{name: "ltc-t17-c3-outoforder", inDim: 2, units: 4, unfolds: 4, batch: 2, T: 17, chunkSize: 3, h0: true, ts: 0.1, loss: 8},
		{name: "ltc-t64-c1-outoforder", inDim: 1, units: 3, unfolds: 2, batch: 1, T: 64, chunkSize: 1, h0: false, ts: 1.0, loss: 8},
		{name: "ltc-t17-c5-outoforder", inDim: 2, units: 4, unfolds: 3, batch: 2, T: 17, chunkSize: 5, h0: true, ts: 1.0, loss: 8},
		{name: "ltc-t17-c17-outoforder", inDim: 3, units: 4, unfolds: 2, batch: 1, T: 17, chunkSize: 17, h0: false, ts: 0.1, loss: 8},

		{name: "cfc-t1-c1-single", cfc: true, inDim: 1, units: 1, batch: 1, T: 1, chunkSize: 1, h0: true, ts: 1.0, loss: 0},
		{name: "cfc-t2-cGeT", cfc: true, inDim: 2, units: 3, batch: 2, T: 2, chunkSize: 5, h0: false, ts: 0.1, loss: 0},
		{name: "cfc-t17-c3", cfc: true, inDim: 4, units: 8, batch: 3, T: 17, chunkSize: 3, h0: true, ts: 1.0, loss: 0},
		{name: "cfc-t17-c4-last", cfc: true, inDim: 4, units: 8, batch: 2, T: 17, chunkSize: 4, h0: false, ts: 0.1, loss: 1},
		{name: "cfc-t64-c8-even", cfc: true, inDim: 3, units: 5, batch: 1, T: 64, chunkSize: 8, h0: true, ts: 1.0, loss: 2},
		{name: "cfc-t64-cEqT", cfc: true, inDim: 2, units: 4, batch: 2, T: 64, chunkSize: 64, h0: false, ts: 0.1, loss: 0},
		{name: "cfc-t17-c1-reg", cfc: true, inDim: 5, units: 7, batch: 2, T: 17, chunkSize: 1, h0: true, ts: 1.0, loss: 3},
		{name: "cfc-t7-c2-last", cfc: true, inDim: 2, units: 6, batch: 1, T: 7, chunkSize: 2, h0: false, ts: 1.0, loss: 1},
		{name: "cfc-t12-c4-twice", cfc: true, inDim: 3, units: 4, batch: 2, T: 12, chunkSize: 4, h0: true, ts: 0.1, loss: 4},
		{name: "cfc-t16-c5-first", cfc: true, inDim: 2, units: 5, batch: 1, T: 16, chunkSize: 5, h0: false, ts: 1.0, loss: 5},
		{name: "cfc-t17-c3-midseg", cfc: true, inDim: 2, units: 4, batch: 2, T: 17, chunkSize: 3, h0: true, ts: 1.0, loss: 7},
		{name: "cfc-t17-c4-outoforder", cfc: true, inDim: 3, units: 5, batch: 1, T: 17, chunkSize: 4, h0: false, ts: 0.1, loss: 8},
		{name: "cfc-t17-c5-outoforder", cfc: true, inDim: 2, units: 4, batch: 2, T: 17, chunkSize: 5, h0: true, ts: 1.0, loss: 8},
		{name: "cfc-t17-c17-outoforder", cfc: true, inDim: 2, units: 3, batch: 1, T: 17, chunkSize: 17, h0: false, ts: 0.1, loss: 8},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cell, params, readout, xs, targets, h0 := rematSetup(tc, 20260801)
			want := runRematReference(cell, params, readout, xs, targets, h0, tc)
			got := runRematCandidate(cell, params, readout, xs, targets, h0, tc)
			if len(got) != len(want) {
				t.Fatalf("snapshot size %d, want %d", len(got), len(want))
			}
			for name, wantBits := range want {
				gotBits, ok := got[name]
				if !ok {
					t.Fatalf("missing snapshot key %q", name)
				}
				diffBits(t, name, gotBits, wantBits)
			}
		})
	}
}

// TestUnrollRematDeterministic runs the full remat pipeline twice from the
// same seed — fresh leaves each time — and requires bit-identical results,
// and a third run reusing the first run's leaves (ZeroGrad in between) to
// pin recomputation determinism on shared state.
func TestUnrollRematDeterministic(t *testing.T) {
	tc := rematCase{name: "det", inDim: 3, units: 5, unfolds: 3, batch: 2, T: 17, chunkSize: 3, h0: true, ts: 0.1, loss: 0}
	cell1, params1, readout1, xs1, targets1, h01 := rematSetup(tc, 77)
	run1 := runRematCandidate(cell1, params1, readout1, xs1, targets1, h01, tc)

	cell2, params2, readout2, xs2, targets2, h02 := rematSetup(tc, 77)
	run2 := runRematCandidate(cell2, params2, readout2, xs2, targets2, h02, tc)

	for name, want := range run1 {
		diffBits(t, "fresh/"+name, run2[name], want)
	}

	run3 := runRematCandidate(cell1, params1, readout1, xs1, targets1, h01, tc)
	for name, want := range run1 {
		diffBits(t, "reused/"+name, run3[name], want)
	}
}

// TestUnrollRematAccumulatesLikeBackward: two UnrollRemat calls without an
// intervening ZeroGrad must accumulate exactly twice the single-run
// gradient, bit for bit — the same linear-rerun contract Backward
// documents, reproduced by the reference pipeline for comparison.
func TestUnrollRematAccumulatesLikeBackward(t *testing.T) {
	tc := rematCase{name: "acc", inDim: 2, units: 4, unfolds: 3, batch: 2, T: 17, chunkSize: 4, h0: true, ts: 1.0, loss: 0}
	cell, params, readout, xs, targets, h0 := rematSetup(tc, 913)

	// Reference: two full-graph runs, no ZeroGrad between them.
	zeroRematLeaves(params, xs, h0)
	lossFn := rematLoss(tc.loss, readout, targets, params[0], tc.T)
	for run := 0; run < 2; run++ {
		ys, _ := Unroll(cell, xs, h0, tc.ts)
		lossFn(ys).Backward()
	}
	want := rematSnapshot(params, xs, h0, nil, nil, autograd.Const(tensor.New(1)))

	zeroRematLeaves(params, xs, h0)
	for run := 0; run < 2; run++ {
		UnrollRemat(cell, params, xs, h0, tc.ts, tc.chunkSize, lossFn)
	}
	got := rematSnapshot(params, xs, h0, nil, nil, autograd.Const(tensor.New(1)))
	for name, wantBits := range want {
		diffBits(t, name, got[name], wantBits)
	}
}

// TestUnrollRematEmptySequence pins the T=0 contract: an empty (non-nil)
// output slice, h0 straight through, and the loss over the empty slice
// still evaluated and backpropagated.
func TestUnrollRematEmptySequence(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	cell := NewLTC(2, 3, nil, 2, rng)
	lossFn := func(ys []*autograd.Variable) *autograd.Variable {
		if len(ys) != 0 {
			t.Fatalf("lossFn got %d outputs for an empty sequence", len(ys))
		}
		return autograd.Scale(autograd.Const(tensor.FromData([]float32{2}, 1)), 3)
	}
	// nil h0: hN must come back nil, exactly like Unroll.
	ys, hN, loss := UnrollRemat(cell, nil, nil, nil, 1.0, 4, lossFn)
	if ys == nil || len(ys) != 0 {
		t.Fatalf("ys = %v, want empty non-nil", ys)
	}
	if hN != nil {
		t.Fatalf("hN = %v, want nil for nil h0", hN)
	}
	if got := loss.Value(); got != 6 {
		t.Fatalf("loss = %v, want 6", got)
	}
	// Non-nil h0: returned unchanged (identity, as Unroll documents).
	h0 := autograd.Var(tensor.New(1, 3))
	_, hN2, _ := UnrollRemat(cell, nil, nil, h0, 1.0, 4, lossFn)
	if hN2 != h0 {
		t.Fatalf("hN is not the caller's h0 for an empty sequence")
	}
}

// TestUnrollRematChunkSizePanic pins the misuse contract: chunkSize < 1 is
// a programming error, reported with a clear panic.
func TestUnrollRematChunkSizePanic(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	cell := NewLTC(2, 3, nil, 2, rng)
	x := autograd.Var(tensor.New(1, 2))
	lossFn := func(ys []*autograd.Variable) *autograd.Variable { return autograd.SumAll(ys[0]) }
	for _, bad := range []int{0, -3} {
		func() {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("chunkSize %d: no panic", bad)
				}
				msg, ok := r.(string)
				if !ok || !strings.Contains(msg, "nn.UnrollRemat: chunkSize") {
					t.Fatalf("chunkSize %d: panic %v, want an nn.UnrollRemat chunkSize message", bad, r)
				}
			}()
			UnrollRemat(cell, nil, []*autograd.Variable{x}, nil, 1.0, bad, lossFn)
		}()
	}
}

// TestUnrollRematTsPanic pins the ts contract: invalid time spans panic
// from the cell's Step, exactly as with Unroll.
func TestUnrollRematTsPanic(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	cell := NewLTC(2, 3, nil, 2, rng)
	x := autograd.Var(tensor.New(1, 2))
	lossFn := func(ys []*autograd.Variable) *autograd.Variable { return autograd.SumAll(ys[0]) }
	for _, bad := range []float64{0, -0.5, math.NaN(), math.Inf(1)} {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatalf("ts %v: no panic", bad)
				}
			}()
			UnrollRemat(cell, nil, []*autograd.Variable{x}, nil, bad, 1, lossFn)
		}()
	}
}

// TestUnrollRematNonScalarLossPanic: a lossFn that does not reduce to a
// scalar fails at the loss backward, as the reference pipeline's
// loss.Backward() would.
func TestUnrollRematNonScalarLossPanic(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	cell := NewCfC(2, 3, nil, rng)
	x := autograd.Var(tensor.New(1, 2))
	lossFn := func(ys []*autograd.Variable) *autograd.Variable { return ys[0] } // [1, 3], not scalar
	defer func() {
		if recover() == nil {
			t.Fatalf("non-scalar loss: no panic")
		}
	}()
	UnrollRemat(cell, nil, []*autograd.Variable{x}, nil, 1.0, 1, lossFn)
}

// TestUnrollRematMissingParamPanic pins the params completeness contract:
// every trainable leaf the Step consumes must be listed in params, or
// UnrollRemat panics naming it — an unlisted leaf is not restored between
// sweeps and would silently accumulate one gradient copy per sweep (a
// 2-3x wrong value). Loss-side-only leaves (a readout's) and cells not
// exposing Parameters() are exempt.
func TestUnrollRematMissingParamPanic(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	cell := NewLTC(2, 4, nil, 3, rng)
	readout := NewLinear(4, 1, rng)
	params := ParametersOf(cell, readout)
	xs := make([]*autograd.Variable, 4)
	for i := range xs {
		xs[i] = autograd.Var(tensor.Uniform(rng, -1, 1, 2, 2))
	}
	h0 := autograd.Var(tensor.Uniform(rng, -0.5, 0.5, 2, 4))
	lossFn := func(ys []*autograd.Variable) *autograd.Variable {
		return autograd.MeanAll(autograd.Hadamard(ys[len(ys)-1], ys[len(ys)-1]))
	}
	panicOf := func(f func()) (msg string) {
		defer func() {
			if r := recover(); r != nil {
				msg = fmt.Sprint(r)
			}
		}()
		f()
		return ""
	}
	omit := func(idx int) []*autograd.Variable {
		listed := make([]*autograd.Variable, 0, len(params))
		for i, p := range params {
			if i != idx {
				listed = append(listed, p)
			}
		}
		return listed
	}
	// gleak (#0) and outW (#11) are Step-consumed: omitting either panics.
	for _, idx := range []int{0, 11} {
		msg := panicOf(func() { UnrollRemat(cell, omit(idx), xs, h0, 1.0, 3, lossFn) })
		want := fmt.Sprintf("parameter #%d", idx)
		if !strings.Contains(msg, want) {
			t.Fatalf("omit param #%d: panic %q, want one naming %q", idx, msg, want)
		}
	}
	// A readout parameter is loss-side only: omitting it stays legal.
	if msg := panicOf(func() { UnrollRemat(cell, omit(len(params)-1), xs, h0, 1.0, 3, lossFn) }); msg != "" {
		t.Fatalf("omit readout param: unexpected panic %q", msg)
	}
	// A cell that does not expose Parameters() opts out of the audit.
	type wrapper struct{ Cell }
	w := wrapper{Cell: cell}
	if msg := panicOf(func() { UnrollRemat(w, omit(0), xs, h0, 1.0, 3, lossFn) }); msg != "" {
		t.Fatalf("wrapped cell: unexpected panic %q", msg)
	}
}

// TestUnrollRematLossSideOrderPanic pins the loss-operand-order contract:
// a regularizer closing over a step-consumed leaf must be visited after
// the seeded outputs (Add(data, pen)); Add(pen, data) reverses the
// whole-graph backward's delivery order for that leaf, an order no sweep
// replays, so UnrollRemat panics instead of drifting. A penalty over a
// loss-side-only leaf (a readout's) is exempt in either order.
func TestUnrollRematLossSideOrderPanic(t *testing.T) {
	rng := rand.New(rand.NewSource(32))
	cell := NewLTC(2, 4, nil, 3, rng)
	readout := NewLinear(4, 1, rng)
	params := ParametersOf(cell, readout)
	xs := make([]*autograd.Variable, 4)
	for i := range xs {
		xs[i] = autograd.Var(tensor.Uniform(rng, -1, 1, 2, 2))
	}
	h0 := autograd.Var(tensor.Uniform(rng, -0.5, 0.5, 2, 4))
	panicOf := func(f func()) (msg string) {
		defer func() {
			if r := recover(); r != nil {
				msg = fmt.Sprint(r)
			}
		}()
		f()
		return ""
	}
	lossFn := func(reg *autograd.Variable, penFirst bool) func(ys []*autograd.Variable) *autograd.Variable {
		return func(ys []*autograd.Variable) *autograd.Variable {
			y := ys[len(ys)-1]
			data := autograd.MeanAll(autograd.Hadamard(y, y))
			pen := autograd.Scale(autograd.SumAll(autograd.Hadamard(reg, reg)), 0.01)
			if penFirst {
				return autograd.Add(pen, data)
			}
			return autograd.Add(data, pen)
		}
	}
	mu := params[3]             // cell parameter: step-consumed
	gleak := params[0]          // [units]-shaped cell parameter (broadcastable against ys)
	rw := params[len(params)-1] // readout parameter: loss-side only
	if msg := panicOf(func() { UnrollRemat(cell, params, xs, h0, 1.0, 3, lossFn(mu, false)) }); msg != "" {
		t.Fatalf("Add(data, pen): unexpected panic %q", msg)
	}
	if msg := panicOf(func() { UnrollRemat(cell, params, xs, h0, 1.0, 3, lossFn(mu, true)) }); !strings.Contains(msg, "closes over") {
		t.Fatalf("Add(pen, data) over a cell parameter: panic %q, want the loss-order message", msg)
	}
	if msg := panicOf(func() { UnrollRemat(cell, params, xs, h0, 1.0, 3, lossFn(rw, true)) }); msg != "" {
		t.Fatalf("Add(pen, data) over a readout parameter: unexpected panic %q", msg)
	}
	if msg := panicOf(func() { UnrollRemat(cell, params, xs, h0, 1.0, 3, lossFn(h0, true)) }); !strings.Contains(msg, "closes over") {
		t.Fatalf("Add(pen, data) over h0: panic %q, want the loss-order message", msg)
	}
	if msg := panicOf(func() { UnrollRemat(cell, params, xs, h0, 1.0, 3, lossFn(xs[2], true)) }); !strings.Contains(msg, "closes over") {
		t.Fatalf("Add(pen, data) over xs[2]: panic %q, want the loss-order message", msg)
	}
	// Folding the penalty into the seeded per-step terms is NOT a remedy:
	// those consumers are visited before the last output — the drift
	// structure the check exists for.
	perTermFold := func(ys []*autograd.Variable) *autograd.Variable {
		var acc *autograd.Variable
		for i, y := range ys {
			term := autograd.SumAll(autograd.Hadamard(y, gleak))
			if i == 0 {
				acc = term
			} else {
				acc = autograd.Add(acc, term)
			}
		}
		return acc
	}
	if msg := panicOf(func() { UnrollRemat(cell, params, xs, h0, 1.0, 3, perTermFold) }); !strings.Contains(msg, "closes over") {
		t.Fatalf("per-term folded penalty: panic %q, want the loss-order message", msg)
	}
	// A constant-ized penalty (over a detached copy) is loss-side only:
	// legal in any position, and mu keeps exactly the data-loss gradient.
	detached := autograd.Detach(mu)
	constPen := func(ys []*autograd.Variable) *autograd.Variable {
		y := ys[len(ys)-1]
		data := autograd.MeanAll(autograd.Hadamard(y, y))
		pen := autograd.Scale(autograd.SumAll(autograd.Hadamard(detached, detached)), 0.01)
		return autograd.Add(pen, data)
	}
	dataOnly := func(ys []*autograd.Variable) *autograd.Variable {
		y := ys[len(ys)-1]
		return autograd.MeanAll(autograd.Hadamard(y, y))
	}
	mu.ZeroGrad()
	ysRef, _ := Unroll(cell, xs, h0, 1.0)
	dataOnly(ysRef).Backward()
	want := append([]float32(nil), mu.Grad.Data...)
	mu.ZeroGrad()
	if msg := panicOf(func() { UnrollRemat(cell, params, xs, h0, 1.0, 3, constPen) }); msg != "" {
		t.Fatalf("constant-ized penalty: unexpected panic %q", msg)
	}
	for i, v := range mu.Grad.Data {
		if v != want[i] {
			t.Fatalf("constant-ized penalty: mu.Grad[%d] = %v, want %v (data-only)", i, v, want[i])
		}
	}
	// A loss touching no output at all seeds nothing: no step-side
	// contributions exist, so there is no order to violate — no panic,
	// and h0 keeps the loss backward's own gradient.
	onlyH0 := func(ys []*autograd.Variable) *autograd.Variable { return autograd.SumAll(h0) }
	h0.ZeroGrad()
	if msg := panicOf(func() { UnrollRemat(cell, params, xs, h0, 1.0, 3, onlyH0) }); msg != "" {
		t.Fatalf("loss over h0 only: unexpected panic %q", msg)
	}
	for i, v := range h0.Grad.Data {
		if v != 1 {
			t.Fatalf("loss over h0 only: h0.Grad[%d] = %v, want 1", i, v)
		}
	}
}

// TestStepDoesNotMutateInputs is the empirical premise for Detach's
// zero-copy aliasing underneath UnrollRemat: neither LTC.Step nor CfC.Step
// may write into the storage of its input x or input state h, and a state
// produced by one step must survive the next step bit for bit. If any of
// these checks ever failed, Detach would have to copy.
func TestStepDoesNotMutateInputs(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	cells := []Cell{
		NewLTC(3, 5, nil, 4, rng),
		NewCfC(3, 5, nil, rng),
	}
	for ci, cell := range cells {
		x := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 3))
		h := autograd.Var(tensor.Uniform(rng, -0.5, 0.5, 2, 5))
		xBefore, hBefore := x.Data.Clone(), h.Data.Clone()

		_, h1 := cell.Step(x, h, 0.1)
		if !sameBitsT(x.Data, xBefore) {
			t.Fatalf("cell %d: Step mutated its input x", ci)
		}
		if !sameBitsT(h.Data, hBefore) {
			t.Fatalf("cell %d: Step mutated its input state h", ci)
		}

		h1Before := h1.Data.Clone()
		_, h2 := cell.Step(x, h1, 0.1)
		if !sameBitsT(h1.Data, h1Before) {
			t.Fatalf("cell %d: a later Step mutated an earlier state", ci)
		}
		h2Before := h2.Data.Clone()
		// Run a backward over a loss spanning both steps: gradient
		// propagation must not rewrite forward values either.
		y1 := autograd.Add(h1, h2)
		autograd.SumAll(y1).Backward()
		if !sameBitsT(h1.Data, h1Before) || !sameBitsT(h2.Data, h2Before) {
			t.Fatalf("cell %d: Backward mutated a forward state", ci)
		}
	}
}

// TestRematFusedCfCFoldClasses pins the fused CfC step's fold classes
// (stage 18b): with h the fused node's FIRST parent, the DFS descent hits
// the state leaf before any parameter chain, so every trainable leaf
// stays state-rest except the OUTPUT-class outW/outB — no spine class,
// hence no σ sweep for the CfC (remat.go's "never for the CfC" survives
// the fusion). A regression that reorders the fused node's parents would
// either move a leaf into classSpine or trip the multi-class panic.
func TestRematFusedCfCFoldClasses(t *testing.T) {
	rng := rand.New(rand.NewSource(97))
	cell := NewCfC(3, 4, nil, rng)
	x := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 3))
	exempt := map[*autograd.Variable]bool{x: true}
	swept := append(append([]*autograd.Variable{}, cell.Parameters()...), x)
	classes, consumed := classifyFoldClasses(cell, x, 0.1, swept, exempt)
	names := []string{"gleak", "vleak", "cm", "mu", "sigma", "w", "sMu", "sSigma", "sW", "inW", "inB", "outW", "outB"}
	for i, p := range cell.Parameters() {
		if !consumed[p] {
			t.Fatalf("parameter %s has no consumer in the fused step graph", names[i])
		}
		want := classStateRest
		if i >= 11 { // outW, outB live on the output branch
			want = classOutput
		}
		if classes[p] != want {
			t.Fatalf("parameter %s: fold class %d, want %d (fused CfC must stay spine-free)", names[i], classes[p], want)
		}
	}
	if classes[x] != classStateRest {
		t.Fatalf("x: fold class %d, want state-rest", classes[x])
	}
}
