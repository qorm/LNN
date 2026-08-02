// Native Go fuzzing for the two v0.5.0 numerical kernels — the fused LTC
// unfold and UnrollRemat's chunked BPTT — as PERMANENT bitwise differential
// gates. The red team's one-shot 5000-seed adversarial sweep (RT-F) is
// crystallized here into resident oracles:
//
//   - FuzzLTCFusedDifferential: LTC.Step (fused kernel, nn/ltc_fused.go) vs
//     legacyLTCStep (the pre-fusion graph path kept verbatim in
//     nn/ltc_fused_diff_test.go). Forward out/hNew, all 13 parameter
//     gradients, and the x/h leaf gradients must be Float32bits-identical —
//     or both paths must panic at the same point with the same message
//     (irregular-seed shapes: broadcast-past expansion classes and
//     non-broadcastable classes, panic parity included in the property).
//   - FuzzUnrollRematDifferential: UnrollRemat vs the whole-graph reference
//     "Unroll + same lossFn + Backward" over LTC and CfC: every leaf
//     gradient bit for bit, with the contract panics (params
//     incompleteness, loss-side consumer order, multi-class shared leaf)
//     asserted as panics that leave no partial gradients behind.
//
// Bit discipline: the assertions are Float32bits equality, never tolerance.
// The single exemption is the documented double-NaN payload corner (the
// fused kernel's mul32 native-multiply arm propagates a NaN through the
// hardware's operand choice, which may differ from the legacy chain's when
// BOTH operands are NaN): a position where both bits are NaN (any
// payload/sign) passes, but the NaN position set and every finite value
// must agree exactly.
//
// Internal test package (nn): the targets drive the unexported oracle
// (legacyLTCStep, boundLegacy) and the shared differential helpers of
// nn/ltc_fused_diff_test.go and nn/remat_test.go.
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

// fuzzPosMod maps a fuzz-derived int (any sign) onto [0, n).
func fuzzPosMod(v, n int) int {
	m := v % n
	if m < 0 {
		m += n
	}
	return m
}

// fuzzPanicOf runs f and returns its panic message ("" when f returns).
func fuzzPanicOf(f func()) (msg string) {
	defer func() {
		if r := recover(); r != nil {
			msg = fmt.Sprint(r)
		}
	}()
	f()
	return ""
}

// fuzzIsNaNBits reports whether b is any NaN (all-ones exponent, nonzero
// mantissa) — payload and sign included.
func fuzzIsNaNBits(b uint32) bool {
	return b&0x7F800000 == 0x7F800000 && b&0x007FFFFF != 0
}

// fuzzCmpBitsNaN compares two bit slices element by element: exact
// Float32bits equality, with the documented double-NaN payload corner
// exempted (both NaN passes; NaN vs finite does not — the NaN position set
// is part of the property).
func fuzzCmpBitsNaN(t *testing.T, name string, got, want []uint32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if got[i] == want[i] {
			continue
		}
		if fuzzIsNaNBits(got[i]) && fuzzIsNaNBits(want[i]) {
			continue // double-NaN payload corner (see file header)
		}
		t.Fatalf("%s: element %d = %#08x (%v), want %#08x (%v)", name, i,
			got[i], math.Float32frombits(got[i]), want[i], math.Float32frombits(want[i]))
	}
}

// fuzzGradSnap is one leaf's gradient snapshot: nil-ness, delivered shape
// (the 1D-lift quirk makes shapes part of the backward contract) and bits.
type fuzzGradSnap struct {
	nilGrad bool
	shape   string
	bits    []uint32
}

func fuzzSnapOne(v *autograd.Variable) fuzzGradSnap {
	if v.Grad == nil {
		return fuzzGradSnap{nilGrad: true}
	}
	return fuzzGradSnap{shape: fmt.Sprint(v.Grad.Shape), bits: dataBits(v.Grad)}
}

// fuzzLTCParamNames mirrors the frozen Parameters() order (nn/ltc.go).
var fuzzLTCParamNames = []string{"gleak", "vleak", "cm", "mu", "sigma", "w", "sMu", "sSigma", "sW", "inW", "inB", "outW", "outB"}

// fuzzSnapGrads snapshots the cell's 13 parameters plus the extra leaves.
func fuzzSnapGrads(cell *LTC, extra map[string]*autograd.Variable) map[string]fuzzGradSnap {
	out := make(map[string]fuzzGradSnap, len(fuzzLTCParamNames)+len(extra))
	for i, p := range cell.Parameters() {
		out[fuzzLTCParamNames[i]] = fuzzSnapOne(p)
	}
	for name, v := range extra {
		out[name] = fuzzSnapOne(v)
	}
	return out
}

// fuzzSnapGradsCfC is fuzzSnapGrads for the CfC cell (same frozen 13-name
// parameter order).
func fuzzSnapGradsCfC(cell *CfC, extra map[string]*autograd.Variable) map[string]fuzzGradSnap {
	out := make(map[string]fuzzGradSnap, len(fuzzLTCParamNames)+len(extra))
	for i, p := range cell.Parameters() {
		out[fuzzLTCParamNames[i]] = fuzzSnapOne(p)
	}
	for name, v := range extra {
		out[name] = fuzzSnapOne(v)
	}
	return out
}

// fuzzZeroAllCfC zeroes the CfC cell's parameters and the extra leaves.
func fuzzZeroAllCfC(cell *CfC, extra map[string]*autograd.Variable) {
	for _, p := range cell.Parameters() {
		p.ZeroGrad()
	}
	for _, v := range extra {
		v.ZeroGrad()
	}
}

// fuzzCmpSnaps compares two gradient snapshots: key sets, nil-ness,
// delivered shapes and bits (double-NaN exempted) must all agree.
func fuzzCmpSnaps(t *testing.T, got, want map[string]fuzzGradSnap) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("gradient key count %d vs %d", len(got), len(want))
	}
	for name, ga := range got {
		gb, ok := want[name]
		if !ok {
			t.Fatalf("missing gradient %q", name)
		}
		if ga.nilGrad != gb.nilGrad {
			t.Fatalf("%s: nil-ness differs (%v vs %v)", name, ga.nilGrad, gb.nilGrad)
		}
		if ga.nilGrad {
			continue
		}
		if ga.shape != gb.shape {
			t.Fatalf("%s: gradient shape %s vs %s", name, ga.shape, gb.shape)
		}
		fuzzCmpBitsNaN(t, "grad "+name, ga.bits, gb.bits)
	}
}

// fuzzPayloadBits is the hostile float32 menu of the RT-F taxonomy: signed
// zeros, subnormals, max finite, infinities and NaNs with assorted
// payloads/signs.
var fuzzPayloadBits = []uint32{
	0x00000000, 0x80000000,
	0x00000001, 0x80000001,
	0x7F7FFFFF, 0xFF7FFFFF,
	0x7F800000, 0xFF800000,
	0x7FC00000, 0xFFC00000,
	0x7FC00001, 0x7FC12345,
}

// fuzzFillHostile overwrites data with the payload menu at the given
// density, leaving the rest untouched.
func fuzzFillHostile(rng *rand.Rand, data []float32, density float64) {
	for i := range data {
		if rng.Float64() < density {
			data[i] = math.Float32frombits(fuzzPayloadBits[rng.Intn(len(fuzzPayloadBits))])
		}
	}
}

// fuzzLTCStepFn is the Step signature both paths are driven through.
type fuzzLTCStepFn func(x, h *autograd.Variable, ts float64) (out, v *autograd.Variable)

// FuzzLTCFusedDifferential drives one random configuration per iteration —
// units <= 8, inDim <= 4, unfolds <= 8, batch <= 4 (a single iteration
// stays cheap), wiring in {zero masks, random sparse, fully connected}, ts
// in {normal, clamped-tiny, 1.0, clamped-huge}, the RT-F payload classes
// {clean, signed-zero inputs, extreme finite weights, NaN/Inf weights,
// NaN/Inf inputs, saturating sigma} and four scenarios: seeded single
// step, short unroll under the five ltcUnrollLoss shapes, the chained
// topology (step 2's input a graph function of step 1's output — the RT-F
// Finding C regression), and irregular root seeds on the state node
// (expansion classes and non-broadcastable classes, both-paths-same-panic
// in the property).
func FuzzLTCFusedDifferential(f *testing.F) {
	// One representative per known divergence class; the same tuples are
	// committed as corpus files under testdata/fuzz/FuzzLTCFusedDifferential.
	f.Add(int64(1), 0, 0, 0)   // base: sparse wiring, ts=1.0, seeded single step
	f.Add(int64(3), 3, 0, 12)  // expansion seed [3,2] over a units==1 state
	f.Add(int64(2), 0, 1, 1)   // zero masks, batch==1, signed-zero inputs and seeds
	f.Add(int64(6), 0, 0, 2)   // sparse, hostile-laced seeds, clamped-huge ts
	f.Add(int64(8), 0, 0, 0)   // units==1 && batch==1
	f.Add(int64(4), 0, 3, 0)   // NaN/Inf recurrent weights, clamped-tiny ts
	f.Add(int64(13), 1, 4, 2)  // NaN/Inf inputs, unroll, out-of-order loss
	f.Add(int64(7), 2, 0, 0)   // chained topology (RT-F Finding C)
	f.Add(int64(5), 3, 0, 8)   // non-broadcastable seed: panic parity
	f.Add(int64(11), 3, 0, 14) // rank-3 seed over units==1: panic parity
	f.Add(int64(2147483647), 0, 0, 0)

	f.Fuzz(func(t *testing.T, seed int64, scenIn, payloadIn, seedSelIn int) {
		rng := rand.New(rand.NewSource(seed))
		units := 1 + rng.Intn(8)
		inDim := 1 + rng.Intn(4)
		unfolds := 1 + rng.Intn(8)
		batch := 1 + rng.Intn(4)
		scen := fuzzPosMod(scenIn, 4)
		payload := fuzzPosMod(payloadIn, 6)
		if scen == 2 {
			inDim = units // the chain feeds a [batch, units] output back as input
		}

		// Wiring: zero masks / random sparse / fully connected.
		var w *Wiring
		switch rng.Intn(4) {
		case 0:
			w = RandomSparse(inDim, units, 0, 0, rng)
		case 1, 2:
			w = RandomSparse(inDim, units, float32(1+rng.Intn(10))/10, float32(1+rng.Intn(10))/10, rng)
		}

		// ts: normal / clamped tiny (float32-subnormal territory) / 1.0 /
		// clamped huge — all positive and finite, per Step's contract.
		var ts float64
		switch rng.Intn(4) {
		case 0:
			ts = 0.05 + 10*rng.Float64()
		case 1:
			ts = math.Pow(10, -45+43*rng.Float64())
		case 2:
			ts = 1.0
		default:
			ts = math.Pow(10, 2+298*rng.Float64())
		}

		cell := NewLTC(inDim, units, w, unfolds, rng)
		x := autograd.Var(tensor.Uniform(rng, -1, 1, batch, inDim))
		h := autograd.Var(tensor.Uniform(rng, -1, 1, batch, units))

		// Payload classes (RT-F): the cell and inputs are rewritten in place
		// before either path runs, so both see identical values.
		switch payload {
		case 1: // signed zeros in x and h
			for i := range x.Data.Data {
				x.Data.Data[i] = math.Float32frombits([]uint32{0, 0x80000000}[rng.Intn(2)])
			}
			for i := range h.Data.Data {
				h.Data.Data[i] = math.Float32frombits([]uint32{0, 0x80000000}[rng.Intn(2)])
			}
		case 2: // extreme finite weights
			for _, p := range cell.Parameters() {
				for i := range p.Data.Data {
					p.Data.Data[i] = -1e30 + 2e30*rng.Float32()
				}
			}
		case 3: // NaN/Inf payloads in the recurrent weights
			for _, pi := range []int{3, 4, 5} {
				fuzzFillHostile(rng, cell.Parameters()[pi].Data.Data, 1.0/6)
			}
		case 4: // NaN/Inf payloads in x and h
			fuzzFillHostile(rng, x.Data.Data, 1.0/6)
			fuzzFillHostile(rng, h.Data.Data, 1.0/6)
		case 5: // saturating sigma: sigmoid hits exactly 1, 1-s is +0 (the
			// signed-zero row-accumulator corner)
			for i := range cell.sigma.Data.Data {
				cell.sigma.Data.Data[i] = 1e37
			}
		}

		// The scenario closes over every draw (both paths must see identical
		// constants) and returns the forward bits to compare.
		var extra map[string]*autograd.Variable
		var body func(step fuzzLTCStepFn) map[string][]uint32
		switch scen {
		case 0: // seeded single step
			extra = map[string]*autograd.Variable{"x": x, "h": h}
			soT := tensor.Uniform(rng, -1, 1, batch, units)
			svT := tensor.Uniform(rng, -1, 1, batch, units)
			switch fuzzPosMod(seedSelIn, 3) {
			case 1: // signed-zero seeds
				for i := range soT.Data {
					soT.Data[i] = math.Float32frombits([]uint32{0, 0x80000000}[rng.Intn(2)])
					svT.Data[i] = math.Float32frombits([]uint32{0, 0x80000000}[rng.Intn(2)])
				}
			case 2: // hostile-laced seeds
				fuzzFillHostile(rng, soT.Data, 0.2)
				fuzzFillHostile(rng, svT.Data, 0.2)
			}
			body = func(step fuzzLTCStepFn) map[string][]uint32 {
				out, v := step(x, h, ts)
				autograd.Add(
					autograd.SumAll(autograd.Hadamard(out, autograd.Const(soT))),
					autograd.SumAll(autograd.Hadamard(v, autograd.Const(svT)))).Backward()
				return map[string][]uint32{"out": dataBits(out.Data), "state": dataBits(v.Data)}
			}
		case 1: // short unroll under the five loss shapes
			T := 2 + rng.Intn(3)
			xs := make([]*autograd.Variable, T)
			extra = map[string]*autograd.Variable{}
			for i := range xs {
				xs[i] = autograd.Var(tensor.Uniform(rng, -1, 1, batch, inDim))
				extra["x"+itoa(i)] = xs[i]
			}
			h0 := autograd.Var(tensor.Uniform(rng, -1, 1, batch, units))
			extra["h0"] = h0
			seeds := make([]*tensor.Tensor, T+1)
			for i := range seeds {
				seeds[i] = tensor.Uniform(rng, -1, 1, batch, units)
			}
			kind := fuzzPosMod(seedSelIn, 5)
			body = func(step fuzzLTCStepFn) map[string][]uint32 {
				hh := h0
				ys := make([]*autograd.Variable, T)
				fwd := make(map[string][]uint32, T+1)
				for i, xi := range xs {
					ys[i], hh = step(xi, hh, ts)
					fwd["y"+itoa(i)] = dataBits(ys[i].Data)
				}
				fwd["hN"] = dataBits(hh.Data)
				ltcUnrollLoss(kind, ys, hh, seeds).Backward()
				return fwd
			}
		case 2: // chained: step 2's input is a graph function of step 1's output
			extra = map[string]*autograd.Variable{"x0": x, "h0": h}
			c1 := tensor.Uniform(rng, -1, 1, 1, units)
			c2 := tensor.Uniform(rng, -1, 1, batch, units)
			sv := tensor.Uniform(rng, -1, 1, batch, units)
			body = func(step fuzzLTCStepFn) map[string][]uint32 {
				y0, h1 := step(x, h, ts)
				x1c := autograd.Add(autograd.Hadamard(y0, autograd.Const(c1)), autograd.Const(c2))
				y1, h2 := step(x1c, h1, ts)
				autograd.SumAll(autograd.Hadamard(h2, autograd.Const(sv))).Backward()
				return map[string][]uint32{
					"y0": dataBits(y0.Data), "h1": dataBits(h1.Data),
					"y1": dataBits(y1.Data), "h2": dataBits(h2.Data),
				}
			}
		default: // irregular root seed on the state node (panic parity in scope)
			extra = map[string]*autograd.Variable{"x": x, "h": h}
			cand := [][]int{
				{1}, {1, 1}, {batch, units}, {1, units}, {batch, 1},
				{batch * units}, {batch*units + 1},
				{batch + 1, units}, {1, units + 1}, {batch, units + 1},
				{1, 2}, {2, 1}, {3, 2}, {2, 3}, {batch, units, 1},
			}
			sh := cand[fuzzPosMod(seedSelIn, len(cand))]
			n := 1
			for _, d := range sh {
				n *= d
			}
			flat := make([]float32, n)
			for i := range flat {
				flat[i] = -1 + 2*rng.Float32()
			}
			if rng.Intn(2) == 0 {
				fuzzFillHostile(rng, flat, 0.2)
			}
			body = func(step fuzzLTCStepFn) map[string][]uint32 {
				out, v := step(x, h, ts)
				v.Grad = tensor.FromData(append([]float32(nil), flat...), sh...)
				v.Backward()
				return map[string][]uint32{"out": dataBits(out.Data), "state": dataBits(v.Data)}
			}
		}

		// One path per run: identical draws, ZeroGrad between. A panic
		// surfaces as its message; parity requires the SAME message from
		// both paths (same point, same text) or identical bits throughout.
		runPath := func(step fuzzLTCStepFn) (fwd map[string][]uint32, grads map[string]fuzzGradSnap, msg string) {
			defer func() {
				if r := recover(); r != nil {
					msg = fmt.Sprint(r)
				}
			}()
			fusedZeroAll(cell, extra)
			fwd = body(step)
			grads = fuzzSnapGrads(cell, extra)
			return fwd, grads, ""
		}
		fwdF, gradsF, mF := runPath(cell.Step)
		fwdL, gradsL, mL := runPath(boundLegacy(cell))
		if mF != mL {
			t.Fatalf("panic parity: fused %q, legacy %q", mF, mL)
		}
		if mF != "" {
			return // both panicked with the identical message: parity holds
		}
		for name, want := range fwdL {
			fuzzCmpBitsNaN(t, "fwd "+name, fwdF[name], want)
		}
		fuzzCmpSnaps(t, gradsF, gradsL)
	})
}

// fuzzRequireNilGrads asserts a contract panic left no partial gradients:
// UnrollRemat's structural audits run before the loss backward, so a
// recover-and-retry can never double-count.
func fuzzRequireNilGrads(t *testing.T, leaves ...*autograd.Variable) {
	t.Helper()
	for i, v := range leaves {
		if v == nil { // a nil h0 stays nil
			continue
		}
		if v.Grad != nil {
			t.Fatalf("leaf %d holds a gradient after a contract panic (shape %v)", i, v.Grad.Shape)
		}
	}
}

// FuzzUnrollRematDifferential drives UnrollRemat against the whole-graph
// reference (Unroll + the same lossFn + Backward) over random
// configurations: cell in {LTC, CfC}, T <= 16, chunkSize in [1, T+3], h0
// nil or not, ts in five regimes, and the loss menu — the ten rematLoss
// kinds, a random subset in random visit order, and gate losses over cell
// parameters of every fold class (2D params with batch in {1, units}; the
// non-lifted 1D params; the softplus-lifted 1D params, whose gate panics
// identically in BOTH pipelines with the engine's shape-mismatch message).
// The contract panics (params incompleteness, loss-side consumer order,
// multi-class shared leaf) are asserted as panics that leave every leaf
// gradient nil.
func FuzzUnrollRematDifferential(f *testing.F) {
	// The same tuples are committed as corpus files under
	// testdata/fuzz/FuzzUnrollRematDifferential.
	f.Add(int64(16), 0, 0)  // LTC, full-sequence MSE, chunkSize=1
	f.Add(int64(1), 1, 0)   // CfC, final-step loss
	f.Add(int64(10), 8, 0)  // out-of-order visits: the affine pass
	f.Add(int64(4), 3, 0)   // CfC, L2 regularizer data-first
	f.Add(int64(13), 10, 0) // random subset in random visit order
	f.Add(int64(21), 11, 0) // gate over a 2D cell parameter
	f.Add(int64(9), 12, 0)  // CfC, gate over a non-lifted 1D parameter
	f.Add(int64(11), 9, 0)  // long-merge adversarial visits, chunk > T
	f.Add(int64(5), 0, 7)   // params incompleteness: must panic, no partial grads
	f.Add(int64(10), 0, 8)  // loss-side consumer order: must panic
	f.Add(int64(14), 0, 9)  // multi-class shared leaf: must panic
	f.Add(int64(17), 13, 0) // gate over a lifted 1D param: both paths panic alike
	f.Add(int64(20), 0, 0)  // LTC units==1, clamped-tiny ts

	f.Fuzz(func(t *testing.T, seed int64, lossSelIn, contractIn int) {
		rng := rand.New(rand.NewSource(seed))
		lossSel := fuzzPosMod(lossSelIn, 14)
		contract := fuzzPosMod(contractIn, 10)

		cfc := rng.Intn(2) == 0
		inDim := 1 + rng.Intn(4)
		units := 1 + rng.Intn(8)
		unfolds := 1 + rng.Intn(6)
		batch := 1 + rng.Intn(3)
		T := 1 + rng.Intn(16)
		if lossSel == 9 && T < 3 {
			T = 3 // the long-merge loss indexes ys[n-2]
		}
		chunkSize := 1 + rng.Intn(T+3)
		h0f := rng.Intn(2) == 0
		if lossSel == 11 {
			// A 2D-parameter gate broadcasts [units, units] against
			// [batch, units] only when batch is 1 or units.
			if rng.Intn(2) == 0 {
				batch = 1
			} else {
				batch = units
			}
		}
		var ts float64
		switch rng.Intn(5) {
		case 0:
			ts = 0.1
		case 1:
			ts = 1.0
		case 2:
			ts = 5.5
		case 3:
			ts = math.Pow(10, -30+28*rng.Float64()) // clamped-tiny regime
		default:
			ts = math.Pow(10, 2+298*rng.Float64()) // clamped-huge regime
		}

		tc := rematCase{cfc: cfc, inDim: inDim, units: units, unfolds: unfolds,
			batch: batch, T: T, chunkSize: chunkSize, h0: h0f, ts: ts}
		cell, params, readout, xs, targets, h0 := rematSetup(tc, seed)

		switch contract {
		case 7: // params incompleteness: the audit must panic naming the index
			m := cell.(Module)
			idx := rng.Intn(len(m.Parameters()))
			listed := make([]*autograd.Variable, 0, len(params)-1)
			for i, p := range params {
				if i != idx {
					listed = append(listed, p)
				}
			}
			lossFn := rematLoss(1, readout, targets, params[3], T)
			zeroRematLeaves(params, xs, h0)
			msg := fuzzPanicOf(func() { UnrollRemat(cell, listed, xs, h0, ts, chunkSize, lossFn) })
			if !strings.Contains(msg, "missing from the params list") ||
				!strings.Contains(msg, fmt.Sprintf("parameter #%d", idx)) {
				t.Fatalf("omit param #%d: panic %q, want the params-completeness message", idx, msg)
			}
			fuzzRequireNilGrads(t, append(append(append([]*autograd.Variable{}, params...), xs...), h0)...)
			return
		case 8: // loss-side consumer order: pen-first must panic, not drift
			reg := params[3]
			lossFn := func(ys []*autograd.Variable) *autograd.Variable {
				y := ys[len(ys)-1]
				data := autograd.MeanAll(autograd.Hadamard(y, y))
				pen := autograd.Scale(autograd.SumAll(autograd.Hadamard(reg, reg)), 0.01)
				return autograd.Add(pen, data)
			}
			zeroRematLeaves(params, xs, h0)
			msg := fuzzPanicOf(func() { UnrollRemat(cell, params, xs, h0, ts, chunkSize, lossFn) })
			if !strings.Contains(msg, "closes over") {
				t.Fatalf("pen-first loss: panic %q, want the loss-order message", msg)
			}
			fuzzRequireNilGrads(t, append(append(append([]*autograd.Variable{}, params...), xs...), h0)...)
			return
		case 9: // a shared leaf consumed in two fold classes: the probe must
			// report it instead of drifting
			rngM := rand.New(rand.NewSource(seed))
			mc := newMixedCell(inDim, units, rngM)
			xsM := make([]*autograd.Variable, T)
			for i := range xsM {
				xsM[i] = autograd.Var(tensor.Uniform(rngM, -1, 1, batch, inDim))
			}
			h0M := autograd.Var(tensor.Uniform(rngM, -0.5, 0.5, batch, units))
			lossFn := func(ys []*autograd.Variable) *autograd.Variable {
				return autograd.MeanAll(autograd.Hadamard(ys[len(ys)-1], ys[len(ys)-1]))
			}
			zeroRematLeaves(mc.Parameters(), xsM, h0M)
			msg := fuzzPanicOf(func() { UnrollRemat(mc, mc.Parameters(), xsM, h0M, ts, chunkSize, lossFn) })
			if !strings.Contains(msg, "fold classes") {
				t.Fatalf("multi-class leaf: panic %q, want the fold-classes message", msg)
			}
			fuzzRequireNilGrads(t, append(append(append([]*autograd.Variable{}, mc.Parameters()...), xsM...), h0M)...)
			return
		}

		// Differential scenarios: build the loss once (every draw closed
		// over), run both pipelines, compare the full snapshot bit for bit.
		var lossFn func(ys []*autograd.Variable) *autograd.Variable
		switch {
		case lossSel <= 9:
			lossFn = rematLoss(lossSel, readout, targets, params[3], T)
		case lossSel == 10: // random subset in random visit order
			perm := rng.Perm(T)
			k := 1 + rng.Intn(T)
			wts := make([]float32, k)
			for i := range wts {
				wts[i] = -2 + 4*rng.Float32()
			}
			lossFn = func(ys []*autograd.Variable) *autograd.Variable {
				var acc *autograd.Variable
				for i := 0; i < k; i++ {
					term := autograd.SumAll(autograd.Scale(ys[perm[i]], wts[i]))
					if acc == nil {
						acc = term
					} else {
						acc = autograd.Add(acc, term)
					}
				}
				return acc
			}
		default: // 11..13: gate losses over a cell parameter
			var pidx int
			switch lossSel {
			case 11:
				// [units, units] recurrent params only (sMu is [inDim,
				// units]: its batch compatibility differs and the batch
				// override above is sized for [units, units]).
				pidx = []int{3, 4, 5}[rng.Intn(3)] // mu/sigma/w (2D)
			case 12:
				pidx = []int{1, 11, 12}[rng.Intn(3)] // vleak/outW/outB (1D, no lift)
			default:
				pidx = []int{0, 2}[rng.Intn(2)] // gleak/cm (1D through Softplus)
			}
			g := params[pidx]
			tgt := autograd.Const(tensor.Uniform(rng, -0.5, 0.5, batch, units))
			gFirst := rng.Intn(2) == 0
			lossFn = func(ys []*autograd.Variable) *autograd.Variable {
				y := ys[len(ys)-1]
				var gated *autograd.Variable
				if gFirst {
					gated = autograd.Hadamard(g, y)
				} else {
					gated = autograd.Hadamard(y, g)
				}
				diff := autograd.Sub(gated, tgt)
				return autograd.MeanAll(autograd.Hadamard(diff, diff))
			}
		}

		if lossSel == 13 {
			// The softplus 1D lift makes the step-side and gate-side
			// deliveries disagree in shape: BOTH pipelines must raise the
			// engine's identical shape-mismatch panic (the penalty-leaf
			// contract), never a one-sided panic or a silent drift.
			mR := fuzzPanicOf(func() {
				zeroRematLeaves(params, xs, h0)
				ys, _ := Unroll(cell, xs, h0, ts)
				lossFn(ys).Backward()
			})
			mC := fuzzPanicOf(func() {
				zeroRematLeaves(params, xs, h0)
				UnrollRemat(cell, params, xs, h0, ts, chunkSize, lossFn)
			})
			if mR == "" || mR != mC || !strings.Contains(mR, "gradient shape mismatch") {
				t.Fatalf("lifted-1D gate: reference %q, remat %q: both must raise the identical shape-mismatch panic", mR, mC)
			}
			return
		}

		zeroRematLeaves(params, xs, h0)
		ysR, hNR := Unroll(cell, xs, h0, ts)
		lossR := lossFn(ysR)
		lossR.Backward()
		want := rematSnapshot(params, xs, h0, ysR, hNR, lossR)

		zeroRematLeaves(params, xs, h0)
		ysC, hNC, lossC := UnrollRemat(cell, params, xs, h0, ts, chunkSize, lossFn)
		got := rematSnapshot(params, xs, h0, ysC, hNC, lossC)

		if len(got) != len(want) {
			t.Fatalf("snapshot size %d, want %d", len(got), len(want))
		}
		for name, wantBits := range want {
			gotBits, ok := got[name]
			if !ok {
				t.Fatalf("missing snapshot key %q", name)
			}
			// Strict Float32bits, no NaN exemption: both pipelines here run
			// the engine's standard graph backward over identical finite
			// values, so a payload drift would be a REAL finding, not the
			// fused kernel's documented mul32 corner.
			diffBits(t, name, gotBits, wantBits)
		}
	})
}

// FuzzCfCFusedDifferential drives one random CfC configuration per
// iteration — units <= 8, inDim <= 4, batch <= 4 (a single iteration
// stays cheap), wiring in {zero masks, random sparse, fully connected},
// ts in {normal, clamped-tiny (the exprel Taylor regime), 1.0,
// clamped-huge (direct/saturation), exprel-boundary (B straddling 1e-2
// per element)}, the payload classes {clean, signed-zero inputs, extreme
// finite weights, NaN/Inf weights, NaN/Inf inputs, saturating sigma} and
// four scenarios: seeded single step, short unroll under the five
// ltcUnrollLoss shapes, the chained topology, and irregular root seeds on
// the state node. The irregular scenario covers the terminal Add's four
// arms (nn/cfc_fused.go): exact shape and the batch == 1 row-vector /
// units == 1 column-vector reductions FLOW with values (full bit
// comparison); the batch == units == 1 scalar arm and every other shape
// must panic with the IDENTICAL message in both paths (parity).
func FuzzCfCFusedDifferential(f *testing.F) {
	// One representative per known divergence class; the same tuples are
	// committed as corpus files under testdata/fuzz/FuzzCfCFusedDifferential.
	f.Add(int64(1), 0, 0, 0)   // base: sparse wiring, ts=1.0, seeded single step
	f.Add(int64(2), 0, 0, 1)   // signed-zero seeds
	f.Add(int64(3), 3, 0, 12)  // irregular seed: expansion/panic arms
	f.Add(int64(4), 0, 3, 0)   // NaN/Inf weights, clamped-tiny ts (Taylor branch)
	f.Add(int64(6), 0, 0, 2)   // hostile-laced seeds, clamped-huge ts (direct branch)
	f.Add(int64(8), 0, 0, 0)   // units==1 && batch==1
	f.Add(int64(13), 1, 4, 2)  // NaN/Inf inputs, unroll, out-of-order loss
	f.Add(int64(7), 2, 0, 0)   // chained topology
	f.Add(int64(5), 3, 0, 8)   // non-broadcastable seed: panic parity
	f.Add(int64(11), 3, 0, 14) // rank-3 seed: panic parity
	f.Add(int64(9), 0, 5, 0)   // saturating sigma: the signed-zero corner
	f.Add(int64(2147483647), 0, 0, 0)

	f.Fuzz(func(t *testing.T, seed int64, scenIn, payloadIn, seedSelIn int) {
		rng := rand.New(rand.NewSource(seed))
		units := 1 + rng.Intn(8)
		inDim := 1 + rng.Intn(4)
		batch := 1 + rng.Intn(4)
		scen := fuzzPosMod(scenIn, 4)
		payload := fuzzPosMod(payloadIn, 6)
		if scen == 2 {
			inDim = units // the chain feeds a [batch, units] output back as input
		}

		// Wiring: zero masks / random sparse / fully connected.
		var w *Wiring
		switch rng.Intn(4) {
		case 0:
			w = RandomSparse(inDim, units, 0, 0, rng)
		case 1, 2:
			w = RandomSparse(inDim, units, float32(1+rng.Intn(10))/10, float32(1+rng.Intn(10))/10, rng)
		}

		// ts: normal / clamped tiny (the exprel Taylor branch) / 1.0 /
		// clamped huge (direct, saturated) / exprel boundary — all
		// positive and finite, per Step's contract.
		var ts float64
		switch rng.Intn(5) {
		case 0:
			ts = 0.05 + 10*rng.Float64()
		case 1:
			ts = math.Pow(10, -45+43*rng.Float64())
		case 2:
			ts = 1.0
		case 3:
			ts = math.Pow(10, 2+298*rng.Float64())
		default:
			ts = 1e-2 * (0.2 + 2*rng.Float64()) // B straddles cfcExprelThreshold per element
		}

		cell := NewCfC(inDim, units, w, rng)
		x := autograd.Var(tensor.Uniform(rng, -1, 1, batch, inDim))
		h := autograd.Var(tensor.Uniform(rng, -1, 1, batch, units))

		// Payload classes: the cell and inputs are rewritten in place
		// before either path runs, so both see identical values.
		switch payload {
		case 1: // signed zeros in x and h
			for i := range x.Data.Data {
				x.Data.Data[i] = math.Float32frombits([]uint32{0, 0x80000000}[rng.Intn(2)])
			}
			for i := range h.Data.Data {
				h.Data.Data[i] = math.Float32frombits([]uint32{0, 0x80000000}[rng.Intn(2)])
			}
		case 2: // extreme finite weights
			for _, p := range cell.Parameters() {
				for i := range p.Data.Data {
					p.Data.Data[i] = -1e30 + 2e30*rng.Float32()
				}
			}
		case 3: // NaN/Inf payloads in the synaptic weights
			for _, pi := range []int{3, 4, 5, 6, 7, 8} {
				fuzzFillHostile(rng, cell.Parameters()[pi].Data.Data, 1.0/6)
			}
		case 4: // NaN/Inf payloads in x and h
			fuzzFillHostile(rng, x.Data.Data, 1.0/6)
			fuzzFillHostile(rng, h.Data.Data, 1.0/6)
		case 5: // saturating sigma: sigmoid hits exactly 1, 1-s is +0 (the
			// signed-zero row-accumulator corner)
			for i := range cell.sigma.Data.Data {
				cell.sigma.Data.Data[i] = 1e37
			}
		}

		// The scenario closes over every draw (both paths must see identical
		// constants) and returns the forward bits to compare.
		var extra map[string]*autograd.Variable
		var body func(step fuzzLTCStepFn) map[string][]uint32
		switch scen {
		case 0: // seeded single step
			extra = map[string]*autograd.Variable{"x": x, "h": h}
			soT := tensor.Uniform(rng, -1, 1, batch, units)
			svT := tensor.Uniform(rng, -1, 1, batch, units)
			switch fuzzPosMod(seedSelIn, 3) {
			case 1: // signed-zero seeds
				for i := range soT.Data {
					soT.Data[i] = math.Float32frombits([]uint32{0, 0x80000000}[rng.Intn(2)])
					svT.Data[i] = math.Float32frombits([]uint32{0, 0x80000000}[rng.Intn(2)])
				}
			case 2: // hostile-laced seeds
				fuzzFillHostile(rng, soT.Data, 0.2)
				fuzzFillHostile(rng, svT.Data, 0.2)
			}
			body = func(step fuzzLTCStepFn) map[string][]uint32 {
				out, v := step(x, h, ts)
				autograd.Add(
					autograd.SumAll(autograd.Hadamard(out, autograd.Const(soT))),
					autograd.SumAll(autograd.Hadamard(v, autograd.Const(svT)))).Backward()
				return map[string][]uint32{"out": dataBits(out.Data), "state": dataBits(v.Data)}
			}
		case 1: // short unroll under the five loss shapes
			T := 2 + rng.Intn(3)
			xs := make([]*autograd.Variable, T)
			extra = map[string]*autograd.Variable{}
			for i := range xs {
				xs[i] = autograd.Var(tensor.Uniform(rng, -1, 1, batch, inDim))
				extra["x"+itoa(i)] = xs[i]
			}
			h0 := autograd.Var(tensor.Uniform(rng, -1, 1, batch, units))
			extra["h0"] = h0
			seeds := make([]*tensor.Tensor, T+1)
			for i := range seeds {
				seeds[i] = tensor.Uniform(rng, -1, 1, batch, units)
			}
			kind := fuzzPosMod(seedSelIn, 5)
			body = func(step fuzzLTCStepFn) map[string][]uint32 {
				hh := h0
				ys := make([]*autograd.Variable, T)
				fwd := make(map[string][]uint32, T+1)
				for i, xi := range xs {
					ys[i], hh = step(xi, hh, ts)
					fwd["y"+itoa(i)] = dataBits(ys[i].Data)
				}
				fwd["hN"] = dataBits(hh.Data)
				ltcUnrollLoss(kind, ys, hh, seeds).Backward()
				return fwd
			}
		case 2: // chained: step 2's input is a graph function of step 1's output
			extra = map[string]*autograd.Variable{"x0": x, "h0": h}
			c1 := tensor.Uniform(rng, -1, 1, 1, units)
			c2 := tensor.Uniform(rng, -1, 1, batch, units)
			sv := tensor.Uniform(rng, -1, 1, batch, units)
			body = func(step fuzzLTCStepFn) map[string][]uint32 {
				y0, h1 := step(x, h, ts)
				x1c := autograd.Add(autograd.Hadamard(y0, autograd.Const(c1)), autograd.Const(c2))
				y1, h2 := step(x1c, h1, ts)
				autograd.SumAll(autograd.Hadamard(h2, autograd.Const(sv))).Backward()
				return map[string][]uint32{
					"y0": dataBits(y0.Data), "h1": dataBits(h1.Data),
					"y1": dataBits(y1.Data), "h2": dataBits(h2.Data),
				}
			}
		default: // irregular root seed on the state node (panic parity in scope)
			extra = map[string]*autograd.Variable{"x": x, "h": h}
			cand := [][]int{
				{1}, {1, 1}, {batch, units}, {1, units}, {batch, 1},
				{batch * units}, {batch*units + 1},
				{batch + 1, units}, {1, units + 1}, {batch, units + 1},
				{1, 2}, {2, 1}, {3, 2}, {2, 3}, {batch, units, 1},
			}
			sh := cand[fuzzPosMod(seedSelIn, len(cand))]
			n := 1
			for _, d := range sh {
				n *= d
			}
			flat := make([]float32, n)
			for i := range flat {
				flat[i] = -1 + 2*rng.Float32()
			}
			if rng.Intn(2) == 0 {
				fuzzFillHostile(rng, flat, 0.2)
			}
			body = func(step fuzzLTCStepFn) map[string][]uint32 {
				out, v := step(x, h, ts)
				v.Grad = tensor.FromData(append([]float32(nil), flat...), sh...)
				v.Backward()
				return map[string][]uint32{"out": dataBits(out.Data), "state": dataBits(v.Data)}
			}
		}

		// One path per run: identical draws, ZeroGrad between. A panic
		// surfaces as its message; parity requires the SAME message from
		// both paths (same point, same text) or identical bits throughout.
		runPath := func(step fuzzLTCStepFn) (fwd map[string][]uint32, grads map[string]fuzzGradSnap, msg string) {
			defer func() {
				if r := recover(); r != nil {
					msg = fmt.Sprint(r)
				}
			}()
			fuzzZeroAllCfC(cell, extra)
			fwd = body(step)
			grads = fuzzSnapGradsCfC(cell, extra)
			return fwd, grads, ""
		}
		fwdF, gradsF, mF := runPath(cell.Step)
		fwdL, gradsL, mL := runPath(boundLegacyCfC(cell))
		if mF != mL {
			t.Fatalf("panic parity: fused %q, legacy %q", mF, mL)
		}
		if mF != "" {
			return // both panicked with the identical message: parity holds
		}
		for name, want := range fwdL {
			fuzzCmpBitsNaN(t, "fwd "+name, fwdF[name], want)
		}
		fuzzCmpSnaps(t, gradsF, gradsL)
	})
}
