package optimizer

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"math/rand"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/nn"
	"github.com/qorm/LNN/serialize"
	"github.com/qorm/LNN/tensor"
)

// --- Helpers ---

// sameBits reports whether a and b hold bit-identical float32s (NaN and
// -0 included, per math.Float32bits).
func sameBits(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}

// paramsBits flattens every parameter's Data into float32 bit patterns.
func paramsBits(params []*autograd.Variable) []uint32 {
	out := make([]uint32, 0, 8)
	for _, p := range params {
		for _, v := range p.Data.Data {
			out = append(out, math.Float32bits(v))
		}
	}
	return out
}

func equalBits(a, b []uint32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// failWriter always fails: exercises the write path's error reporting.
type failWriter struct{}

func (failWriter) Write([]byte) (int, error) { return 0, errors.New("disk full") }

// limitWriter fails after n bytes, so tests can fail the blob phase
// separately from the header phase.
type limitWriter struct {
	buf bytes.Buffer
	n   int
}

func (lw *limitWriter) Write(p []byte) (int, error) {
	if lw.buf.Len()+len(p) > lw.n {
		return 0, errors.New("quota exhausted")
	}
	return lw.buf.Write(p)
}

// stateAllocBytesPerRun returns the average heap bytes allocated per call
// of f (TotalAlloc delta over n runs). testing.AllocsPerRun counts
// allocations but not their size, and the threat a hostile size claim
// bills is exactly size — the nn/save_test.go allocBytesPerRun pattern,
// self-contained here.
func stateAllocBytesPerRun(n int, f func()) uint64 {
	var m0, m1 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m0)
	for i := 0; i < n; i++ {
		f()
	}
	runtime.ReadMemStats(&m1)
	return (m1.TotalAlloc - m0.TotalAlloc) / uint64(n)
}

// --- 1. Resume bit-exactness: the core acceptance test ---

// stateFitData builds a fixed deterministic regression dataset (y = 2x+1),
// identical for every call with the same seed, so both runs of a resume
// test see bit-identical inputs and therefore bit-identical gradients.
func stateFitData(seed int64) (x, y *autograd.Variable) {
	rng := rand.New(rand.NewSource(seed))
	const n = 32
	xs, ys := make([]float32, n), make([]float32, n)
	for i := range xs {
		xs[i] = rng.Float32()*2 - 1
		ys[i] = 2*xs[i] + 1
	}
	return autograd.Const(tensor.FromData(xs, n, 1)), autograd.Const(tensor.FromData(ys, n, 1))
}

// stateTrainStep runs one four-phase training iteration on the linear
// layer and returns the loss measured before the update.
func stateTrainStep(l *nn.Linear, x, y *autograd.Variable, o Optimizer) float32 {
	pred := l.Forward(x)
	diff := autograd.Sub(pred, y)
	loss := autograd.MeanAll(autograd.Hadamard(diff, diff))
	v := loss.Value()
	params := l.Parameters()
	for _, p := range params {
		p.ZeroGrad()
	}
	loss.Backward()
	o.Step(params)
	return v
}

// stateResumeBitExact drives the acceptance scenario: train a linear model
// for split steps, checkpoint model (nn.SaveLinear) and optimizer state
// (SaveState), load both into fresh objects, and train for the remaining
// steps. The resumed loss trajectory and per-parameter bit trajectory of
// steps split+1..total must equal an uninterrupted total-step run bit for
// bit — for every optimizer kind.
func stateResumeBitExact(t *testing.T, name string, newOpt func() Optimizer) {
	t.Helper()
	const total, split = 100, 50

	// Control: one optimizer, one model, no interruption.
	control := nn.NewLinear(1, 1, rand.New(rand.NewSource(42)))
	xc, yc := stateFitData(7)
	optc := newOpt()
	controlLoss := make([]float32, 0, total)
	controlTraj := make([][]uint32, 0, total)
	for i := 0; i < total; i++ {
		controlLoss = append(controlLoss, stateTrainStep(control, xc, yc, optc))
		controlTraj = append(controlTraj, paramsBits(control.Parameters()))
	}

	// Resume: identical construction, interrupted at `split` by a
	// save/load cycle through fresh objects.
	resumed := nn.NewLinear(1, 1, rand.New(rand.NewSource(42)))
	xr, yr := stateFitData(7)
	optr := newOpt()
	resumedLoss := make([]float32, 0, total)
	resumedTraj := make([][]uint32, 0, total)
	for i := 0; i < split; i++ {
		resumedLoss = append(resumedLoss, stateTrainStep(resumed, xr, yr, optr))
		resumedTraj = append(resumedTraj, paramsBits(resumed.Parameters()))
	}

	// The first halves must already agree: both runs consumed identical
	// gradients through identical arithmetic. If this fails the harness,
	// not the persistence, is at fault.
	for i := 0; i < split; i++ {
		if math.Float32bits(resumedLoss[i]) != math.Float32bits(controlLoss[i]) {
			t.Fatalf("%s: harness nondeterminism: first-half loss %d differs (resumed %v vs control %v)",
				name, i, resumedLoss[i], controlLoss[i])
		}
		if !equalBits(resumedTraj[i], controlTraj[i]) {
			t.Fatalf("%s: harness nondeterminism: first-half trajectory %d differs", name, i)
		}
	}

	// Checkpoint: model parameters via nn.SaveLinear, optimizer state via
	// SaveState.
	var modelBuf, stateBuf bytes.Buffer
	if err := nn.SaveLinear(&modelBuf, resumed); err != nil {
		t.Fatalf("%s: SaveLinear: %v", name, err)
	}
	if err := SaveState(&stateBuf, optr, resumed.Parameters()); err != nil {
		t.Fatalf("%s: SaveState: %v", name, err)
	}

	// Fresh objects: a new layer from the model stream (built with a
	// throwaway seed inside LoadLinear — the seed is irrelevant, Load
	// overwrites every field) and a new optimizer from the state stream.
	loaded, err := nn.LoadLinear(bytes.NewReader(modelBuf.Bytes()))
	if err != nil {
		t.Fatalf("%s: LoadLinear: %v", name, err)
	}
	optl := newOpt()
	if err := LoadState(bytes.NewReader(stateBuf.Bytes()), optl, loaded.Parameters()); err != nil {
		t.Fatalf("%s: LoadState: %v", name, err)
	}

	for i := split; i < total; i++ {
		resumedLoss = append(resumedLoss, stateTrainStep(loaded, xr, yr, optl))
		resumedTraj = append(resumedTraj, paramsBits(loaded.Parameters()))
	}

	// The acceptance assertion: steps split+1..total, per parameter, bit
	// for bit — trajectory and loss alike.
	for i := split; i < total; i++ {
		if math.Float32bits(resumedLoss[i]) != math.Float32bits(controlLoss[i]) {
			t.Fatalf("%s: resumed loss[%d] = %v (%#08x), uninterrupted = %v (%#08x): optimizer state lost across checkpoint",
				name, i, resumedLoss[i], math.Float32bits(resumedLoss[i]), controlLoss[i], math.Float32bits(controlLoss[i]))
		}
		if !equalBits(resumedTraj[i], controlTraj[i]) {
			t.Fatalf("%s: resumed parameter trajectory at step %d differs from uninterrupted training bit for bit", name, i)
		}
	}
	t.Logf("%s: resume trajectory bit-exact over steps %d-%d; final loss %v (%#08x)",
		name, split+1, total, controlLoss[total-1], math.Float32bits(controlLoss[total-1]))
}

func TestResumeBitExactAdam(t *testing.T) {
	stateResumeBitExact(t, "Adam", func() Optimizer { return NewAdamDefault(0.1) })
}

func TestResumeBitExactMomentum(t *testing.T) {
	stateResumeBitExact(t, "Momentum", func() Optimizer { return NewMomentum(0.05, 0.9) })
}

func TestResumeBitExactSGD(t *testing.T) {
	stateResumeBitExact(t, "SGD", func() Optimizer { return NewSGD(0.1) })
}

// --- 2. Round-trip: internal state bit-exactness + behavioral equality ---

func TestRoundTripMomentumStateBits(t *testing.T) {
	o := NewMomentum(0.05, 0.9)
	p0 := param([]float32{1, -2}, []float32{0.5, 3})
	p1 := param([]float32{7}, []float32{-1})
	params := []*autograd.Variable{p0, p1}
	grads := [][]float32{{0.25, -0.75}, {2}}
	for i := 0; i < 4; i++ {
		o.Step(params)
		setGrad(p0, grads[0])
		setGrad(p1, grads[1])
	}
	var buf bytes.Buffer
	if err := SaveState(&buf, o, params); err != nil {
		t.Fatal(err)
	}
	o2 := NewMomentum(0.05, 0.9)
	if err := LoadState(bytes.NewReader(buf.Bytes()), o2, params); err != nil {
		t.Fatal(err)
	}
	// Internal state, bit for bit.
	if len(o2.velocity) != 2 {
		t.Fatalf("loaded velocity map has %d entries, want 2", len(o2.velocity))
	}
	for _, p := range params {
		if !sameBits(o2.velocity[p], o.velocity[p]) {
			t.Errorf("velocity of %p differs after round trip", p)
		}
	}
	// Behavioral: load the same stream into a third optimizer keyed by
	// pre-step clones (state attaches to the pointers given to LoadState),
	// then step both: the outputs must agree bit for bit.
	p0b := param(p0.Data.Data, []float32{1.5, -4})
	p1b := param(p1.Data.Data, []float32{0.125})
	o3 := NewMomentum(0.05, 0.9)
	if err := LoadState(bytes.NewReader(buf.Bytes()), o3, []*autograd.Variable{p0b, p1b}); err != nil {
		t.Fatal(err)
	}
	setGrad(p0, []float32{1.5, -4})
	setGrad(p1, []float32{0.125})
	o.Step([]*autograd.Variable{p0, p1})
	o3.Step([]*autograd.Variable{p0b, p1b})
	if !sameBits(p0b.Data.Data, p0.Data.Data) || !sameBits(p1b.Data.Data, p1.Data.Data) {
		t.Error("Step output differs between loaded and original optimizer")
	}
}

func TestRoundTripAdamStateBits(t *testing.T) {
	o := NewAdam(0.01, 0.85, 0.99, 1e-7)
	p0 := param([]float32{0.5, -1.5, 2}, []float32{1, -1, 0.25})
	p1 := param([]float32{-3, 4}, []float32{0.75, -2})
	params := []*autograd.Variable{p0, p1}
	for i := 0; i < 5; i++ {
		o.Step(params)
		setGrad(p0, []float32{float32(i) - 2, 0.5, -0.125})
		setGrad(p1, []float32{1.5, float32(-i)})
	}
	var buf bytes.Buffer
	if err := SaveState(&buf, o, params); err != nil {
		t.Fatal(err)
	}
	o2 := NewAdam(0.01, 0.85, 0.99, 1e-7)
	if err := LoadState(bytes.NewReader(buf.Bytes()), o2, params); err != nil {
		t.Fatal(err)
	}
	// Internal state: m, v bit for bit, t exact, pow1/pow2 bit for bit.
	for _, p := range params {
		a, b := o.state[p], o2.state[p]
		if b == nil {
			t.Fatalf("state of %p missing after round trip", p)
		}
		if !sameBits(a.m, b.m) || !sameBits(a.v, b.v) {
			t.Errorf("moments of %p differ after round trip", p)
		}
		if a.t != b.t {
			t.Errorf("update count of %p: got %d, want %d", p, b.t, a.t)
		}
		if math.Float32bits(a.pow1) != math.Float32bits(b.pow1) || math.Float32bits(a.pow2) != math.Float32bits(b.pow2) {
			t.Errorf("bias-correction powers of %p differ after round trip", p)
		}
	}
	// Behavioral: load the same stream into a third optimizer keyed by
	// pre-step clones, then step both: outputs must agree bit for bit.
	p0b := param(p0.Data.Data, []float32{3, 3, 3})
	p1b := param(p1.Data.Data, []float32{-0.5, 0.5})
	o3 := NewAdam(0.01, 0.85, 0.99, 1e-7)
	if err := LoadState(bytes.NewReader(buf.Bytes()), o3, []*autograd.Variable{p0b, p1b}); err != nil {
		t.Fatal(err)
	}
	setGrad(p0, []float32{3, 3, 3})
	setGrad(p1, []float32{-0.5, 0.5})
	o.Step([]*autograd.Variable{p0, p1})
	o3.Step([]*autograd.Variable{p0b, p1b})
	if !sameBits(p0b.Data.Data, p0.Data.Data) || !sameBits(p1b.Data.Data, p1.Data.Data) {
		t.Error("Step output differs between loaded and original optimizer")
	}
}

func TestRoundTripSGDIdentity(t *testing.T) {
	o := NewSGD(0.1)
	p := param([]float32{1, 2}, []float32{4, 8})
	params := []*autograd.Variable{p}
	o.Step(params)
	var buf bytes.Buffer
	if err := SaveState(&buf, o, params); err != nil {
		t.Fatal(err)
	}
	// The SGD stream is header + empty blob: 10 header bytes + a
	// zero-tensor serialize stream (4 magic + 1 version + 4 count).
	if n := buf.Len(); n != 19 {
		t.Errorf("SGD state stream is %d bytes, want 19 (self-framing, no payload)", n)
	}
	o2 := NewSGD(0.1)
	if err := LoadState(bytes.NewReader(buf.Bytes()), o2, params); err != nil {
		t.Fatal(err)
	}
	setGrad(p, []float32{1, 1})
	pb := param(p.Data.Data, []float32{1, 1})
	o.Step([]*autograd.Variable{p})
	o2.Step([]*autograd.Variable{pb})
	if !sameBits(pb.Data.Data, p.Data.Data) {
		t.Error("Step output differs after SGD state round trip")
	}
}

// --- 3. Adversarial streams ---

// stateStreamHeader hand-builds a state stream header.
func stateStreamHeader(kind byte, count uint32) []byte {
	b := make([]byte, 10)
	copy(b, []byte("LNO1"))
	b[4] = 1 // version
	b[5] = kind
	binary.LittleEndian.PutUint32(b[6:], count)
	return b
}

func TestLoadStateRejectsCrossKind(t *testing.T) {
	p := param([]float32{1, 2}, []float32{3, 4})
	params := []*autograd.Variable{p}
	savers := map[string]Optimizer{
		"SGD":      NewSGD(0.1),
		"Momentum": NewMomentum(0.1, 0.9),
		"Adam":     NewAdamDefault(0.1),
	}
	streams := map[string][]byte{}
	for name, o := range savers {
		o.Step(params) // give Momentum/Adam something to save
		setGrad(p, []float32{3, 4})
		var buf bytes.Buffer
		if err := SaveState(&buf, o, params); err != nil {
			t.Fatalf("saving %s state: %v", name, err)
		}
		streams[name] = buf.Bytes()
	}
	// All six cross-loadings must fail with the kind-mismatch error.
	pairs := [][2]string{
		{"SGD", "Momentum"}, {"SGD", "Adam"},
		{"Momentum", "SGD"}, {"Momentum", "Adam"},
		{"Adam", "SGD"}, {"Adam", "Momentum"},
	}
	for _, pr := range pairs {
		loaders := map[string]Optimizer{
			"SGD":      NewSGD(0.1),
			"Momentum": NewMomentum(0.1, 0.9),
			"Adam":     NewAdamDefault(0.1),
		}
		err := LoadState(bytes.NewReader(streams[pr[0]]), loaders[pr[1]], params)
		if err == nil || !strings.Contains(err.Error(), "does not match optimizer kind") {
			t.Errorf("%s stream into %s: got %v, want a kind-mismatch error", pr[0], pr[1], err)
		}
	}
}

func TestLoadStateRejectsHeaderCorruption(t *testing.T) {
	p := param([]float32{1}, []float32{2})
	params := []*autograd.Variable{p}
	o := NewMomentum(0.1, 0.9)
	o.Step(params)
	var buf bytes.Buffer
	if err := SaveState(&buf, o, params); err != nil {
		t.Fatal(err)
	}
	valid := buf.Bytes()
	mutate := func(pos int, v byte) []byte {
		b := append([]byte(nil), valid...)
		b[pos] = v
		return b
	}

	cases := []struct {
		name    string
		stream  []byte
		wantSub string
	}{
		{"bad magic", mutate(0, 'X'), "bad magic"},
		{"version 0", mutate(4, 0), "unsupported state format version 0"},
		{"version 99", mutate(4, 99), "unsupported state format version 99"},
		{"presence flag 2", func() []byte {
			// header(count=1) + presence=2; the blob is never reached
			b := stateStreamHeader(byte(kindMomentum), 1)
			return append(b, 2)
		}(), "presence flag 2 outside {0, 1}"},
	}
	for _, c := range cases {
		err := LoadState(bytes.NewReader(c.stream), NewMomentum(0.1, 0.9), params)
		if err == nil || !strings.Contains(err.Error(), c.wantSub) {
			t.Errorf("%s: got %v, want error containing %q", c.name, err, c.wantSub)
		}
	}

	// Version skew messages carry the direction (serialize's style).
	err := LoadState(bytes.NewReader(mutate(4, 99)), NewMomentum(0.1, 0.9), params)
	if !strings.Contains(err.Error(), "newer version") {
		t.Errorf("version 99: %v should say the stream comes from a newer version", err)
	}
	err = LoadState(bytes.NewReader(mutate(4, 0)), NewMomentum(0.1, 0.9), params)
	if !strings.Contains(err.Error(), "corrupt or forged") {
		t.Errorf("version 0: %v should say corrupt or forged", err)
	}

	// Count mismatch: the stream carries 1 record, the destination has 2.
	p2 := param([]float32{5}, nil)
	err = LoadState(bytes.NewReader(valid), NewMomentum(0.1, 0.9), []*autograd.Variable{p, p2})
	if err == nil || !strings.Contains(err.Error(), "1 parameter records") || !strings.Contains(err.Error(), "2 parameters") {
		t.Errorf("count mismatch: got %v, want both counts named", err)
	}
	// And the reverse: count patched to 2, destination has 1.
	err = LoadState(bytes.NewReader(mutate(6, 2)), NewMomentum(0.1, 0.9), params)
	if err == nil || !strings.Contains(err.Error(), "2 parameter records") {
		t.Errorf("patched count: got %v, want a count mismatch", err)
	}
}

func TestLoadStateRejectsShapeMismatchWithoutSideEffects(t *testing.T) {
	// Source: state for a 2-element parameter.
	src := param([]float32{1, 2}, []float32{3, 4})
	mo := NewMomentum(0.1, 0.9)
	mo.Step([]*autograd.Variable{src})
	adam := NewAdamDefault(0.1)
	adam.Step([]*autograd.Variable{src})
	var momBuf, adamBuf bytes.Buffer
	if err := SaveState(&momBuf, mo, []*autograd.Variable{src}); err != nil {
		t.Fatal(err)
	}
	if err := SaveState(&adamBuf, adam, []*autograd.Variable{src}); err != nil {
		t.Fatal(err)
	}

	t.Run("Momentum", func(t *testing.T) {
		// Destination: a 3-element parameter with an existing velocity.
		dst := param([]float32{9, 9, 9}, []float32{1, 1, 1})
		dm := NewMomentum(0.1, 0.9)
		dm.Step([]*autograd.Variable{dst})
		setGrad(dst, []float32{1, 1, 1})
		before := append([]float32(nil), dm.velocity[dst]...)
		err := LoadState(bytes.NewReader(momBuf.Bytes()), dm, []*autograd.Variable{dst})
		if err == nil || !strings.Contains(err.Error(), "velocity shape mismatch") {
			t.Fatalf("got %v, want a velocity shape mismatch", err)
		}
		if !sameBits(dm.velocity[dst], before) {
			t.Error("failed load modified the destination velocity")
		}
	})
	t.Run("Adam", func(t *testing.T) {
		dst := param([]float32{9, 9, 9}, []float32{1, 1, 1})
		da := NewAdamDefault(0.1)
		da.Step([]*autograd.Variable{dst})
		setGrad(dst, []float32{1, 1, 1})
		stBefore := da.state[dst]
		mBefore := append([]float32(nil), stBefore.m...)
		vBefore := append([]float32(nil), stBefore.v...)
		err := LoadState(bytes.NewReader(adamBuf.Bytes()), da, []*autograd.Variable{dst})
		if err == nil || !strings.Contains(err.Error(), "moment m shape mismatch") {
			t.Fatalf("got %v, want a moment m shape mismatch", err)
		}
		stAfter := da.state[dst]
		if stAfter != stBefore || stAfter.t != 1 ||
			math.Float32bits(stAfter.pow1) != math.Float32bits(stBefore.pow1) ||
			!sameBits(stAfter.m, mBefore) || !sameBits(stAfter.v, vBefore) {
			t.Error("failed load modified the destination Adam state")
		}
	})
	t.Run("AdamLateV", func(t *testing.T) {
		// A legitimate stream always saves m and v with the same shape,
		// so reaching the v branch takes a forged blob: two parameters
		// whose first fits entirely and whose second m fits but v does
		// not. Validate-all-then-apply must reach parameter 1's v and
		// leave both destination entries untouched.
		f32le := func(v float32) []byte {
			var b [4]byte
			binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
			return b[:]
		}
		raw := stateStreamHeader(byte(kindAdam), 2)
		for i := 0; i < 2; i++ {
			raw = append(raw, 1) // present
			var tb [4]byte
			binary.LittleEndian.PutUint32(tb[:], 1) // t = 1
			raw = append(raw, tb[:]...)
			raw = append(raw, f32le(0.9)...)   // pow1 = Beta1^1
			raw = append(raw, f32le(0.999)...) // pow2 = Beta2^1
		}
		var blob bytes.Buffer
		if err := serialize.WriteTensors(&blob, []*tensor.Tensor{
			{Shape: []int{2}, Data: []float32{1, 2}},    // m0: fits
			{Shape: []int{2}, Data: []float32{3, 4}},    // v0: fits
			{Shape: []int{2}, Data: []float32{5, 6}},    // m1: fits
			{Shape: []int{3}, Data: []float32{7, 8, 9}}, // v1: mismatch
		}); err != nil {
			t.Fatal(err)
		}
		raw = append(raw, blob.Bytes()...)

		d0 := param([]float32{0, 0}, []float32{1, 1})
		d1 := param([]float32{0, 0}, []float32{1, 1})
		dst2 := NewAdamDefault(0.1)
		dst2.Step([]*autograd.Variable{d0, d1})
		m0 := append([]float32(nil), dst2.state[d0].m...)
		m1 := append([]float32(nil), dst2.state[d1].m...)
		err := LoadState(bytes.NewReader(raw), dst2, []*autograd.Variable{d0, d1})
		if err == nil || !strings.Contains(err.Error(), "parameter 1") || !strings.Contains(err.Error(), "moment v shape mismatch") {
			t.Fatalf("got %v, want parameter 1 moment v shape mismatch", err)
		}
		if !sameBits(dst2.state[d0].m, m0) || !sameBits(dst2.state[d1].m, m1) {
			t.Error("failed load modified the destination state")
		}
	})
}

func TestLoadStateRejectsInconsistentAdamCounters(t *testing.T) {
	p := param([]float32{1, 2}, []float32{3, 4})
	params := []*autograd.Variable{p}
	o := NewAdamDefault(0.1)
	for i := 0; i < 3; i++ {
		o.Step(params)
		setGrad(p, []float32{3, 4})
	}
	var buf bytes.Buffer
	if err := SaveState(&buf, o, params); err != nil {
		t.Fatal(err)
	}
	valid := buf.Bytes()
	// Layout: 10 header + presence(1) + t(4) + pow1(4) + pow2(4) + blob.
	// pow1 occupies bytes 15..18, pow2 bytes 19..22.
	flip := func(pos int) []byte {
		b := append([]byte(nil), valid...)
		b[pos] ^= 0x01
		return b
	}
	err := LoadState(bytes.NewReader(flip(15)), NewAdamDefault(0.1), params)
	if err == nil || !strings.Contains(err.Error(), "pow1") {
		t.Errorf("flipped pow1: got %v, want a pow1 consistency error", err)
	}
	err = LoadState(bytes.NewReader(flip(19)), NewAdamDefault(0.1), params)
	if err == nil || !strings.Contains(err.Error(), "pow2") {
		t.Errorf("flipped pow2: got %v, want a pow2 consistency error", err)
	}
	// Same stream, optimizer built with different betas: the counters no
	// longer reproduce, so the load must fail as corruption-or-skew
	// rather than resume with inconsistent bias correction.
	err = LoadState(bytes.NewReader(valid), NewAdam(0.1, 0.5, 0.999, 1e-8), params)
	if err == nil || !strings.Contains(err.Error(), "different Adam hyperparameters") {
		t.Errorf("beta skew: got %v, want a hyperparameter-skew error", err)
	}
	// The update-count load limit fires before any blob parsing.
	big := stateStreamHeader(byte(kindAdam), 1)
	big = append(big, 1) // present
	var tBytes [4]byte
	binary.LittleEndian.PutUint32(tBytes[:], math.MaxUint32)
	big = append(big, tBytes[:]...)
	big = append(big, valid[15:23]...) // pow1, pow2: never reached
	err = LoadState(bytes.NewReader(big), NewAdamDefault(0.1), params)
	if err == nil || !strings.Contains(err.Error(), "exceeds the load limit") {
		t.Errorf("huge t: got %v, want the load-limit error", err)
	}
}

func TestLoadStateRejectsTruncatedStreams(t *testing.T) {
	p := param([]float32{1, 2}, []float32{3, 4})
	params := []*autograd.Variable{p}
	o := NewAdamDefault(0.1)
	o.Step(params)
	var buf bytes.Buffer
	if err := SaveState(&buf, o, params); err != nil {
		t.Fatal(err)
	}
	full := buf.Bytes()
	cuts := map[string]int{
		"mid-magic":  2,             // inside the 4-byte magic
		"mid-kind":   5,             // magic + version read, kind byte missing
		"mid-count":  7,             // count field cut in half
		"mid-tensor": len(full) - 3, // inside the blob's float payload
	}
	loaders := func() Optimizer { return NewAdamDefault(0.1) }
	for name, n := range cuts {
		err := LoadState(bytes.NewReader(full[:n]), loaders(), params)
		if err == nil {
			t.Errorf("%s: no error on a %d-byte truncation", name, n)
			continue
		}
		if !errors.Is(err, io.ErrUnexpectedEOF) {
			t.Errorf("%s: got %v, want errors.Is(io.ErrUnexpectedEOF)", name, err)
		}
	}
	// Trailing garbage after a complete stream is corruption.
	err := LoadState(bytes.NewReader(append(append([]byte(nil), full...), 0x00)), loaders(), params)
	if err == nil || !strings.Contains(err.Error(), "trailing") {
		t.Errorf("trailing byte: got %v, want a trailing-data error", err)
	}
}

// noopOpt satisfies Optimizer but is not one of the three persisted kinds.
type noopOpt struct{}

func (noopOpt) Step([]*autograd.Variable) {}

func TestSaveLoadStateRejectUnsupportedOptimizers(t *testing.T) {
	var o Optimizer = noopOpt{}
	p := param([]float32{1}, []float32{1})
	params := []*autograd.Variable{p}
	var buf bytes.Buffer
	if err := SaveState(&buf, o, params); err == nil || !strings.Contains(err.Error(), "unsupported optimizer type") {
		t.Errorf("SaveState(noop): got %v, want unsupported-type error", err)
	}
	if err := LoadState(bytes.NewReader(nil), o, params); err == nil || !strings.Contains(err.Error(), "unsupported optimizer type") {
		t.Errorf("LoadState(noop): got %v, want unsupported-type error", err)
	}
	if err := SaveState(&buf, nil, params); err == nil || !strings.Contains(err.Error(), "unsupported optimizer type") {
		t.Errorf("SaveState(nil): got %v, want unsupported-type error", err)
	}
	// Nil or data-less parameters are valued errors on both paths.
	if err := SaveState(&buf, NewSGD(0.1), []*autograd.Variable{nil}); err == nil || !strings.Contains(err.Error(), "parameter 0 has no data") {
		t.Errorf("SaveState(nil param): got %v", err)
	}
	if err := LoadState(bytes.NewReader(nil), NewSGD(0.1), []*autograd.Variable{nil}); err == nil || !strings.Contains(err.Error(), "parameter 0 has no data") {
		t.Errorf("LoadState(nil param): got %v", err)
	}
}

func TestSaveStateReportsWriteErrors(t *testing.T) {
	p := param([]float32{1, 2}, []float32{3, 4})
	params := []*autograd.Variable{p}
	mo := NewMomentum(0.1, 0.9)
	mo.Step(params)
	// The header write fails.
	if err := SaveState(failWriter{}, mo, params); err == nil || !strings.Contains(err.Error(), "writing state header") {
		t.Errorf("header write failure: got %v", err)
	}
	// The header fits the quota but the blob does not.
	lw := &limitWriter{n: 12}
	if err := SaveState(lw, mo, params); err == nil {
		t.Error("blob write failure: no error")
	}
	// Adam's update count must fit the format's uint32 field.
	a := NewAdamDefault(0.1)
	a.Step(params)
	st := a.state[p]
	badT := int64(math.MaxUint32)
	badT++ // runtime arithmetic: portable across int widths (wraps negative on 32-bit)
	st.t = int(badT)
	var buf bytes.Buffer
	if err := SaveState(&buf, a, params); err == nil || !strings.Contains(err.Error(), "uint32") {
		t.Errorf("oversized t: got %v, want a uint32-fit error", err)
	}
	st.t = -1
	if err := SaveState(&buf, a, params); err == nil || !strings.Contains(err.Error(), "uint32") {
		t.Errorf("negative t: got %v, want a uint32-fit error", err)
	}
}

// errReader returns a plain I/O error: the read path must propagate it
// (not misread it as truncation, and certainly not panic).
type errReader struct{}

func (errReader) Read([]byte) (int, error) { return 0, errors.New("connection reset") }

func TestLoadStatePropagatesReaderErrors(t *testing.T) {
	p := param([]float32{1}, nil)
	err := LoadState(errReader{}, NewSGD(0.1), []*autograd.Variable{p})
	if err == nil || !strings.Contains(err.Error(), "connection reset") || errors.Is(err, io.ErrUnexpectedEOF) {
		t.Errorf("got %v, want the raw I/O error propagated", err)
	}
}

func TestLoadStateRejectsUnknownKindByte(t *testing.T) {
	p := param([]float32{1}, nil)
	params := []*autograd.Variable{p}
	raw := stateStreamHeader(7, 1) // no optimizer has kind 7
	err := LoadState(bytes.NewReader(raw), NewMomentum(0.1, 0.9), params)
	if err == nil || !strings.Contains(err.Error(), "kind 7 (unknown)") || !strings.Contains(err.Error(), "does not match optimizer kind") {
		t.Errorf("got %v, want a mismatch error naming kind 7 as unknown", err)
	}
}

func TestLoadStateRejectsMidRecordTruncation(t *testing.T) {
	p0 := param([]float32{1}, nil)
	p1 := param([]float32{2}, nil)
	params := []*autograd.Variable{p0, p1}
	// count = 2 but only one presence byte delivered.
	raw := stateStreamHeader(byte(kindMomentum), 2)
	raw = append(raw, 0)
	err := LoadState(bytes.NewReader(raw), NewMomentum(0.1, 0.9), params)
	if err == nil || !errors.Is(err, io.ErrUnexpectedEOF) || !strings.Contains(err.Error(), "parameter 1") {
		t.Errorf("mid-presence truncation: got %v, want ErrUnexpectedEOF at parameter 1", err)
	}
	// Adam: present flag delivered, counters cut in half.
	raw = stateStreamHeader(byte(kindAdam), 1)
	raw = append(raw, 1, 5, 0) // present, then 2 of the 4 t bytes
	err = LoadState(bytes.NewReader(raw), NewAdamDefault(0.1), []*autograd.Variable{p0})
	if err == nil || !errors.Is(err, io.ErrUnexpectedEOF) || !strings.Contains(err.Error(), "Adam counters") {
		t.Errorf("mid-counter truncation: got %v, want ErrUnexpectedEOF in the Adam counters", err)
	}
}

func TestLoadStateRejectsTensorCountMismatch(t *testing.T) {
	p := param([]float32{1}, nil)
	params := []*autograd.Variable{p}
	// Record section claims one present Momentum parameter, but the blob
	// carries zero tensors.
	raw := stateStreamHeader(byte(kindMomentum), 1)
	raw = append(raw, 1)
	var blob bytes.Buffer
	if err := serialize.WriteTensors(&blob, nil); err != nil {
		t.Fatal(err)
	}
	raw = append(raw, blob.Bytes()...)
	err := LoadState(bytes.NewReader(raw), NewMomentum(0.1, 0.9), params)
	if err == nil || !strings.Contains(err.Error(), "holds 0 tensors") || !strings.Contains(err.Error(), "want 1") {
		t.Errorf("got %v, want a blob tensor-count mismatch", err)
	}
}

// Loading into optimizers built as struct literals (nil state maps) must
// work: LoadState initializes the maps rather than panicking on them.
func TestLoadStateInitializesZeroValueOptimizers(t *testing.T) {
	p := param([]float32{1, 2}, []float32{3, 4})
	params := []*autograd.Variable{p}
	src := NewMomentum(0.25, 0.5)
	src.Step(params)
	var buf bytes.Buffer
	if err := SaveState(&buf, src, params); err != nil {
		t.Fatal(err)
	}
	mom := &Momentum{LR: 0.25, Mu: 0.5} // nil velocity map
	if err := LoadState(bytes.NewReader(buf.Bytes()), mom, params); err != nil {
		t.Fatal(err)
	}
	if !sameBits(mom.velocity[p], src.velocity[p]) {
		t.Error("velocity not restored into the zero-value Momentum")
	}

	srcA := NewAdamDefault(0.1)
	srcA.Step(params)
	buf.Reset()
	if err := SaveState(&buf, srcA, params); err != nil {
		t.Fatal(err)
	}
	adam := &Adam{LR: 0.1, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8} // nil state map
	if err := LoadState(bytes.NewReader(buf.Bytes()), adam, params); err != nil {
		t.Fatal(err)
	}
	if !sameBits(adam.state[p].m, srcA.state[p].m) || adam.state[p].t != srcA.state[p].t {
		t.Error("state not restored into the zero-value Adam")
	}
}

// --- 4. Byte-budget gate: hostile claims must not allocate ---

func TestLoadStateHostileClaimsStayWithinByteBudget(t *testing.T) {
	dst := param([]float32{1}, nil)
	params := []*autograd.Variable{dst}

	t.Run("GiantDim", func(t *testing.T) {
		// Momentum header + present flag + a blob claiming one rank-1
		// tensor of 1<<62 elements and no payload: the claim is rejected
		// inside serialize before any buffer is allocated.
		raw := stateStreamHeader(byte(kindMomentum), 1)
		raw = append(raw, 1) // present
		blob := []byte{'L', 'N', 'N', 'S', 1}
		blob = append(blob, 1, 0, 0, 0) // count = 1 tensor
		blob = append(blob, 1)          // rank = 1
		var dim [8]byte
		binary.LittleEndian.PutUint64(dim[:], 1<<62)
		blob = append(blob, dim[:]...)
		raw = append(raw, blob...)

		allocs := testing.AllocsPerRun(20, func() {
			if err := LoadState(bytes.NewReader(raw), NewMomentum(0.1, 0.9), params); err == nil {
				t.Fatal("hostile dimension accepted")
			}
		})
		if allocs > 50 {
			t.Errorf("rejection took %.0f allocations; the limit must fire before parsing", allocs)
		}
		if b := stateAllocBytesPerRun(20, func() { LoadState(bytes.NewReader(raw), NewMomentum(0.1, 0.9), params) }); b >= 1<<20 {
			t.Errorf("rejection allocated %d bytes per run; want < 1 MiB", b)
		}
	})
	t.Run("GiantDimTruncated", func(t *testing.T) {
		// A claim below serialize's element limit but far above the bytes
		// delivered: the known-length reader checks the payload against
		// the bytes actually remaining before allocating anything.
		raw := stateStreamHeader(byte(kindMomentum), 1)
		raw = append(raw, 1)
		blob := []byte{'L', 'N', 'N', 'S', 1}
		blob = append(blob, 1, 0, 0, 0)
		blob = append(blob, 1)
		var dim [8]byte
		binary.LittleEndian.PutUint64(dim[:], 1<<28) // 1 GiB of payload claimed
		blob = append(blob, dim[:]...)
		raw = append(raw, blob...) // zero data bytes delivered

		err := LoadState(bytes.NewReader(raw), NewMomentum(0.1, 0.9), params)
		if err == nil || !errors.Is(err, io.ErrUnexpectedEOF) {
			t.Fatalf("got %v, want ErrUnexpectedEOF", err)
		}
		if b := stateAllocBytesPerRun(20, func() { LoadState(bytes.NewReader(raw), NewMomentum(0.1, 0.9), params) }); b >= 1<<20 {
			t.Errorf("rejection allocated %d bytes per run; want < 1 MiB (claim: 1 GiB)", b)
		}
	})
	t.Run("GiantCount", func(t *testing.T) {
		// Header claims nearly 2^32 parameter records while the
		// destination has one parameter: refused before any record is
		// parsed, so the count never bills a staging slice.
		raw := stateStreamHeader(byte(kindAdam), math.MaxUint32)
		allocs := testing.AllocsPerRun(20, func() {
			if err := LoadState(bytes.NewReader(raw), NewAdamDefault(0.1), params); err == nil {
				t.Fatal("hostile count accepted")
			}
		})
		if allocs > 50 {
			t.Errorf("rejection took %.0f allocations; the count check must fire first", allocs)
		}
		if b := stateAllocBytesPerRun(20, func() { LoadState(bytes.NewReader(raw), NewAdamDefault(0.1), params) }); b >= 1<<20 {
			t.Errorf("rejection allocated %d bytes per run; want < 1 MiB", b)
		}
	})
}

// --- 5. Presence semantics: sparse velocity maps survive ---

func TestSparsePresenceRoundTrip(t *testing.T) {
	p0 := param([]float32{1}, []float32{2})
	p1 := param([]float32{3}, nil) // nil Grad: Step never gives it a velocity
	p2 := param([]float32{5}, []float32{6})
	params := []*autograd.Variable{p0, p1, p2}
	o := NewMomentum(0.25, 0.5)
	o.Step(params)
	if _, ok := o.velocity[p1]; ok {
		t.Fatal("test setup broken: p1 unexpectedly has a velocity")
	}

	var buf bytes.Buffer
	if err := SaveState(&buf, o, params); err != nil {
		t.Fatal(err)
	}
	o2 := NewMomentum(0.25, 0.5)
	if err := LoadState(bytes.NewReader(buf.Bytes()), o2, params); err != nil {
		t.Fatal(err)
	}
	if !sameBits(o2.velocity[p0], o.velocity[p0]) || !sameBits(o2.velocity[p2], o.velocity[p2]) {
		t.Error("present velocities lost in the round trip")
	}
	if _, ok := o2.velocity[p1]; ok {
		t.Error("absent velocity fabricated by the round trip: the load must not invent zero state")
	}
	// Behavioral proof: p1's first step under o2 equals a fresh
	// optimizer's first step (velocity starts from zero).
	setGrad(p1, []float32{8})
	o2.Step([]*autograd.Variable{p1})
	pf := param([]float32{3}, []float32{8})
	NewMomentum(0.25, 0.5).Step([]*autograd.Variable{pf})
	if !sameBits(p1.Data.Data, pf.Data.Data) {
		t.Errorf("sparse parameter's first update = %v, want fresh %v", p1.Data.Data, pf.Data.Data)
	}
}

// An absent record overwrites a pre-existing entry: after the load the
// optimizer's state for params is precisely what the stream describes.
func TestAbsentRecordDeletesExistingState(t *testing.T) {
	p0 := param([]float32{1}, []float32{2})
	p1 := param([]float32{3}, []float32{4})
	// Source: only p0 has state (p1 stepped with nil Grad).
	src := NewAdamDefault(0.1)
	src.Step([]*autograd.Variable{p0})
	var buf bytes.Buffer
	if err := SaveState(&buf, src, []*autograd.Variable{p0, p1}); err != nil {
		t.Fatal(err)
	}
	// Destination: both parameters already have state; a stale key for a
	// variable outside params is expected to survive the load.
	stale := param([]float32{9}, []float32{9})
	dst := NewAdamDefault(0.1)
	dst.Step([]*autograd.Variable{p0, p1, stale})
	if err := LoadState(bytes.NewReader(buf.Bytes()), dst, []*autograd.Variable{p0, p1}); err != nil {
		t.Fatal(err)
	}
	if !sameBits(dst.state[p0].m, src.state[p0].m) {
		t.Error("present record not restored")
	}
	if _, ok := dst.state[p1]; ok {
		t.Error("absent record did not delete the existing entry")
	}
	if _, ok := dst.state[stale]; !ok {
		t.Error("stale key for a variable outside params was deleted (documented to survive)")
	}
}

// --- 6. Writer freeze: identical state, identical bytes ---

func TestSaveStateDeterministic(t *testing.T) {
	build := func() ([]*autograd.Variable, []Optimizer) {
		p0 := param([]float32{0.5, -1.25}, []float32{1, 2})
		p1 := param([]float32{3}, []float32{-4})
		ps := []*autograd.Variable{p0, p1}
		opts := []Optimizer{NewSGD(0.1), NewMomentum(0.1, 0.9), NewAdamDefault(0.1)}
		for i := 0; i < 3; i++ {
			for _, o := range opts {
				o.Step(ps)
			}
			setGrad(p0, []float32{float32(i), -0.5})
			setGrad(p1, []float32{2.5})
		}
		return ps, opts
	}
	ps1, opts1 := build()
	ps2, opts2 := build()
	names := []string{"SGD", "Momentum", "Adam"}
	for i := range opts1 {
		var b1, b2 bytes.Buffer
		if err := SaveState(&b1, opts1[i], ps1); err != nil {
			t.Fatal(err)
		}
		if err := SaveState(&b2, opts2[i], ps2); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(b1.Bytes(), b2.Bytes()) {
			t.Errorf("%s: two saves of identical state produced different bytes", names[i])
		}
	}
}

// TestCountToUint32 pins SaveState's count guard as the pure function it was
// extracted into: a genuinely oversized params slice (2^32 pointers, 32 GiB)
// can never exist in a test, so the boundary, the wrapped-negative rejection
// and the error wording are asserted here directly, while SaveState runs the
// same call on every save.
func TestCountToUint32(t *testing.T) {
	// The in-range side of the boundary converts exactly, boundary included.
	for _, n := range []int{0, 1, 4096, int(uint32(math.MaxUint32))} {
		got, err := countToUint32(n)
		if err != nil || got != uint32(n) {
			t.Fatalf("countToUint32(%d) = (%d, %v), want (%d, nil)", n, got, err, uint32(n))
		}
	}
	// The out-of-range side, expressible only where int is 64-bit.
	if strconv.IntSize == 64 {
		for _, n := range []int{1 << 32, 1<<32 + 17, math.MaxInt64} {
			if got, err := countToUint32(n); err == nil {
				t.Fatalf("countToUint32(%d) = %d, want the stream count limit error", n, got)
			}
		}
		if _, err := countToUint32(1 << 32); err.Error() !=
			"4294967296 parameters exceed the stream count limit 4294967295" {
			t.Fatalf("countToUint32 boundary error %q reworded", err)
		}
	}
	// Negatives are unreachable from the len() call site but expressible on
	// the pure function: they wrap to >= 2^63 under the uint64 conversion and
	// are rejected by the same comparison.
	if _, err := countToUint32(-1); err == nil {
		t.Fatal("countToUint32(-1) accepted")
	}
}
