package serialize

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"runtime"
	"strings"
	"testing"
	"testing/quick"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/tensor"
)

// sameBits reports whether a and b have identical shapes and bit-identical
// float32 payloads (bit comparison so NaN, -0 and denormals count as equal
// to themselves).
func sameBits(a, b *tensor.Tensor) bool {
	if len(a.Shape) != len(b.Shape) {
		return false
	}
	for i := range a.Shape {
		if a.Shape[i] != b.Shape[i] {
			return false
		}
	}
	if len(a.Data) != len(b.Data) {
		return false
	}
	for i := range a.Data {
		if math.Float32bits(a.Data[i]) != math.Float32bits(b.Data[i]) {
			return false
		}
	}
	return true
}

func TestWriteReadRoundTripBitExact(t *testing.T) {
	nan := float32(math.NaN())
	inf := float32(math.Inf(1))
	// A shape zoo: 1D, 2D, scalar (rank 0), legal empties, a 3D storage
	// tensor, and one payload that crosses the internal encode chunk size.
	big := tensor.New(2*chunk + 7)
	for i := range big.Data {
		big.Data[i] = float32(i)*0.5 - 17
	}
	in := []*tensor.Tensor{
		tensor.FromData([]float32{1.5, -2.25, 0, nan, inf, -inf, math.SmallestNonzeroFloat32, -0.0}, 8),
		tensor.FromData([]float32{1, 2, 3, 4, 5, 6}, 2, 3),
		tensor.FromData([]float32{42}, 1),
		tensor.New(),     // rank-0 scalar
		tensor.New(0),    // empty 1D
		tensor.New(2, 0), // empty 2D
		tensor.FromData([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 2),
		big,
	}
	var buf bytes.Buffer
	if err := WriteTensors(&buf, in); err != nil {
		t.Fatalf("WriteTensors: %v", err)
	}
	out, err := ReadTensors(&buf)
	if err != nil {
		t.Fatalf("ReadTensors: %v", err)
	}
	if len(out) != len(in) {
		t.Fatalf("read %d tensors, want %d", len(out), len(in))
	}
	for i := range in {
		if !sameBits(in[i], out[i]) {
			t.Errorf("tensor %d: round trip is not bit-exact:\n in: %v\nout: %v", i, in[i].Shape, out[i].Shape)
		}
	}
}

func TestWriteIsDeterministic(t *testing.T) {
	ts := []*tensor.Tensor{tensor.FromData([]float32{1, -2, 3}, 3), tensor.New(2, 2)}
	var a, b bytes.Buffer
	if err := WriteTensors(&a, ts); err != nil {
		t.Fatal(err)
	}
	if err := WriteTensors(&b, ts); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a.Bytes(), b.Bytes()) {
		t.Error("two writes of the same tensors produced different bytes")
	}
	// Sanity: the header is the documented one.
	raw := a.Bytes()
	if string(raw[:4]) != "LNNS" || raw[4] != byte(Version) || binary.LittleEndian.Uint32(raw[5:9]) != 2 {
		t.Errorf("unexpected header bytes % x", raw[:9])
	}
}

func TestEmptyStreamRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteTensors(&buf, nil); err != nil {
		t.Fatal(err)
	}
	out, err := ReadTensors(&buf)
	if err != nil {
		t.Fatalf("ReadTensors: %v", err)
	}
	if len(out) != 0 {
		t.Fatalf("expected no tensors, got %d", len(out))
	}
}

// frame prepends a valid magic+version header to raw, for forging streams.
func frame(raw []byte) []byte {
	out := []byte{'L', 'N', 'N', 'S', byte(Version)}
	return append(out, raw...)
}

func count(n uint32) []byte {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], n)
	return b[:]
}

func shape64(dims ...int64) []byte {
	var out []byte
	var b [8]byte
	for _, d := range dims {
		binary.LittleEndian.PutUint64(b[:], uint64(d))
		out = append(out, b[:]...)
	}
	return out
}

func floats(fs ...float32) []byte {
	var out []byte
	var b [4]byte
	for _, f := range fs {
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(f))
		out = append(out, b[:]...)
	}
	return out
}

func TestReadTensorsRejectsHostileStreams(t *testing.T) {
	valid := func() []byte { // one [2] tensor with two floats
		var s []byte
		s = append(s, 1) // rank
		s = append(s, shape64(2)...)
		s = append(s, floats(1, 2)...)
		return frame(append(count(1), s...))
	}

	cases := []struct {
		name    string
		stream  []byte
		wantSub string
	}{
		{"bad magic", []byte("XXXX\x01" + string(count(0))), "magic"},
		{"version 99", []byte{'L', 'N', 'N', 'S', 99}, "version"},
		{
			// The red-team case: one tensor claiming a single 1<<62-wide axis
			// must error without allocating anywhere near 1<<62 floats.
			"dim 1<<62",
			frame(append(count(1), append([]byte{1}, shape64(1<<62)...)...)),
			"elements",
		},
		{
			// Each factor alone passes the limit (1<<30 == maxElems); only
			// their product overflows uint64, exercising the bits.Mul64 hi!=0
			// branch rather than the limit branch.
			"overflowing product",
			frame(append(count(1), append([]byte{2}, shape64(1<<30, 1<<34)...)...)),
			"overflow",
		},
		{
			"product over limit",
			frame(append(count(1), append([]byte{2}, shape64(1<<31, 8)...)...)), // 2^34 > maxElems
			"limit",
		},
		{
			"negative dimension",
			frame(append(count(1), append([]byte{1}, shape64(-7)...)...)),
			"negative",
		},
		{
			"rank over limit",
			frame(append(count(1), byte(200))),
			"rank",
		},
		{
			"count over limit",
			frame(count(0xFFFFFFFF)),
			"count",
		},
		{
			"truncated payload",
			// Claims two floats, provides one.
			frame(append(append(append(count(1), 1), shape64(2)...), floats(1)...)),
			"truncated",
		},
		{
			"truncated header",
			[]byte{'L', 'N'},
			"truncated",
		},
		{
			"trailing bytes",
			append(valid(), 0xFF),
			"trailing",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ReadTensors(bytes.NewReader(tc.stream))
			if err == nil {
				t.Fatalf("hostile stream accepted: %v", got)
			}
			if !strings.Contains(err.Error(), tc.wantSub) {
				t.Errorf("error %q does not mention %q", err, tc.wantSub)
			}
		})
	}
}

// TestReadTensorsRejectsUnknownVersionsActionably pins the versioning
// contract: unknown versions are refused rather than parsed on a guess, and
// the error must tell the caller which way the skew goes. A version above
// the current one is a checkpoint from the future — the message carries the
// actionable "written by a newer version / update" hint; a version below v1
// can only be corruption, since no older layout was ever released. The
// historic prefix of the message is asserted too, so documentation quoting
// it keeps matching.
func TestReadTensorsRejectsUnknownVersionsActionably(t *testing.T) {
	for _, tc := range []struct {
		name    string
		version byte
		want    string
	}{
		{"version 2 (next release)", 2, "written by a newer version"},
		{"version 99 (far future)", 99, "update this build"},
		{"version 0 (never released)", 0, "no earlier layout exists"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stream := []byte{'L', 'N', 'N', 'S', tc.version, 0, 0, 0, 0} // magic, version, empty count
			_, err := ReadTensors(bytes.NewReader(stream))
			if err == nil {
				t.Fatal("unknown version accepted")
			}
			msg := err.Error()
			prefix := fmt.Sprintf("unsupported format version %d (this build reads version %d)", tc.version, Version)
			if !strings.Contains(msg, prefix) {
				t.Errorf("error %q loses the historic prefix %q", msg, prefix)
			}
			if !strings.Contains(msg, tc.want) {
				t.Errorf("error %q lacks the actionable hint %q", msg, tc.want)
			}
		})
	}
}

// TestHostileDimDoesNotAllocate proves the V-05 discipline: a stream claiming
// a 1<<62-wide dimension is rejected by validation before any data buffer is
// allocated. The whole read is required to run in a handful of allocations
// (error strings and small scratch), never the 1<<62-float make.
func TestHostileDimDoesNotAllocate(t *testing.T) {
	stream := frame(append(count(1), append([]byte{1}, shape64(1<<62)...)...))
	allocs := testing.AllocsPerRun(20, func() {
		if _, err := ReadTensors(bytes.NewReader(stream)); err == nil {
			t.Fatal("hostile stream accepted")
		}
	})
	if allocs > 50 {
		t.Errorf("hostile read took %.0f allocations; validation must run before allocation", allocs)
	}
}

func TestHostileCountDoesNotAllocate(t *testing.T) {
	stream := frame(count(0xFFFFFFFF))
	allocs := testing.AllocsPerRun(20, func() {
		if _, err := ReadTensors(bytes.NewReader(stream)); err == nil {
			t.Fatal("hostile stream accepted")
		}
	})
	if allocs > 50 {
		t.Errorf("hostile read took %.0f allocations", allocs)
	}
}

// noLen hides Len() so the reader exercises the unknown-length path, where
// truncation can only be detected by the failing read itself.
type noLen struct{ r io.Reader }

func (n noLen) Read(p []byte) (int, error) { return n.r.Read(p) }

func TestTruncationIsErrUnexpectedEOF(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteTensors(&buf, []*tensor.Tensor{tensor.FromData([]float32{1, 2, 3, 4}, 4)}); err != nil {
		t.Fatal(err)
	}
	raw := buf.Bytes()
	for _, cut := range []int{len(raw) - 1, len(raw) / 2, 4, 0} {
		// Known-length reader: the pre-allocation remaining-bytes check fires.
		if _, err := ReadTensors(bytes.NewReader(raw[:cut])); !errors.Is(err, io.ErrUnexpectedEOF) {
			t.Errorf("cut at %d: got %v, want errors.Is(io.ErrUnexpectedEOF)", cut, err)
		}
		// Unknown-length reader: the failing io.ReadFull must surface the
		// same sentinel.
		if _, err := ReadTensors(noLen{bytes.NewReader(raw[:cut])}); !errors.Is(err, io.ErrUnexpectedEOF) {
			t.Errorf("cut at %d (no Len): got %v, want errors.Is(io.ErrUnexpectedEOF)", cut, err)
		}
	}
}

func TestReadTensorsPropagatesReaderErrors(t *testing.T) {
	boom := errors.New("boom")
	_, err := ReadTensors(&errReader{err: boom})
	if !errors.Is(err, boom) {
		t.Errorf("got %v, want wrapping %v", err, boom)
	}
}

type errReader struct{ err error }

func (r *errReader) Read([]byte) (int, error) { return 0, r.err }

func TestWriteTensorsRejectsBadInputs(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteTensors(&buf, []*tensor.Tensor{tensor.New(1), nil}); err == nil {
		t.Error("nil tensor accepted")
	}
	buf.Reset()
	lying := &tensor.Tensor{Shape: []int{2}, Data: []float32{1, 2, 3}}
	if err := WriteTensors(&buf, []*tensor.Tensor{lying}); err == nil {
		t.Error("shape/data mismatch accepted")
	}
	buf.Reset()
	rank9 := &tensor.Tensor{Shape: []int{1, 1, 1, 1, 1, 1, 1, 1, 1}, Data: []float32{1}}
	if err := WriteTensors(&buf, []*tensor.Tensor{rank9}); err == nil {
		t.Error("rank-9 tensor accepted")
	}
	buf.Reset()
	negDim := &tensor.Tensor{Shape: []int{-1}, Data: nil}
	if err := WriteTensors(&buf, []*tensor.Tensor{negDim}); err == nil {
		t.Error("negative dimension accepted")
	}
	buf.Reset()
	overflowing := &tensor.Tensor{Shape: []int{1 << 62, 4}, Data: nil}
	if err := WriteTensors(&buf, []*tensor.Tensor{overflowing}); err == nil {
		t.Error("overflowing shape accepted")
	}
	buf.Reset()
	tooMany := make([]*tensor.Tensor, maxCount+1) // 8 MiB of nils, never written
	if err := WriteTensors(&buf, tooMany); err == nil {
		t.Error("over-limit tensor count accepted")
	}
}

func TestReadTensorsErrorsMidShapeAndTrailingProbe(t *testing.T) {
	// Truncated mid-shape: rank says 1, the int64 dimension arrives half-full.
	midShape := frame(append(append(count(1), 1), shape64(2)[:4]...))
	if _, err := ReadTensors(bytes.NewReader(midShape)); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Errorf("mid-shape cut: got %v, want ErrUnexpectedEOF", err)
	}

	// A valid payload followed by a reader-level failure during the trailing
	// byte probe must surface that error, not be swallowed.
	var buf bytes.Buffer
	if err := WriteTensors(&buf, []*tensor.Tensor{tensor.New(1)}); err != nil {
		t.Fatal(err)
	}
	boom := errors.New("probe failure")
	tail := io.MultiReader(bytes.NewReader(buf.Bytes()), &errReader{err: boom})
	if _, err := ReadTensors(tail); !errors.Is(err, boom) {
		t.Errorf("probe failure: got %v, want wrapping %v", err, boom)
	}
}

func TestWriteTensorsReportsIOErrors(t *testing.T) {
	err := WriteTensors(&limitWriter{limit: 6}, []*tensor.Tensor{tensor.New(3)})
	if err == nil {
		t.Fatal("short write accepted")
	}
	if !strings.Contains(err.Error(), "space") {
		t.Errorf("error %q loses the underlying cause", err)
	}
}

// limitWriter accepts limit bytes, then fails like a full disk.
type limitWriter struct{ limit int }

func (w *limitWriter) Write(p []byte) (int, error) {
	if len(p) > w.limit {
		n := w.limit
		w.limit = 0
		return n, errors.New("no space left on device")
	}
	w.limit -= len(p)
	return len(p), nil
}

func TestParametersRoundTripInPlaceKeepsGraphUsable(t *testing.T) {
	w := autograd.Var(tensor.FromData([]float32{2, 0, 0, 3}, 2, 2))
	b := autograd.Var(tensor.FromData([]float32{0.5, -1}, 2))
	params := []*autograd.Variable{w, b}

	var buf bytes.Buffer
	if err := WriteParameters(&buf, params); err != nil {
		t.Fatalf("WriteParameters: %v", err)
	}

	// Corrupt the live parameters, then restore from the stream in place.
	w.Data.Data[0] = 999
	b.Data.Data[1] = -999
	if err := LoadParameters(&buf, params); err != nil {
		t.Fatalf("LoadParameters: %v", err)
	}
	if w.Data.Data[0] != 2 || b.Data.Data[1] != -1 {
		t.Fatalf("in-place restore failed: w=%v b=%v", w.Data.Data, b.Data.Data)
	}

	// Variable and storage identity must survive the load: the graph keeps
	// referencing these exact objects.
	if params[0] != w || params[1] != b {
		t.Fatal("LoadParameters replaced Variable pointers")
	}

	// The restored parameters still differentiate: run a forward/backward and
	// check the gradients by hand. loss = mean((x@w + b)^2), x = [1, 2].
	x := autograd.Const(tensor.FromData([]float32{1, 2}, 1, 2))
	pred := autograd.Add(autograd.MatMul(x, w), b)
	loss := autograd.MeanAll(autograd.Hadamard(pred, pred))
	loss.Backward()
	if w.Grad == nil || b.Grad == nil {
		t.Fatal("backward did not populate gradients after an in-place load")
	}
	// x@w = [1,2]@[[2,0],[0,3]] = [2,6], pred = [2.5, 5];
	// d loss/d pred = 2*pred/2 = pred; d loss/d b = pred; d loss/d w = x^T pred.
	wantB := []float32{2.5, 5}
	wantW := []float32{2.5, 5, 5, 10}
	for i := range wantB {
		if b.Grad.Data[i] != wantB[i] {
			t.Errorf("b.Grad[%d] = %v, want %v", i, b.Grad.Data[i], wantB[i])
		}
	}
	for i := range wantW {
		if w.Grad.Data[i] != wantW[i] {
			t.Errorf("w.Grad[%d] = %v, want %v", i, w.Grad.Data[i], wantW[i])
		}
	}
}

func TestLoadParametersCountMismatch(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteParameters(&buf, []*autograd.Variable{
		autograd.Var(tensor.New(2)), autograd.Var(tensor.New(3)),
	}); err != nil {
		t.Fatal(err)
	}
	dst := []*autograd.Variable{autograd.Var(tensor.New(2)), autograd.Var(tensor.New(3)), autograd.Var(tensor.New(1))}
	err := LoadParameters(bytes.NewReader(buf.Bytes()), dst)
	if err == nil || !strings.Contains(err.Error(), "count") {
		t.Errorf("got %v, want a count-mismatch error", err)
	}
}

func TestLoadParametersShapeMismatchLeavesParamsUntouched(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteParameters(&buf, []*autograd.Variable{
		autograd.Var(tensor.FromData([]float32{1, 2, 3, 4, 5, 6}, 2, 3)),
		autograd.Var(tensor.FromData([]float32{7, 8}, 2)),
	}); err != nil {
		t.Fatal(err)
	}
	// Destination 0 has the right element count but the wrong layout, and the
	// stream's second tensor is [2] while the destination is [3]: the first
	// mismatch must abort before anything is copied (validate-all-then-copy).
	ok := autograd.Var(tensor.New(3, 2))
	bad := autograd.Var(tensor.New(3))
	origOK := append([]float32(nil), ok.Data.Data...)
	err := LoadParameters(bytes.NewReader(buf.Bytes()), []*autograd.Variable{ok, bad})
	if err == nil || !strings.Contains(err.Error(), "shape") {
		t.Fatalf("got %v, want a shape-mismatch error", err)
	}
	for i := range origOK {
		if ok.Data.Data[i] != origOK[i] {
			t.Fatalf("failed load mutated parameter 0: %v, want untouched %v", ok.Data.Data, origOK)
		}
	}
}

func TestLoadParametersRejectsNilDestination(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteParameters(&buf, []*autograd.Variable{autograd.Var(tensor.New(1))}); err != nil {
		t.Fatal(err)
	}
	if err := LoadParameters(bytes.NewReader(buf.Bytes()), []*autograd.Variable{nil}); err == nil {
		t.Error("nil destination accepted")
	}
	if err := LoadParameters(bytes.NewReader(buf.Bytes()), []*autograd.Variable{{Data: nil}}); err == nil {
		t.Error("dataless destination accepted")
	}
}

func TestWriteParametersRejectsNilSource(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteParameters(&buf, []*autograd.Variable{nil}); err == nil {
		t.Error("nil parameter accepted")
	}
	if err := WriteParameters(&buf, []*autograd.Variable{{Data: nil}}); err == nil {
		t.Error("dataless parameter accepted")
	}
}

// TestRoundTripFuzzish exercises random shapes and payloads through the
// chunked codec, comparing bit patterns.
func TestRoundTripFuzzish(t *testing.T) {
	f := func(seed int64) bool {
		r := rand.New(rand.NewSource(seed))
		n := r.Intn(4) + 1
		ts := make([]*tensor.Tensor, n)
		for i := range ts {
			rank := r.Intn(3)
			shape := make([]int, rank)
			total := 1
			for d := range shape {
				shape[d] = r.Intn(6)
				total *= shape[d]
			}
			data := make([]float32, total)
			for j := range data {
				data[j] = float32(r.NormFloat64())
			}
			ts[i] = tensor.FromData(data, shape...)
		}
		var buf bytes.Buffer
		if err := WriteTensors(&buf, ts); err != nil {
			return false
		}
		out, err := ReadTensors(&buf)
		if err != nil || len(out) != len(ts) {
			return false
		}
		for i := range ts {
			if !sameBits(ts[i], out[i]) {
				return false
			}
		}
		return true
	}
	if err := quick.Check(func(seed int64) bool { return f(seed) }, &quick.Config{MaxCount: 200}); err != nil {
		t.Fatal(err)
	}
}

// allocatedBytes returns the heap bytes allocated during f (a delta of
// runtime.MemStats.TotalAlloc), for pinning the red-team allocation bounds.
func allocatedBytes(f func()) uint64 {
	var m0, m1 runtime.MemStats
	runtime.ReadMemStats(&m0)
	f()
	runtime.ReadMemStats(&m1)
	return m1.TotalAlloc - m0.TotalAlloc
}

// hostileClaimStream forges the red team's F1 stream: magic+version+count,
// rank 1 and a single giant dimension — len(stream) bytes total, zero
// payload. It must be handed to a reader without Len(), or the
// remaining-bytes fast path (correctly) rejects it before allocation.
func hostileClaimStream(dim int64) []byte {
	return frame(append(count(1), append([]byte{1}, shape64(dim)...)...))
}

func TestHostileClaimOnUnknownLengthReaderErrors(t *testing.T) {
	for _, dim := range []int64{1 << 24, 1 << 30} { // 64 MiB and 4 GiB claims
		stream := hostileClaimStream(dim)
		if len(stream) != 18 {
			t.Fatalf("forged stream is %d bytes, want exactly 18", len(stream))
		}
		// Hand-rolled no-Len reader: truncation is detectable only by the
		// failing read itself.
		if _, err := ReadTensors(noLen{bytes.NewReader(stream)}); !errors.Is(err, io.ErrUnexpectedEOF) {
			t.Errorf("dim %d via noLen: got %v, want errors.Is(io.ErrUnexpectedEOF)", dim, err)
		}
		// A real unknown-length reader: io.Pipe delivers the 18 bytes, then EOF.
		pr, pw := io.Pipe()
		go func() {
			pw.Write(stream)
			pw.Close()
		}()
		if _, err := ReadTensors(pr); !errors.Is(err, io.ErrUnexpectedEOF) {
			t.Errorf("dim %d via io.Pipe: got %v, want errors.Is(io.ErrUnexpectedEOF)", dim, err)
		}
	}
}

// TestHostileUnknownLengthReaderAllocatesLittle pins red team F1: on a
// reader without Len(), an 18-byte stream claiming a 1<<30-element (4 GiB)
// payload must not front-load the payload allocation. The old code ran
// make([]float32, n) before the first read — the red team measured 64 MiB
// for a 1<<24 claim; the incremental-growth path must stay under 1 MiB in
// total, proportional to the 18 bytes actually delivered.
func TestHostileUnknownLengthReaderAllocatesLittle(t *testing.T) {
	stream := hostileClaimStream(1 << 30)
	read := func() {
		if _, err := ReadTensors(noLen{bytes.NewReader(stream)}); !errors.Is(err, io.ErrUnexpectedEOF) {
			t.Fatalf("got %v, want errors.Is(io.ErrUnexpectedEOF)", err)
		}
	}
	if b := allocatedBytes(read); b > 1<<20 {
		t.Errorf("hostile read allocated %d bytes; must scale with the 18 bytes delivered, not the 4 GiB claimed", b)
	}
	if allocs := testing.AllocsPerRun(20, read); allocs > 50 {
		t.Errorf("hostile read took %.0f allocations; growth must stay chunk-sized", allocs)
	}
}

// TestLargeTensorThroughPipeRoundTripBitExact proves the incremental path
// corrupts nothing on legitimate streams: a payload spanning many chunks
// round-trips bit-exactly through both a noLen wrapper and a real io.Pipe.
func TestLargeTensorThroughPipeRoundTripBitExact(t *testing.T) {
	big := tensor.New(5*chunk + 11)
	for i := range big.Data {
		big.Data[i] = float32(i)*0.25 - 100
	}
	big.Data[3] = float32(math.NaN())
	big.Data[4] = float32(math.Inf(-1))
	big.Data[5] = -0.0
	in := []*tensor.Tensor{tensor.FromData([]float32{1, -2}, 2), big}

	var buf bytes.Buffer
	if err := WriteTensors(&buf, in); err != nil {
		t.Fatal(err)
	}
	raw := buf.Bytes()

	check := func(out []*tensor.Tensor, err error, via string) {
		t.Helper()
		if err != nil {
			t.Fatalf("%s: ReadTensors: %v", via, err)
		}
		if len(out) != len(in) {
			t.Fatalf("%s: read %d tensors, want %d", via, len(out), len(in))
		}
		for i := range in {
			if !sameBits(in[i], out[i]) {
				t.Errorf("%s: tensor %d is not bit-exact after round trip", via, i)
			}
		}
	}
	out, err := ReadTensors(noLen{bytes.NewReader(raw)})
	check(out, err, "noLen")

	pr, pw := io.Pipe()
	go func() {
		if _, err := pw.Write(raw); err != nil {
			pw.CloseWithError(err)
			return
		}
		pw.Close()
	}()
	out, err = ReadTensors(pr)
	check(out, err, "io.Pipe")
}

// TestHostileCountGrowsSliceIncrementally pins red team F3: a 9-byte stream
// claiming count = maxCount-1 tensors forced an ~8 MiB pointer slice up
// front in the old code. Incremental growth must keep the total allocation
// of the rejected read well under 64 KiB, on both reader classes.
func TestHostileCountGrowsSliceIncrementally(t *testing.T) {
	stream := frame(count(uint32(maxCount - 1)))
	for name, mk := range map[string]func() io.Reader{
		"known length":   func() io.Reader { return bytes.NewReader(stream) },
		"unknown length": func() io.Reader { return noLen{bytes.NewReader(stream)} },
	} {
		read := func() {
			if _, err := ReadTensors(mk()); err == nil {
				t.Fatal("hostile count stream accepted")
			}
		}
		if b := allocatedBytes(read); b > 64<<10 {
			t.Errorf("%s: hostile count read allocated %d bytes, want < 64 KiB", name, b)
		}
	}
}

func TestLoadParametersPreservesGrad(t *testing.T) {
	w := autograd.Var(tensor.FromData([]float32{1, 2}, 2))
	// Populate w.Grad on a throwaway graph: d mean(w*w) / dw = w.
	autograd.MeanAll(autograd.Hadamard(w, w)).Backward()
	if w.Grad == nil {
		t.Fatal("backward did not populate Grad")
	}
	gradBefore := w.Grad
	gradVals := append([]float32(nil), w.Grad.Data...)
	nonZero := false
	for _, v := range gradVals {
		nonZero = nonZero || v != 0
	}
	if !nonZero {
		t.Fatal("setup produced an all-zero gradient; the test would prove nothing")
	}

	// The stream carries different values.
	var buf bytes.Buffer
	if err := WriteParameters(&buf, []*autograd.Variable{autograd.Var(tensor.FromData([]float32{9, -9}, 2))}); err != nil {
		t.Fatal(err)
	}
	if err := LoadParameters(bytes.NewReader(buf.Bytes()), []*autograd.Variable{w}); err != nil {
		t.Fatalf("LoadParameters: %v", err)
	}

	// Data is overwritten with the streamed values...
	if w.Data.Data[0] != 9 || w.Data.Data[1] != -9 {
		t.Errorf("Data = %v, want the streamed [9 -9]", w.Data.Data)
	}
	// ...while Grad survives untouched — the same object, still holding the
	// now-stale gradients of the earlier graph (the documented contract; the
	// caller ZeroGrads before reusing the variable in a new graph).
	if w.Grad != gradBefore {
		t.Error("LoadParameters replaced the Grad tensor; it must be preserved")
	}
	for i := range gradVals {
		if w.Grad.Data[i] != gradVals[i] {
			t.Errorf("Grad[%d] = %v, want preserved %v", i, w.Grad.Data[i], gradVals[i])
		}
	}
}

// TestMutatedTensorStreamsNeverPanic replays the red team's mutation
// technique at smoke scale: 300 single-bit flips of a valid multi-tensor
// stream, each through both reader classes, must produce an error or a
// successful load — never a panic.
func TestMutatedTensorStreamsNeverPanic(t *testing.T) {
	ts := []*tensor.Tensor{
		tensor.FromData([]float32{1.5, -2.25, 0, -0.0}, 4),
		tensor.FromData([]float32{1, 2, 3, 4, 5, 6}, 2, 3),
		tensor.New(),
		tensor.New(0),
		tensor.FromData([]float32{1, 2, 3}, 1, 3),
	}
	var buf bytes.Buffer
	if err := WriteTensors(&buf, ts); err != nil {
		t.Fatal(err)
	}
	raw := buf.Bytes()
	rng := rand.New(rand.NewSource(7501))
	for i := 0; i < 300; i++ {
		mut := append([]byte(nil), raw...)
		pos := rng.Intn(len(mut))
		mut[pos] ^= 1 << uint(rng.Intn(8))
		func() {
			defer func() {
				if p := recover(); p != nil {
					t.Fatalf("mutant %d (byte %d bit-flipped): panic: %v", i, pos, p)
				}
			}()
			if i%2 == 0 {
				ReadTensors(bytes.NewReader(mut))
			} else {
				ReadTensors(noLen{bytes.NewReader(mut)})
			}
		}()
	}
}

func ExampleWriteTensors() {
	ts := []*tensor.Tensor{tensor.FromData([]float32{1, 2, 3}, 3)}
	var buf bytes.Buffer
	if err := WriteTensors(&buf, ts); err != nil {
		panic(err)
	}
	back, err := ReadTensors(&buf)
	if err != nil {
		panic(err)
	}
	fmt.Println(back[0].Data)
	// Output: [1 2 3]
}
