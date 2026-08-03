// Native Go fuzzing for the serialize package's untrusted-stream contract
// (doc/persistence.md): every read-path failure is an error, never a panic;
// the known-length reader (bytes.Reader fast path) and the unknown-length
// reader (progressive allocation, as from io.Pipe) must agree on the outcome
// of any byte stream; and a successful decode re-saves idempotently.
//
// These targets crystallize the red team's ad-hoc mutation discipline (8,700+
// hand mutants) into sustainable `go test -fuzz` targets. They live in the
// external test package (serialize_test) on purpose: they exercise only the
// exported API, exactly as a downstream consumer would, and stay free of the
// internal test helpers in serialize_test.go (no shared private state).
package serialize_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"os"
	"testing"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/serialize"
	"github.com/qorm/LNN/tensor"
)

// fuzzNoLen hides Len() so ReadTensors takes the unknown-length, progressive
// allocation path, where truncation is detectable only by the failing read.
type fuzzNoLen struct{ r io.Reader }

func (n fuzzNoLen) Read(p []byte) (int, error) { return n.r.Read(p) }

// fuzzSameBits reports whether a and b have identical shapes and bit-identical
// float32 payloads (Float32bits so NaN, -0 and denormals equal themselves).
func fuzzSameBits(a, b *tensor.Tensor) bool {
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

// --- Stream-forging helpers (self-contained; the wire format is documented) ---

func fuzzFrame(raw []byte) []byte {
	return append([]byte{'L', 'N', 'N', 'S', byte(serialize.Version)}, raw...)
}

func fuzzCount(n uint32) []byte {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], n)
	return b[:]
}

func fuzzShape64(dims ...int64) []byte {
	var out []byte
	var b [8]byte
	for _, d := range dims {
		binary.LittleEndian.PutUint64(b[:], uint64(d))
		out = append(out, b[:]...)
	}
	return out
}

func fuzzFloats(fs ...float32) []byte {
	var out []byte
	var b [4]byte
	for _, v := range fs {
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
		out = append(out, b[:]...)
	}
	return out
}

// fuzzValidTensorStream is a well-formed multi-tensor stream: a shape zoo
// (1D, 2D, rank-0 scalar, legal empties) that round-trips bit-exactly. It is
// the "happy path" seed every hostile mutation derives from.
func fuzzValidTensorStream() []byte {
	ts := []*tensor.Tensor{
		tensor.FromData([]float32{1.5, -2.25, 0, float32(math.NaN()), float32(math.Inf(1)), -0.0}, 6),
		tensor.FromData([]float32{1, 2, 3, 4, 5, 6}, 2, 3),
		tensor.New(),     // rank-0 scalar
		tensor.New(0),    // empty 1D
		tensor.New(2, 0), // empty 2D
	}
	var buf bytes.Buffer
	if err := serialize.WriteTensors(&buf, ts); err != nil {
		panic(err) // seeding bug, not a library fault
	}
	return buf.Bytes()
}

// fuzzGoldenSeed returns the committed golden model stream of the given kind
// and family, if present. These are nn model streams (a kind byte + header
// before the "LNNS" blob), so ReadTensors rejects them on magic — but they
// are dense, structurally realistic byte sequences for the mutator to start
// from. Both families are seeded: the v1 files exercise the legacy read path,
// the v2 files the checksum path.
func fuzzGoldenSeed(family, kind string) []byte {
	raw, err := os.ReadFile("testdata/golden_" + family + "_" + kind + ".lnns")
	if err != nil {
		return nil
	}
	return raw
}

// FuzzReadTensors feeds arbitrary byte streams to ReadTensors. Oracle:
//   - never panics (a recovered panic fails the input);
//   - the known-length and unknown-length readers agree: both succeed (to
//     bit-identical tensors) or both fail;
//   - when both fail, they agree on whether the failure is a truncation
//     (errors.Is(io.ErrUnexpectedEOF)) — the one classification the two reader
//     strategies can surface from different code paths;
//   - on success, the decode re-saves idempotently (Read→Write→Read is
//     bit-stable), proving the reader only ever yields re-encodable tensors.
func FuzzReadTensors(f *testing.F) {
	// (1) Golden / legitimate streams.
	valid := fuzzValidTensorStream()
	f.Add(valid)
	for _, k := range []string{"ltc", "cfc", "linear"} {
		for _, fam := range []string{"v1", "v2"} {
			if g := fuzzGoldenSeed(fam, k); g != nil {
				f.Add(g) // model stream: rejected on magic, a realistic mutation base
			}
		}
	}
	// Empty stream (magic read fails) and trivially small streams.
	f.Add([]byte{})
	f.Add([]byte{'L'})
	f.Add([]byte("LNNS"))
	var emptyBuf bytes.Buffer
	if err := serialize.WriteTensors(&emptyBuf, nil); err != nil {
		panic(err) // seeding bug, not a library fault
	}
	f.Add(emptyBuf.Bytes())        // genuine valid empty v2 stream (count 0 + checksum)
	f.Add(fuzzFrame(fuzzCount(0))) // header-only empty stream: checksum truncated off

	// (2) Checksum-path probes: the mutator gets realistic starting points for
	// the v2 verification path — a flipped checksum byte, checksums truncated
	// by 1 and 3 bytes, and trailing bytes after a valid checksum.
	badCRC := append([]byte(nil), valid...)
	badCRC[len(badCRC)-1] ^= 0xFF
	f.Add(badCRC)
	f.Add(valid[:len(valid)-1])
	f.Add(valid[:len(valid)-3])
	f.Add(append(append([]byte(nil), valid...), 0xFF))

	// (3) Hand-forged red-team classics. Each comment names the attack intent.
	f.Add([]byte("XXXX\x01" + string(fuzzCount(0))))                                               // bad magic
	f.Add([]byte{'L', 'N', 'N', 'S', 99})                                                          // version 99 (far future)
	f.Add([]byte{'L', 'N', 'N', 'S', 0, 0, 0, 0, 0})                                               // version 0 (never released)
	f.Add(fuzzFrame(fuzzCount(0xFFFFFFFF)))                                                        // count = 2^32-1 (over limit)
	f.Add(fuzzFrame(append(fuzzCount(1), append([]byte{1}, fuzzShape64(1<<62)...)...)))            // 1<<62-wide axis
	f.Add(fuzzFrame(append(fuzzCount(1), append([]byte{2}, fuzzShape64(1<<30, 1<<34)...)...)))     // overflowing product
	f.Add(fuzzFrame(append(fuzzCount(1), append([]byte{1}, fuzzShape64(-7)...)...)))               // negative dimension
	f.Add(fuzzFrame(append(fuzzCount(1), byte(200))))                                              // rank over limit
	f.Add(fuzzFrame(append(append(append(fuzzCount(1), 1), fuzzShape64(2)...), fuzzFloats(1)...))) // mid-tensor truncation
	f.Add(append(fuzzValidTensorStream(), 0xFF))                                                   // trailing byte
	f.Add([]byte{'L', 'N'})                                                                        // truncated header

	f.Fuzz(func(t *testing.T, data []byte) {
		var knownOut []*tensor.Tensor
		var knownErr, unkErr error
		var unkOut []*tensor.Tensor

		func() {
			defer func() {
				if p := recover(); p != nil {
					t.Fatalf("ReadTensors panicked on known-length reader: %v", p)
				}
			}()
			knownOut, knownErr = serialize.ReadTensors(bytes.NewReader(data))
		}()
		func() {
			defer func() {
				if p := recover(); p != nil {
					t.Fatalf("ReadTensors panicked on unknown-length reader: %v", p)
				}
			}()
			unkOut, unkErr = serialize.ReadTensors(fuzzNoLen{bytes.NewReader(data)})
		}()

		switch {
		case knownErr == nil && unkErr == nil:
			if len(knownOut) != len(unkOut) {
				t.Fatalf("reader disagreement: known decoded %d tensors, unknown decoded %d",
					len(knownOut), len(unkOut))
			}
			for i := range knownOut {
				if !fuzzSameBits(knownOut[i], unkOut[i]) {
					t.Fatalf("reader disagreement: tensor %d differs between reader classes", i)
				}
			}
			// Successful decode must re-save idempotently.
			var buf bytes.Buffer
			if err := serialize.WriteTensors(&buf, knownOut); err != nil {
				t.Fatalf("re-saving decoded tensors failed: %v", err)
			}
			again, err := serialize.ReadTensors(bytes.NewReader(buf.Bytes()))
			if err != nil {
				t.Fatalf("re-saved stream does not re-read: %v", err)
			}
			if len(again) != len(knownOut) {
				t.Fatalf("idempotence broken: %d tensors, want %d", len(again), len(knownOut))
			}
			for i := range again {
				if !fuzzSameBits(again[i], knownOut[i]) {
					t.Fatalf("idempotence broken: tensor %d changed across Read->Write->Read", i)
				}
			}
		case knownErr == nil || unkErr == nil:
			t.Fatalf("reader disagreement: known err=%v, unknown err=%v", knownErr, unkErr)
		default:
			// Both failed: they must agree on the truncation classification.
			if errors.Is(knownErr, io.ErrUnexpectedEOF) != errors.Is(unkErr, io.ErrUnexpectedEOF) {
				t.Fatalf("truncation classification disagrees: known=%v, unknown=%v", knownErr, unkErr)
			}
		}
	})
}

// FuzzLoadParameters feeds arbitrary byte streams to LoadParameters over a
// fixed two-parameter destination. Oracle: never panics; on error the
// destination is untouched (validate-all-then-copy — captured before, compared
// bit for bit after); on success every destination buffer equals the decoded
// stream bit for bit.
func FuzzLoadParameters(f *testing.F) {
	f.Add(fuzzValidTensorStream())
	f.Add([]byte{})
	f.Add([]byte("LNNS"))
	f.Add(fuzzFrame(fuzzCount(0)))
	f.Add(fuzzFrame(fuzzCount(0xFFFFFFFF)))
	f.Add(fuzzFrame(append(fuzzCount(1), append([]byte{1}, fuzzShape64(1<<62)...)...)))
	// A stream that matches the destination below exactly: one [2] and one [3].
	matching := func() []byte {
		var buf bytes.Buffer
		if err := serialize.WriteParameters(&buf, []*autograd.Variable{
			autograd.Var(tensor.FromData([]float32{9, -9}, 2)),
			autograd.Var(tensor.FromData([]float32{1, 2, 3}, 3)),
		}); err != nil {
			panic(err)
		}
		return buf.Bytes()
	}()
	f.Add(matching)
	f.Add(append(matching, 0xDE)) // trailing byte after an otherwise valid stream

	f.Fuzz(func(t *testing.T, data []byte) {
		newDst := func() []*autograd.Variable {
			return []*autograd.Variable{
				autograd.Var(tensor.FromData([]float32{100, 200}, 2)),
				autograd.Var(tensor.FromData([]float32{1, 2, 3}, 3)),
			}
		}
		dst := newDst()
		before := make([][]float32, len(dst))
		for i, p := range dst {
			before[i] = append([]float32(nil), p.Data.Data...)
		}

		var err error
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("LoadParameters panicked: %v", r)
				}
			}()
			err = serialize.LoadParameters(bytes.NewReader(data), dst)
		}()

		if err != nil {
			// Failed load: every destination buffer must be exactly as it was.
			for i, p := range dst {
				for j := range before[i] {
					if math.Float32bits(p.Data.Data[j]) != math.Float32bits(before[i][j]) {
						t.Fatalf("failed load mutated dst[%d][%d]: got %v, want %v", i, j, p.Data.Data[j], before[i][j])
					}
				}
			}
			return
		}
		// Successful load: decode independently and compare bit for bit.
		ts, rerr := serialize.ReadTensors(bytes.NewReader(data))
		if rerr != nil || len(ts) != len(dst) {
			t.Fatalf("LoadParameters succeeded but ReadTensors disagrees: err=%v n=%d", rerr, len(ts))
		}
		for i, p := range dst {
			if !fuzzSameBits(p.Data, ts[i]) {
				t.Fatalf("dst[%d] does not match the decoded stream", i)
			}
		}
	})
}
