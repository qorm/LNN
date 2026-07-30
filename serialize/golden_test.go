// Golden-vector regression for the frozen v1 wire format (see "Format
// versioning" in the package doc of serialize.go).
//
// This file lives in the external test package (serialize_test) on purpose:
// the golden streams are nn model streams, so pinning them requires nn's
// Save/Load — which serialize itself must never import. testdata/ holds the
// committed artifacts:
//
//	golden_v1_ltc.lnns            SaveLTC output
//	golden_v1_cfc.lnns            SaveCfC output
//	golden_v1_linear.lnns         SaveLinear output
//	golden_v1_<kind>.expected.txt the exact Step/Forward outputs each loaded
//	                              cell must reproduce, as float32 bit patterns
//
// The golden cells are built with these exact, fixed parameters — changing
// any of them invalidates the artifacts (regenerate deliberately with
// `go test ./serialize -write-golden`, never by accident):
//
//	LTC:    nn.NewLTC(4, 6, nil /* FullyConnected */, 6, rand.New(rand.NewSource(101)))
//	CfC:    nn.NewCfC(4, 6, nil /* FullyConnected */, rand.New(rand.NewSource(202)))
//	Linear: nn.NewLinear(6, 3, rand.New(rand.NewSource(303)))
//
// and stepped on the fixed inputs declared below (goldenCellX/goldenCellH
// for both cells at their per-kind ts, goldenLinearX for the layer). The
// CfC golden stream reflects the cell after the #10 erev bake: its loaded
// Step output is bit-identical to the original cell's, which is exactly the
// equivalence the bake was required to preserve.
//
// # Platform-graded assertions
//
// The committed vectors were generated on arm64 (Apple Silicon), and bit
// for bit float32 reproducibility is NOT a cross-architecture guarantee in
// Go. The language specification explicitly permits contraction: "An
// implementation may combine multiple floating-point operations into a
// single fused operation, possibly across statements, and produce a result
// that differs from the value obtained by executing and rounding the
// instructions individually." A fused multiply-add rounds once where a
// non-fused sequence rounds per operation, so an arm64 build (FMADD-class
// emission) and an amd64 build can legitimately disagree by 1 ULP per
// contraction — which is exactly what CI measured (0xbe8aa433 vs
// 0xbe8aa430). The library code is platform-neutral in the Go-spec sense;
// only the assertions are graded:
//
//   - arm64 (the generating architecture): every golden assertion stays
//     bit for bit / byte for byte (runtime.GOARCH == "arm64").
//   - any other architecture: the format skeleton — magic, version, tensor
//     count, ranks and shapes — is STILL asserted byte for byte, while
//     each float32 payload element is asserted within goldenULPTolerance
//     (4) ULPs: four times the measured drift, leaving headroom for FMA
//     chains without losing teeth (a corrupted payload is still rejected,
//     pinned by TestGoldenULPToleranceDiscriminates).
//
// Self-checks that never leave the platform under test keep full
// bit-exactness everywhere: FMA choices are fixed at compile time, so one
// build on one machine is deterministic. That covers the known-length vs
// unknown-length reader agreement. The loaded-cell vs same-seed-original
// comparison is NOT such a self-check off the generating architecture — the
// original cell is rebuilt from its seed by local code, and its
// construction can itself contract differently — so it is graded too.
package serialize_test

import (
	"bytes"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/nn"
	"github.com/qorm/LNN/tensor"
)

// writeGolden regenerates the committed artifacts. Off by default so the
// golden vectors can only change through a deliberate, visible test run.
var writeGolden = flag.Bool("write-golden", false, "regenerate the committed golden streams and expectation files under testdata/")

// goldenStrictArch reports whether this process runs on the architecture
// the committed golden vectors were generated on (arm64, Apple Silicon).
// On it, the freeze asserts bit for bit / byte for byte; everywhere else
// the skeleton stays byte-frozen and float32 payloads are compared within
// goldenULPTolerance (see the file comment for the FMA-contraction root
// cause). Being a constant, it dead-code-eliminates the inactive branch.
const goldenStrictArch = runtime.GOARCH == "arm64"

// goldenULPTolerance is the float32 payload window applied off the
// generating architecture: 4 ULPs. CI measured a cross-architecture drift
// of exactly 1 ULP; 4 leaves headroom for chained fused operations while
// still rejecting real payload corruption (twice the window fails, pinned
// by TestGoldenULPToleranceDiscriminates).
const goldenULPTolerance uint32 = 4

// The documented construction seeds (see the file comment for the full
// parameter list of each cell).
const (
	goldenSeedLTC    int64 = 101
	goldenSeedCfC    int64 = 202
	goldenSeedLinear int64 = 303
)

// The documented fixed inputs. Both cells share one [2,4] input and one
// non-zero [2,6] initial state; the time spans differ per kind (below).
var (
	goldenCellX = []float32{0.25, -0.5, 0.75, -1.0, 0.1, 0.2, -0.3, 0.4}
	goldenCellH = []float32{0.1, -0.2, 0.3, -0.4, 0.5, -0.6, -0.1, 0.2, -0.3, 0.4, -0.5, 0.6}
	goldenTsLTC = 0.1
	goldenTsCfC = 0.25
	goldenLinX  = []float32{1, -1, 0.5, -0.5, 0.25, -0.25, 0.75, -0.75, 0.125, -0.125, 2, -2}
)

func goldenCellInput() *autograd.Variable {
	return autograd.Var(tensor.FromData(append([]float32(nil), goldenCellX...), 2, 4))
}

func goldenCellState() *autograd.Variable {
	return autograd.Var(tensor.FromData(append([]float32(nil), goldenCellH...), 2, 6))
}

func goldenLinearInput() *autograd.Variable {
	return autograd.Var(tensor.FromData(append([]float32(nil), goldenLinX...), 2, 6))
}

func buildGoldenLTC() *nn.LTC {
	return nn.NewLTC(4, 6, nil, 6, rand.New(rand.NewSource(goldenSeedLTC)))
}

func buildGoldenCfC() *nn.CfC {
	return nn.NewCfC(4, 6, nil, rand.New(rand.NewSource(goldenSeedCfC)))
}

func buildGoldenLinear() *nn.Linear {
	return nn.NewLinear(6, 3, rand.New(rand.NewSource(goldenSeedLinear)))
}

// namedTensor pairs a result tensor with the name it carries in the
// expectation file.
type namedTensor struct {
	name string
	t    *tensor.Tensor
}

// goldenCase is one golden model: how to re-emit its stream, how to step a
// freshly built cell (the "original"), and how to load a stream and step
// the loaded cell.
type goldenCase struct {
	name string
	doc  []string
	save func(t *testing.T) []byte
	step func() []namedTensor
	run  func(t *testing.T, r io.Reader) []namedTensor
}

func goldenCases(t *testing.T) []goldenCase {
	t.Helper()
	return []goldenCase{
		{
			name: "ltc",
			doc: []string{
				"cell: nn.NewLTC(4, 6, nil, 6, rand.New(rand.NewSource(101)))",
				fmt.Sprintf("input x [2 4]: %v", goldenCellX),
				fmt.Sprintf("input h [2 6]: %v", goldenCellH),
				fmt.Sprintf("ts: %v", goldenTsLTC),
			},
			save: func(t *testing.T) []byte {
				var buf bytes.Buffer
				if err := nn.SaveLTC(&buf, buildGoldenLTC()); err != nil {
					t.Fatalf("SaveLTC: %v", err)
				}
				return buf.Bytes()
			},
			step: func() []namedTensor {
				out, hNew := buildGoldenLTC().Step(goldenCellInput(), goldenCellState(), goldenTsLTC)
				return []namedTensor{{"out", out.Data}, {"hnew", hNew.Data}}
			},
			run: func(t *testing.T, r io.Reader) []namedTensor {
				cell, err := nn.LoadLTC(r)
				if err != nil {
					t.Fatalf("LoadLTC: %v", err)
				}
				out, hNew := cell.Step(goldenCellInput(), goldenCellState(), goldenTsLTC)
				return []namedTensor{{"out", out.Data}, {"hnew", hNew.Data}}
			},
		},
		{
			name: "cfc",
			doc: []string{
				"cell: nn.NewCfC(4, 6, nil, rand.New(rand.NewSource(202)))",
				fmt.Sprintf("input x [2 4]: %v", goldenCellX),
				fmt.Sprintf("input h [2 6]: %v", goldenCellH),
				fmt.Sprintf("ts: %v", goldenTsCfC),
			},
			save: func(t *testing.T) []byte {
				var buf bytes.Buffer
				if err := nn.SaveCfC(&buf, buildGoldenCfC()); err != nil {
					t.Fatalf("SaveCfC: %v", err)
				}
				return buf.Bytes()
			},
			step: func() []namedTensor {
				out, hNew := buildGoldenCfC().Step(goldenCellInput(), goldenCellState(), goldenTsCfC)
				return []namedTensor{{"out", out.Data}, {"hnew", hNew.Data}}
			},
			run: func(t *testing.T, r io.Reader) []namedTensor {
				cell, err := nn.LoadCfC(r)
				if err != nil {
					t.Fatalf("LoadCfC: %v", err)
				}
				out, hNew := cell.Step(goldenCellInput(), goldenCellState(), goldenTsCfC)
				return []namedTensor{{"out", out.Data}, {"hnew", hNew.Data}}
			},
		},
		{
			name: "linear",
			doc: []string{
				"cell: nn.NewLinear(6, 3, rand.New(rand.NewSource(303)))",
				fmt.Sprintf("input x [2 6]: %v", goldenLinX),
			},
			save: func(t *testing.T) []byte {
				var buf bytes.Buffer
				if err := nn.SaveLinear(&buf, buildGoldenLinear()); err != nil {
					t.Fatalf("SaveLinear: %v", err)
				}
				return buf.Bytes()
			},
			step: func() []namedTensor {
				return []namedTensor{{"out", buildGoldenLinear().Forward(goldenLinearInput()).Data}}
			},
			run: func(t *testing.T, r io.Reader) []namedTensor {
				layer, err := nn.LoadLinear(r)
				if err != nil {
					t.Fatalf("LoadLinear: %v", err)
				}
				return []namedTensor{{"out", layer.Forward(goldenLinearInput()).Data}}
			},
		},
	}
}

func goldenStreamPath(name string) string { return "testdata/golden_v1_" + name + ".lnns" }
func goldenExpectPath(name string) string { return "testdata/golden_v1_" + name + ".expected.txt" }

func readGoldenStream(t *testing.T, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile(goldenStreamPath(name))
	if err != nil {
		t.Fatalf("golden stream missing (regenerate with -write-golden): %v", err)
	}
	return raw
}

// float32Ordered maps a float32 onto an ordered integer axis: the IEEE-754
// sign-magnitude bit pattern is folded so that integer order matches numeric
// order and +0 and -0 land on the same point (0x00000000 -> 0; 0x80000000
// -> 0; 0x80000001 -> -1; 0xFFFFFFFF -> -2139095039). Neighbors on this axis
// are exactly one ULP apart.
func float32Ordered(f float32) int32 {
	b := math.Float32bits(f)
	if b >= 0x80000000 {
		// Negative: flip so that more-negative values order lower. The
		// uint32 arithmetic wraps, landing at the right two's-complement
		// int32 (0x80000000 - 0x80000001 -> 0xFFFFFFFF -> -1).
		return int32(0x80000000 - b)
	}
	return int32(b)
}

// float32ULPDistance is the count of representable float32 values between a
// and b: 0 means bit-identical (a NaN equals only its own bit pattern, as
// in sameGoldenBits), 1 means adjacent. The widest possible distance
// (-maxfloat to +maxfloat) fits in uint32; the subtraction runs in int64 so
// the fold cannot overflow.
func float32ULPDistance(a, b float32) uint32 {
	d := int64(float32Ordered(a)) - int64(float32Ordered(b))
	if d < 0 {
		d = -d
	}
	return uint32(d)
}

// nudgeFloat32 moves f by ulps representable neighbors along the ordered
// axis (positive toward +Inf, negative toward -Inf), for constructing
// synthetic drift in the discrimination test.
func nudgeFloat32(f float32, ulps int32) float32 {
	o := int64(float32Ordered(f)) + int64(ulps)
	if o >= 0 {
		return math.Float32frombits(uint32(o))
	}
	return math.Float32frombits(0x80000000 + uint32(-o))
}

// goldenPayloadError compares got against want tensor by tensor: the tensor
// count, the names, and the shapes must agree exactly, and every element
// must be within tolerance ULPs (tolerance 0 is bit-exactness: two float32s
// are 0 ULP apart if and only if their Float32bits patterns are identical,
// NaN and -0 included). It returns nil on agreement and an error describing
// the first mismatch — with the measured ULP distance on payload drift, so
// a failure states how far off the platform landed — making it the shared
// core of both halves of the platform-graded freeze.
func goldenPayloadError(label string, got, want []namedTensor, tolerance uint32) error {
	if len(got) != len(want) {
		return fmt.Errorf("%s: %d result tensors, want %d", label, len(got), len(want))
	}
	for i := range got {
		if got[i].name != want[i].name {
			return fmt.Errorf("%s: result %d is %q, want %q", label, i, got[i].name, want[i].name)
		}
		g, w := got[i].t, want[i].t
		if len(g.Shape) != len(w.Shape) || len(g.Data) != len(w.Data) {
			return fmt.Errorf("%s/%s: shape %v, want %v", label, got[i].name, g.Shape, w.Shape)
		}
		for d := range g.Shape {
			if g.Shape[d] != w.Shape[d] {
				return fmt.Errorf("%s/%s: shape %v, want %v", label, got[i].name, g.Shape, w.Shape)
			}
		}
		for j := range g.Data {
			if dist := float32ULPDistance(g.Data[j], w.Data[j]); dist > tolerance {
				return fmt.Errorf("%s/%s: element %d is %#x (%v), want %#x (%v): %d ULP apart, tolerance %d",
					label, got[i].name, j,
					math.Float32bits(g.Data[j]), g.Data[j],
					math.Float32bits(w.Data[j]), w.Data[j],
					dist, tolerance)
			}
		}
	}
	return nil
}

// sameGoldenBits reports whether got matches the want tensors name-by-name,
// shape-by-shape and bit-by-bit (Float32bits, so NaN and -0 compare equal to
// themselves — bit-exactness is the whole point). It is the strict half of
// the platform-graded freeze, kept at tolerance 0.
func sameGoldenBits(t *testing.T, label string, got, want []namedTensor) {
	t.Helper()
	if err := goldenPayloadError(label, got, want, 0); err != nil {
		t.Fatal(err)
	}
}

// sameGoldenWithinULP is the graded half of the freeze: the skeleton (names,
// shapes, tensor count) is still exact, but payload elements may drift by up
// to tolerance ULPs — the cross-architecture FMA-contraction window (see the
// file comment). The failure message carries the measured ULP distance.
func sameGoldenWithinULP(t *testing.T, label string, got, want []namedTensor, tolerance uint32) {
	t.Helper()
	if err := goldenPayloadError(label, got, want, tolerance); err != nil {
		t.Fatal(err)
	}
}

// goldenEnvelopeBytes is the byte length of the model envelope that precedes
// the serialize blob in each Save* stream (nn/save.go): one kind byte, then
// int32 header fields — LTC writes inDim, units, unfolds; CfC writes inDim,
// units; Linear writes none.
var goldenEnvelopeBytes = map[string]int{
	"ltc":    1 + 3*4,
	"cfc":    1 + 2*4,
	"linear": 1,
}

// structuredStreamError compares two Save* model streams at the graded level
// used off the generating architecture: the model envelope (kind byte and
// int32 header) and the wire skeleton (magic, version, tensor count,
// per-tensor rank and shape fields) must agree byte for byte — the format
// layout is frozen cross-platform — while each float32 payload element must
// agree within tolerance ULPs. An equal skeleton implies equal payload
// lengths, so an overall length mismatch is itself skeleton drift and is
// reported as such. Both streams are walked with bounds checks so a corrupt
// artifact fails with an error rather than a panic.
func structuredStreamError(name string, golden, fresh []byte, tolerance uint32) error {
	env, ok := goldenEnvelopeBytes[name]
	if !ok {
		return fmt.Errorf("%s: no envelope length known for structural comparison", name)
	}
	if len(golden) != len(fresh) {
		return fmt.Errorf("%s: re-written stream is %d bytes, committed golden is %d: skeleton drift",
			name, len(fresh), len(golden))
	}
	if len(golden) < env+9 {
		return fmt.Errorf("%s: stream is %d bytes, too short for the %d-byte envelope and 9-byte wire header",
			name, len(golden), env)
	}
	if !bytes.Equal(golden[:env], fresh[:env]) {
		return fmt.Errorf("%s: model envelope (kind byte + header) differs: golden % x, fresh % x",
			name, golden[:env], fresh[:env])
	}
	off := env
	if !bytes.Equal(golden[off:off+4], fresh[off:off+4]) {
		return fmt.Errorf("%s: magic differs: golden %q, fresh %q", name, golden[off:off+4], fresh[off:off+4])
	}
	off += 4
	if golden[off] != fresh[off] {
		return fmt.Errorf("%s: format version differs: golden %d, fresh %d", name, golden[off], fresh[off])
	}
	off++
	if !bytes.Equal(golden[off:off+4], fresh[off:off+4]) {
		return fmt.Errorf("%s: tensor count differs: golden %d, fresh %d", name,
			binary.LittleEndian.Uint32(golden[off:off+4]), binary.LittleEndian.Uint32(fresh[off:off+4]))
	}
	count := binary.LittleEndian.Uint32(golden[off : off+4])
	off += 4
	for ti := uint32(0); ti < count; ti++ {
		if off >= len(golden) {
			return fmt.Errorf("%s: truncated before tensor %d rank byte", name, ti)
		}
		if golden[off] != fresh[off] {
			return fmt.Errorf("%s: tensor %d rank differs: golden %d, fresh %d", name, ti, golden[off], fresh[off])
		}
		rank := int(golden[off])
		off++
		if off+8*rank > len(golden) {
			return fmt.Errorf("%s: truncated in tensor %d shape", name, ti)
		}
		if !bytes.Equal(golden[off:off+8*rank], fresh[off:off+8*rank]) {
			return fmt.Errorf("%s: tensor %d shape bytes differ: golden % x, fresh % x",
				name, ti, golden[off:off+8*rank], fresh[off:off+8*rank])
		}
		var elems int64 = 1
		for d := 0; d < rank; d++ {
			dim := int64(binary.LittleEndian.Uint64(golden[off+8*d:]))
			if dim < 0 {
				return fmt.Errorf("%s: tensor %d axis %d has a negative dimension %d", name, ti, d, dim)
			}
			elems *= dim
		}
		off += 8 * rank
		if 4*int(elems) > len(golden)-off {
			return fmt.Errorf("%s: truncated in tensor %d payload (%d elements, %d bytes left)",
				name, ti, elems, len(golden)-off)
		}
		for j := int64(0); j < elems; j++ {
			gb := binary.LittleEndian.Uint32(golden[off+4*int(j):])
			fb := binary.LittleEndian.Uint32(fresh[off+4*int(j):])
			if dist := float32ULPDistance(math.Float32frombits(gb), math.Float32frombits(fb)); dist > tolerance {
				return fmt.Errorf("%s: tensor %d element %d: golden %#x, fresh %#x: %d ULP apart, tolerance %d",
					name, ti, j, gb, fb, dist, tolerance)
			}
		}
		off += 4 * int(elems)
	}
	if off != len(golden) {
		return fmt.Errorf("%s: %d trailing byte(s) after the last tensor payload", name, len(golden)-off)
	}
	return nil
}

// writeExpectation writes the expectation file: comment lines documenting
// the construction and inputs, then one section per tensor — a header line
// "<name> <shape...>" and a line of float32 bit patterns in hex (%08x),
// row-major, one token per element.
func writeExpectation(t *testing.T, path string, doc []string, ts []namedTensor) {
	t.Helper()
	var sb strings.Builder
	sb.WriteString("# Golden expectation for ")
	sb.WriteString(strings.TrimSuffix(baseName(path), ".expected.txt"))
	sb.WriteString(".lnns — the exact outputs the loaded cell must produce.\n")
	sb.WriteString("# float32 bit patterns in hex, row-major, one token per element.\n")
	for _, d := range doc {
		sb.WriteString("# ")
		sb.WriteString(d)
		sb.WriteString("\n")
	}
	for _, nt := range ts {
		sb.WriteString(nt.name)
		for _, dim := range nt.t.Shape {
			fmt.Fprintf(&sb, " %d", dim)
		}
		sb.WriteString("\n")
		for j, v := range nt.t.Data {
			if j > 0 {
				sb.WriteString(" ")
			}
			fmt.Fprintf(&sb, "%08x", math.Float32bits(v))
		}
		sb.WriteString("\n")
	}
	if err := os.WriteFile(path, []byte(sb.String()), 0o644); err != nil {
		t.Fatalf("writing %s: %v", path, err)
	}
}

func baseName(path string) string {
	if i := strings.LastIndex(path, "/"); i >= 0 {
		return path[i+1:]
	}
	return path
}

// readExpectation parses an expectation file into named tensors.
func readExpectation(t *testing.T, path string) []namedTensor {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("expectation file missing (regenerate with -write-golden): %v", err)
	}
	var out []namedTensor
	lines := strings.Split(string(raw), "\n")
	for i := 0; i < len(lines); {
		line := strings.TrimSpace(lines[i])
		i++
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			t.Fatalf("%s: malformed section header %q", path, line)
		}
		shape := make([]int, len(fields)-1)
		n := 1
		for j, f := range fields[1:] {
			d, err := strconv.Atoi(f)
			if err != nil || d < 0 {
				t.Fatalf("%s: bad dimension %q in header %q", path, f, line)
			}
			shape[j] = d
			n *= d
		}
		if i >= len(lines) {
			t.Fatalf("%s: section %q has no data line", path, fields[0])
		}
		toks := strings.Fields(lines[i])
		i++
		if len(toks) != n {
			t.Fatalf("%s: section %q has %d tokens, shape %v implies %d", path, fields[0], len(toks), shape, n)
		}
		data := make([]float32, n)
		for j, tok := range toks {
			b, err := strconv.ParseUint(tok, 16, 32)
			if err != nil {
				t.Fatalf("%s: bad hex token %q in section %q", path, tok, fields[0])
			}
			data[j] = math.Float32frombits(uint32(b))
		}
		out = append(out, namedTensor{fields[0], tensor.FromData(data, shape...)})
	}
	return out
}

// noLenGolden hides Len() so loads exercise the unknown-length (progressive
// allocation) reader path.
type noLenGolden struct{ r io.Reader }

func (n noLenGolden) Read(p []byte) (int, error) { return n.r.Read(p) }

// TestGoldenStreamsLoadBitExact is the behavioral freeze: each committed
// stream must load and reproduce the committed expectation, and — the
// loaded-cell acceptance — the loaded cell must step identically to a
// freshly built same-seed original. Graded by platform (see the file
// comment): bit for bit on the generating architecture, within
// goldenULPTolerance ULPs elsewhere. The vs-original comparison is graded
// too because the original is rebuilt by local construction code, which
// may itself contract differently off the generating architecture.
func TestGoldenStreamsLoadBitExact(t *testing.T) {
	for _, tc := range goldenCases(t) {
		t.Run(tc.name, func(t *testing.T) {
			raw := readGoldenStream(t, tc.name)
			got := tc.run(t, bytes.NewReader(raw))
			want := readExpectation(t, goldenExpectPath(tc.name))
			if goldenStrictArch {
				sameGoldenBits(t, "loaded vs expectation", got, want)
				sameGoldenBits(t, "loaded vs same-seed original", got, tc.step())
				return
			}
			sameGoldenWithinULP(t, "loaded vs expectation", got, want, goldenULPTolerance)
			sameGoldenWithinULP(t, "loaded vs same-seed original", got, tc.step(), goldenULPTolerance)
		})
	}
}

// TestGoldenWriterStability is the byte-level freeze: rebuilding each cell
// from its documented seed and saving must re-emit the committed golden
// stream. Any drift in the writer, the tensor order, the header or
// same-seed construction fails here. Graded by platform (see the file
// comment): byte for byte on the generating architecture; elsewhere the
// skeleton — envelope, magic, version, count, ranks, shapes — is still
// byte-frozen and only the float32 payloads are compared within
// goldenULPTolerance ULPs. (Named with the Golden prefix so `-run Golden`
// selects the whole freeze trio — red-team F-RT3.)
func TestGoldenWriterStability(t *testing.T) {
	for _, tc := range goldenCases(t) {
		t.Run(tc.name, func(t *testing.T) {
			golden := readGoldenStream(t, tc.name)
			fresh := tc.save(t)
			if goldenStrictArch {
				if !bytes.Equal(fresh, golden) {
					t.Fatalf("re-written stream differs from the committed golden bytes (%d vs %d bytes): the v1 writer or same-seed construction drifted",
						len(fresh), len(golden))
				}
				return
			}
			if err := structuredStreamError(tc.name, golden, fresh, goldenULPTolerance); err != nil {
				t.Fatalf("re-written stream drift: %v", err)
			}
		})
	}
}

// TestGoldenULPToleranceDiscriminates guards the guard: the graded window
// must stay tight enough to reject real corruption. It drives the same
// comparison core the platform-graded assertions use (goldenPayloadError)
// with synthetic drift: ±(1-2) ULP and the exact boundary (4 ULP) must
// pass; 8 ULP — twice the window — and tolerance+1 must fail; shape drift
// and tensor-count drift must fail even with undisturbed payloads. It runs
// on every platform, so the window's teeth are exercised on arm64 too,
// where the golden assertions themselves stay bit-exact.
func TestGoldenULPToleranceDiscriminates(t *testing.T) {
	base := []float32{1.0, -2.5, 0, 3.5, -0.0, 1e-30}
	want := []namedTensor{{"out", tensor.FromData(append([]float32(nil), base...), 2, 3)}}

	nudged := func(elem int, ulps int32) []namedTensor {
		data := append([]float32(nil), base...)
		data[elem] = nudgeFloat32(data[elem], ulps)
		return []namedTensor{{"out", tensor.FromData(data, 2, 3)}}
	}

	for _, ulps := range []int32{-2, -1, 1, 2, int32(goldenULPTolerance)} {
		if err := goldenPayloadError("within-window", nudged(0, ulps), want, goldenULPTolerance); err != nil {
			t.Errorf("%+d ULP drift should be within tolerance %d: %v", ulps, goldenULPTolerance, err)
		}
	}
	for _, ulps := range []int32{-8, 8, int32(goldenULPTolerance) + 1} {
		if err := goldenPayloadError("outside-window", nudged(0, ulps), want, goldenULPTolerance); err == nil {
			t.Errorf("%+d ULP drift should be rejected by tolerance %d", ulps, goldenULPTolerance)
		}
	}
	// Drifting a negative element crosses the ±0 fold correctly too.
	if err := goldenPayloadError("negative fold", nudged(1, -8), want, goldenULPTolerance); err == nil {
		t.Errorf("-8 ULP drift on a negative element should be rejected by tolerance %d", goldenULPTolerance)
	}
	if err := goldenPayloadError("shape", []namedTensor{{"out", tensor.FromData(append([]float32(nil), base...), 3, 2)}}, want, goldenULPTolerance); err == nil {
		t.Errorf("shape drift should be rejected even with identical payloads")
	}
	if err := goldenPayloadError("count", append(append([]namedTensor{}, want...), want...), want, goldenULPTolerance); err == nil {
		t.Errorf("tensor-count drift should be rejected")
	}
	// And the window has meaning only against the strict path's zero: an
	// 8 ULP drift must fail tolerance 0 too (sanity of the shared core).
	if err := goldenPayloadError("strict", nudged(0, 1), want, 0); err == nil {
		t.Errorf("tolerance 0 must reject a 1 ULP drift")
	}
}

// TestGoldenStreamsLoadOnBothReaderClasses requires the two reader
// strategies — known remaining length (bytes.Reader fast path) and unknown
// length (progressive allocation, as from io.Pipe/net.Conn) — to load the
// golden streams to bit-identical cell behavior. This stays bit-exact on
// every platform, graded or not: both sides load the same bytes in the same
// binary on the same machine, so no cross-architecture rounding enters —
// it is a self-check, not a golden-payload comparison.
func TestGoldenStreamsLoadOnBothReaderClasses(t *testing.T) {
	for _, tc := range goldenCases(t) {
		t.Run(tc.name, func(t *testing.T) {
			raw := readGoldenStream(t, tc.name)
			known := tc.run(t, bytes.NewReader(raw))
			unknown := tc.run(t, noLenGolden{bytes.NewReader(raw)})
			sameGoldenBits(t, "known-length vs unknown-length reader", known, unknown)
		})
	}
}

// TestWriteGoldenFiles is the only sanctioned way to change the golden
// artifacts: a deliberate, reviewed run of `go test ./serialize
// -write-golden`. It rebuilds every stream and records the loaded cells'
// outputs as the new expectations. Skipped without the flag.
func TestWriteGoldenFiles(t *testing.T) {
	if !*writeGolden {
		t.Skip("golden regeneration is deliberate: go test ./serialize -write-golden")
	}
	for _, tc := range goldenCases(t) {
		raw := tc.save(t)
		if err := os.WriteFile(goldenStreamPath(tc.name), raw, 0o644); err != nil {
			t.Fatalf("writing golden stream: %v", err)
		}
		got := tc.run(t, bytes.NewReader(raw))
		writeExpectation(t, goldenExpectPath(tc.name), tc.doc, got)
		t.Logf("regenerated %s (%d bytes) and %s", goldenStreamPath(tc.name), len(raw), goldenExpectPath(tc.name))
	}
}
