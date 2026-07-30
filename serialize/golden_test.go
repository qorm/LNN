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
package serialize_test

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
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

// sameGoldenBits reports whether got matches the want tensors name-by-name,
// shape-by-shape and bit-by-bit (Float32bits, so NaN and -0 compare equal to
// themselves — bit-exactness is the whole point).
func sameGoldenBits(t *testing.T, label string, got, want []namedTensor) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d result tensors, want %d", label, len(got), len(want))
	}
	for i := range got {
		if got[i].name != want[i].name {
			t.Fatalf("%s: result %d is %q, want %q", label, i, got[i].name, want[i].name)
		}
		g, w := got[i].t, want[i].t
		if len(g.Shape) != len(w.Shape) || len(g.Data) != len(w.Data) {
			t.Fatalf("%s/%s: shape %v, want %v", label, got[i].name, g.Shape, w.Shape)
		}
		for d := range g.Shape {
			if g.Shape[d] != w.Shape[d] {
				t.Fatalf("%s/%s: shape %v, want %v", label, got[i].name, g.Shape, w.Shape)
			}
		}
		for j := range g.Data {
			if math.Float32bits(g.Data[j]) != math.Float32bits(w.Data[j]) {
				t.Fatalf("%s/%s: element %d is %#x (%v), want %#x (%v)",
					label, got[i].name, j,
					math.Float32bits(g.Data[j]), g.Data[j],
					math.Float32bits(w.Data[j]), w.Data[j])
			}
		}
	}
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
// stream must load and reproduce the committed expectation bit for bit, and
// — the loaded-cell acceptance — the loaded cell must step bit-identically
// to a freshly built same-seed original.
func TestGoldenStreamsLoadBitExact(t *testing.T) {
	for _, tc := range goldenCases(t) {
		t.Run(tc.name, func(t *testing.T) {
			raw := readGoldenStream(t, tc.name)
			got := tc.run(t, bytes.NewReader(raw))
			want := readExpectation(t, goldenExpectPath(tc.name))
			sameGoldenBits(t, "loaded vs expectation", got, want)
			sameGoldenBits(t, "loaded vs same-seed original", got, tc.step())
		})
	}
}

// TestWriterStability is the byte-level freeze: rebuilding each cell from
// its documented seed and saving must re-emit the committed golden stream
// byte for byte. Any drift in the writer, the tensor order, the header or
// same-seed construction fails here.
func TestWriterStability(t *testing.T) {
	for _, tc := range goldenCases(t) {
		t.Run(tc.name, func(t *testing.T) {
			golden := readGoldenStream(t, tc.name)
			fresh := tc.save(t)
			if !bytes.Equal(fresh, golden) {
				t.Fatalf("re-written stream differs from the committed golden bytes (%d vs %d bytes): the v1 writer or same-seed construction drifted",
					len(fresh), len(golden))
			}
		})
	}
}

// TestGoldenStreamsLoadOnBothReaderClasses requires the two reader
// strategies — known remaining length (bytes.Reader fast path) and unknown
// length (progressive allocation, as from io.Pipe/net.Conn) — to load the
// golden streams to bit-identical cell behavior.
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
