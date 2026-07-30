package nn

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"math/rand"
	"strings"
	"testing"

	"lnn/autograd"
	"lnn/serialize"
	"lnn/tensor"
)

// saveSameBits reports whether a and b have identical shapes and bit-identical
// payloads, comparing float32 bit patterns so NaN and -0 behave as equal to
// themselves (bit-exactness is the whole point of these tests).
func saveSameBits(a, b *tensor.Tensor) bool {
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

// saveParamsEqual asserts that two parameter lists are bit-identical.
func saveParamsEqual(t *testing.T, label string, a, b []*autograd.Variable) {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("%s: parameter count %d vs %d", label, len(a), len(b))
	}
	for i := range a {
		if !saveSameBits(a[i].Data, b[i].Data) {
			t.Errorf("%s: parameter %d differs after round trip:\n in: %v\nout: %v",
				label, i, a[i].Data.Shape, b[i].Data.Shape)
		}
	}
}

// saveInput returns a deterministic [batch, inDim] input constant.
func saveInput(batch, inDim int) *autograd.Variable {
	r := rand.New(rand.NewSource(99))
	d := make([]float32, batch*inDim)
	for i := range d {
		d[i] = r.Float32()*2 - 1
	}
	return autograd.Const(tensor.FromData(d, batch, inDim))
}

func TestSaveLoadLTCBitExact(t *testing.T) {
	cell := NewLTC(4, 6, nil, 5, rand.New(rand.NewSource(11)))

	var buf bytes.Buffer
	if err := SaveLTC(&buf, cell); err != nil {
		t.Fatalf("SaveLTC: %v", err)
	}
	loaded, err := LoadLTC(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("LoadLTC: %v", err)
	}

	// All 13 trainable parameters survive bit-exactly.
	saveParamsEqual(t, "LTC", cell.Parameters(), loaded.Parameters())
	// The fixed +/-1 reversal potentials (outside Parameters) do too.
	if !saveSameBits(cell.erev.Data, loaded.erev.Data) || !saveSameBits(cell.sErev.Data, loaded.sErev.Data) {
		t.Error("reversal potentials differ after round trip")
	}

	// Same input, same (non-zero) initial state: outputs must agree bit for
	// bit, and so must the gradients through the loaded cell's fresh graph.
	x := saveInput(3, 4)
	h := autograd.Var(tensor.FromData([]float32{
		0.1, -0.2, 0.3, 0.4, -0.5, 0.6,
		-0.1, 0.2, -0.3, -0.4, 0.5, -0.6,
		0.7, 0.0, -0.7, 0.2, 0.1, -0.3,
	}, 3, 6))
	out1, h1 := cell.Step(x, h, 0.1)
	out2, h2 := loaded.Step(x, h, 0.1)
	if !saveSameBits(out1.Data, out2.Data) {
		t.Error("LTC Step output differs after round trip")
	}
	if !saveSameBits(h1.Data, h2.Data) {
		t.Error("LTC Step state differs after round trip")
	}
	assertFinite(t, "loaded LTC output", out2, h2)

	l1 := autograd.MeanAll(autograd.Hadamard(out1, out1))
	l2 := autograd.MeanAll(autograd.Hadamard(out2, out2))
	l1.Backward()
	l2.Backward()
	p1, p2 := cell.Parameters(), loaded.Parameters()
	for i := range p1 {
		if p2[i].Grad == nil {
			t.Fatalf("loaded parameter %d has no gradient after Backward", i)
		}
		if !saveSameBits(p1[i].Grad, p2[i].Grad) {
			t.Errorf("gradient of parameter %d differs after round trip", i)
		}
	}
}

func TestSaveLoadLTCSparseWiringAndErevPreserved(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	wiring := RandomSparse(4, 6, 0.3, 0.3, rng)
	sensory, recurrent := wiring.Sensory(), wiring.Recurrent()
	// Sanity: the test is only meaningful if the masks are actually sparse.
	if z := maskSum(sensory); z == 0 || z == float32(len(sensory.Data)) {
		t.Fatalf("sensory mask is degenerate (sum %v); pick another seed", z)
	}
	cell := NewLTC(4, 6, wiring, 4, rng)

	var buf bytes.Buffer
	if err := SaveLTC(&buf, cell); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadLTC(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	ls, lr := loaded.wiring.Sensory(), loaded.wiring.Recurrent()
	if !saveSameBits(sensory, ls) || !saveSameBits(recurrent, lr) {
		t.Error("sparse wiring masks differ after round trip")
	}
	// The graph-constant mask copies used inside Step agree with the wiring.
	if !saveSameBits(ls, loaded.maskS.Data) || !saveSameBits(lr, loaded.maskR.Data) {
		t.Error("loaded cell's Step-time masks disagree with its wiring")
	}

	// Reversal potentials keep the +/-1 pattern and their exact layout.
	for name, erev := range map[string]*autograd.Variable{"erev": loaded.erev, "sErev": loaded.sErev} {
		for i, v := range erev.Data.Data {
			if v != 1 && v != -1 {
				t.Errorf("%s[%d] = %v after round trip, want +/-1", name, i, v)
			}
		}
	}
	if !saveSameBits(cell.erev.Data, loaded.erev.Data) || !saveSameBits(cell.sErev.Data, loaded.sErev.Data) {
		t.Error("erev pattern differs after round trip")
	}

	// Sparse wiring changes which synapses fire, so a masked output equality
	// check is the real proof that the masks took effect after loading.
	x := saveInput(2, 4)
	out1, _ := cell.Step(x, nil, 0.1)
	out2, _ := loaded.Step(x, nil, 0.1)
	if !saveSameBits(out1.Data, out2.Data) {
		t.Error("sparse-wired LTC output differs after round trip")
	}
}

func TestSaveLoadLTCUnrollMultiStep(t *testing.T) {
	cell := NewLTC(3, 5, nil, 6, rand.New(rand.NewSource(5)))
	var buf bytes.Buffer
	if err := SaveLTC(&buf, cell); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadLTC(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}

	xs := make([]*autograd.Variable, 4)
	for i := range xs {
		xs[i] = saveInput(2, 3)
	}
	ys1, hN1 := Unroll(cell, xs, nil, 0.05)
	ys2, hN2 := Unroll(loaded, xs, nil, 0.05)
	for i := range ys1 {
		if !saveSameBits(ys1[i].Data, ys2[i].Data) {
			t.Errorf("unroll step %d output differs after round trip", i)
		}
	}
	if !saveSameBits(hN1.Data, hN2.Data) {
		t.Error("unroll final state differs after round trip")
	}
}

func TestSaveLoadCfCBitExact(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	wiring := RandomSparse(4, 6, 0.3, 0.3, rng)
	cell := NewCfC(4, 6, wiring, rng)

	var buf bytes.Buffer
	if err := SaveCfC(&buf, cell); err != nil {
		t.Fatalf("SaveCfC: %v", err)
	}
	loaded, err := LoadCfC(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("LoadCfC: %v", err)
	}

	saveParamsEqual(t, "CfC", cell.Parameters(), loaded.Parameters())
	if !saveSameBits(cell.erev.Data, loaded.erev.Data) || !saveSameBits(cell.sErev.Data, loaded.sErev.Data) {
		t.Error("reversal potentials differ after round trip")
	}
	for i, v := range append(append([]float32{}, loaded.erev.Data.Data...), loaded.sErev.Data.Data...) {
		if v != 1 && v != -1 {
			t.Errorf("loaded reversal potential %d = %v, want +/-1", i, v)
		}
	}
	if !saveSameBits(cell.wiring.Sensory(), loaded.wiring.Sensory()) ||
		!saveSameBits(cell.wiring.Recurrent(), loaded.wiring.Recurrent()) {
		t.Error("sparse wiring masks differ after round trip")
	}

	x := saveInput(3, 4)
	out1, h1 := cell.Step(x, nil, 0.1)
	out2, h2 := loaded.Step(x, nil, 0.1)
	if !saveSameBits(out1.Data, out2.Data) || !saveSameBits(h1.Data, h2.Data) {
		t.Error("CfC Step output/state differs after round trip")
	}
	assertFinite(t, "loaded CfC output", out2, h2)

	// Gradients through the loaded cell agree with the original's.
	l1 := autograd.MeanAll(autograd.Hadamard(out1, out1))
	l2 := autograd.MeanAll(autograd.Hadamard(out2, out2))
	l1.Backward()
	l2.Backward()
	p1, p2 := cell.Parameters(), loaded.Parameters()
	for i := range p1 {
		if p2[i].Grad == nil || !saveSameBits(p1[i].Grad, p2[i].Grad) {
			t.Errorf("CfC gradient of parameter %d differs after round trip", i)
		}
	}
}

func TestSaveLoadCfCUnrollMultiStep(t *testing.T) {
	cell := NewCfC(3, 5, nil, rand.New(rand.NewSource(23)))
	var buf bytes.Buffer
	if err := SaveCfC(&buf, cell); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadCfC(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	xs := make([]*autograd.Variable, 4)
	for i := range xs {
		xs[i] = saveInput(2, 3)
	}
	ys1, hN1 := Unroll(cell, xs, nil, 0.2)
	ys2, hN2 := Unroll(loaded, xs, nil, 0.2)
	for i := range ys1 {
		if !saveSameBits(ys1[i].Data, ys2[i].Data) {
			t.Errorf("unroll step %d output differs after round trip", i)
		}
	}
	if !saveSameBits(hN1.Data, hN2.Data) {
		t.Error("unroll final state differs after round trip")
	}
}

func TestSaveLoadLinearRoundTrip(t *testing.T) {
	layer := NewLinear(3, 4, rand.New(rand.NewSource(31)))
	var buf bytes.Buffer
	if err := SaveLinear(&buf, layer); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadLinear(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if !saveSameBits(layer.W.Data, loaded.W.Data) || !saveSameBits(layer.B.Data, loaded.B.Data) {
		t.Error("Linear weights differ after round trip")
	}
	x := saveInput(5, 3)
	if !saveSameBits(layer.Forward(x).Data, loaded.Forward(x).Data) {
		t.Error("Linear forward output differs after round trip")
	}
}

func TestLoadModelCrossKindMismatch(t *testing.T) {
	ltc := NewLTC(2, 3, nil, 2, rand.New(rand.NewSource(1)))
	cfc := NewCfC(2, 3, nil, rand.New(rand.NewSource(2)))
	lin := NewLinear(2, 3, rand.New(rand.NewSource(3)))

	var ltcBuf, cfcBuf, linBuf bytes.Buffer
	if err := SaveLTC(&ltcBuf, ltc); err != nil {
		t.Fatal(err)
	}
	if err := SaveCfC(&cfcBuf, cfc); err != nil {
		t.Fatal(err)
	}
	if err := SaveLinear(&linBuf, lin); err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		name string
		load func(io.Reader) error
		data []byte
	}{
		{"LTC stream into LoadCfC", func(r io.Reader) error { _, err := LoadCfC(r); return err }, ltcBuf.Bytes()},
		{"LTC stream into LoadLinear", func(r io.Reader) error { _, err := LoadLinear(r); return err }, ltcBuf.Bytes()},
		{"CfC stream into LoadLTC", func(r io.Reader) error { _, err := LoadLTC(r); return err }, cfcBuf.Bytes()},
		{"CfC stream into LoadLinear", func(r io.Reader) error { _, err := LoadLinear(r); return err }, cfcBuf.Bytes()},
		{"Linear stream into LoadLTC", func(r io.Reader) error { _, err := LoadLTC(r); return err }, linBuf.Bytes()},
		{"garbage kind byte", func(r io.Reader) error { _, err := LoadLTC(r); return err }, []byte{99, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
		{"empty stream", func(r io.Reader) error { _, err := LoadLTC(r); return err }, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.load(bytes.NewReader(tc.data))
			if err == nil {
				t.Fatal("cross-loaded stream accepted")
			}
			if !strings.Contains(err.Error(), "kind") && !errors.Is(err, io.ErrUnexpectedEOF) {
				t.Errorf("error %q lacks semantic context (kind/EOF)", err)
			}
		})
	}
}

// writeModelStream frames ts under an explicit kind/int32 header, for forging
// streams the savers themselves would never produce.
func writeModelStream(t *testing.T, kind uint8, header []int, ts []*tensor.Tensor) []byte {
	t.Helper()
	var buf bytes.Buffer
	buf.WriteByte(kind)
	var b [4]byte
	for _, v := range header {
		binary.LittleEndian.PutUint32(b[:], uint32(int32(v)))
		buf.Write(b[:])
	}
	if err := serialize.WriteTensors(&buf, ts); err != nil {
		t.Fatalf("forging stream: %v", err)
	}
	return buf.Bytes()
}

func TestLoadLTCRejectsCorruptStreams(t *testing.T) {
	cell := NewLTC(4, 6, nil, 5, rand.New(rand.NewSource(7)))
	good := ltcTensors(cell) // masks, 13 parameters, erev, sErev

	t.Run("truncated", func(t *testing.T) {
		var buf bytes.Buffer
		if err := SaveLTC(&buf, cell); err != nil {
			t.Fatal(err)
		}
		raw := buf.Bytes()
		if _, err := LoadLTC(bytes.NewReader(raw[:len(raw)/2])); err == nil {
			t.Fatal("truncated stream accepted")
		}
	})

	t.Run("trailing bytes", func(t *testing.T) {
		var buf bytes.Buffer
		if err := SaveLTC(&buf, cell); err != nil {
			t.Fatal(err)
		}
		raw := append(buf.Bytes(), 0xDE, 0xAD)
		if _, err := LoadLTC(bytes.NewReader(raw)); err == nil || !strings.Contains(err.Error(), "trailing") {
			t.Fatalf("trailing bytes: got %v, want trailing-data error", err)
		}
	})

	t.Run("non-binary mask", func(t *testing.T) {
		ts := append([]*tensor.Tensor{}, good...)
		bad := ts[0].Clone()
		bad.Data[3] = 0.5
		ts[0] = bad
		raw := writeModelStream(t, kindLTC, []int{4, 6, 5}, ts)
		if _, err := LoadLTC(bytes.NewReader(raw)); err == nil || !strings.Contains(err.Error(), "{0, 1}") {
			t.Fatalf("non-binary mask: got %v, want {0, 1} error", err)
		}
	})

	t.Run("mask shape disagrees with header", func(t *testing.T) {
		raw := writeModelStream(t, kindLTC, []int{7, 6, 5}, good) // header says inDim=7
		if _, err := LoadLTC(bytes.NewReader(raw)); err == nil || !strings.Contains(err.Error(), "sensory mask shape") {
			t.Fatalf("mask/header mismatch: got %v, want shape error", err)
		}
	})

	t.Run("unfolds zero", func(t *testing.T) {
		raw := writeModelStream(t, kindLTC, []int{4, 6, 0}, good)
		if _, err := LoadLTC(bytes.NewReader(raw)); err == nil || !strings.Contains(err.Error(), "unfolds") {
			t.Fatalf("unfolds=0: got %v, want unfolds error", err)
		}
	})

	t.Run("zero dims", func(t *testing.T) {
		raw := writeModelStream(t, kindLTC, []int{0, 6, 5}, good)
		if _, err := LoadLTC(bytes.NewReader(raw)); err == nil || !strings.Contains(err.Error(), "dims") {
			t.Fatalf("inDim=0: got %v, want dims error", err)
		}
	})

	t.Run("wrong tensor count", func(t *testing.T) {
		raw := writeModelStream(t, kindLTC, []int{4, 6, 5}, good[:len(good)-1])
		if _, err := LoadLTC(bytes.NewReader(raw)); err == nil || !strings.Contains(err.Error(), "16 tensors, want 17") {
			t.Fatalf("short blob: got %v, want count error", err)
		}
	})

	t.Run("parameter shape mismatch", func(t *testing.T) {
		ts := append([]*tensor.Tensor{}, good...)
		ts[2] = tensor.FromData(ts[2].Data, 2, 3) // gleak [6] -> [2,3]: same size, wrong layout
		raw := writeModelStream(t, kindLTC, []int{4, 6, 5}, ts)
		if _, err := LoadLTC(bytes.NewReader(raw)); err == nil || !strings.Contains(err.Error(), "shape mismatch") {
			t.Fatalf("gleak layout swap: got %v, want shape mismatch", err)
		}
	})
}

func TestLoadCfCRejectsCorruptStreams(t *testing.T) {
	cell := NewCfC(4, 6, nil, rand.New(rand.NewSource(13)))
	good := cfcTensors(cell)

	t.Run("wrong tensor count", func(t *testing.T) {
		raw := writeModelStream(t, kindCfC, []int{4, 6}, good[:5])
		if _, err := LoadCfC(bytes.NewReader(raw)); err == nil || !strings.Contains(err.Error(), "5 tensors, want 17") {
			t.Fatalf("short blob: got %v, want count error", err)
		}
	})
	t.Run("bad dims", func(t *testing.T) {
		raw := writeModelStream(t, kindCfC, []int{4, -1}, good)
		if _, err := LoadCfC(bytes.NewReader(raw)); err == nil || !strings.Contains(err.Error(), "dims") {
			t.Fatalf("units=-1: got %v, want dims error", err)
		}
	})
	t.Run("truncated header", func(t *testing.T) {
		if _, err := LoadCfC(bytes.NewReader([]byte{kindCfC, 0x04})); !errors.Is(err, io.ErrUnexpectedEOF) {
			t.Fatalf("mid-header cut: got %v, want ErrUnexpectedEOF", err)
		}
	})
	t.Run("bad sensory mask shape", func(t *testing.T) {
		ts := append([]*tensor.Tensor{}, good...)
		ts[0] = tensor.New(5, 6)
		raw := writeModelStream(t, kindCfC, []int{4, 6}, ts)
		if _, err := LoadCfC(bytes.NewReader(raw)); err == nil || !strings.Contains(err.Error(), "sensory mask shape") {
			t.Fatalf("mask shape: got %v, want shape error", err)
		}
	})
	t.Run("parameter shape mismatch", func(t *testing.T) {
		ts := append([]*tensor.Tensor{}, good...)
		ts[5] = tensor.FromData(ts[5].Data, 36, 1) // mu [6,6] -> [36,1]
		raw := writeModelStream(t, kindCfC, []int{4, 6}, ts)
		if _, err := LoadCfC(bytes.NewReader(raw)); err == nil || !strings.Contains(err.Error(), "shape mismatch") {
			t.Fatalf("mu layout swap: got %v, want shape mismatch", err)
		}
	})
	t.Run("truncated blob after good header", func(t *testing.T) {
		raw := writeModelStream(t, kindCfC, []int{4, 6}, good)
		if _, err := LoadCfC(bytes.NewReader(raw[:len(raw)-3])); !errors.Is(err, io.ErrUnexpectedEOF) {
			t.Fatalf("truncated blob: got %v, want ErrUnexpectedEOF", err)
		}
	})
}

// failWriter accepts nothing: every Save must report its I/O failure.
type failWriter struct{}

func (failWriter) Write([]byte) (int, error) { return 0, errors.New("disk on fire") }

// failReader fails every read: every Load must surface the error, not panic.
type failReader struct{}

func (failReader) Read([]byte) (int, error) { return 0, errors.New("socket reset") }

func TestSaveLoadReportIOErrors(t *testing.T) {
	ltc := NewLTC(2, 3, nil, 2, rand.New(rand.NewSource(41)))
	cfc := NewCfC(2, 3, nil, rand.New(rand.NewSource(42)))
	lin := NewLinear(2, 3, rand.New(rand.NewSource(43)))

	var buf bytes.Buffer
	if err := SaveLTC(failWriter{}, ltc); err == nil {
		t.Error("SaveLTC swallowed a write failure")
	}
	if err := SaveCfC(failWriter{}, cfc); err == nil {
		t.Error("SaveCfC swallowed a write failure")
	}
	if err := SaveLinear(failWriter{}, lin); err == nil {
		t.Error("SaveLinear swallowed a write failure")
	}
	if err := SaveLTC(&buf, ltc); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadLTC(failReader{}); err == nil {
		t.Error("LoadLTC swallowed a read failure")
	}
	if _, err := LoadCfC(bytes.NewReader(buf.Bytes()[:1])); err == nil {
		t.Error("LoadCfC accepted a one-byte stream")
	}
	if _, err := LoadLinear(bytes.NewReader(nil)); err == nil {
		t.Error("LoadLinear accepted an empty stream")
	}
	// Kind byte present, header cut mid-field.
	if _, err := LoadLTC(bytes.NewReader([]byte{kindLTC, 0x01})); err == nil ||
		!errors.Is(err, io.ErrUnexpectedEOF) {
		t.Errorf("mid-header cut: got %v, want ErrUnexpectedEOF", err)
	}
}

func TestHeaderBounds(t *testing.T) {
	var buf bytes.Buffer
	hw := &headerWriter{w: &buf}
	hw.i32(-1)
	if hw.err == nil {
		t.Error("negative header value accepted")
	}
	hw = &headerWriter{w: &buf}
	hw.i32(int(1) << 40)
	if hw.err == nil {
		t.Error("over-int32 header value accepted")
	}
	// After a failed write the writer must stay dead, not retry.
	hw = &headerWriter{w: failWriter{}}
	hw.u8(kindLTC)
	hw.u8(kindCfC)
	if hw.err == nil {
		t.Error("failed writer did not capture the error")
	}
}

func TestLoadLTCRejectsNonBinaryRecurrentMask(t *testing.T) {
	cell := NewLTC(4, 6, nil, 5, rand.New(rand.NewSource(47)))
	ts := ltcTensors(cell)
	bad := ts[1].Clone()
	bad.Data[10] = 2
	ts[1] = bad
	raw := writeModelStream(t, kindLTC, []int{4, 6, 5}, ts)
	if _, err := LoadLTC(bytes.NewReader(raw)); err == nil || !strings.Contains(err.Error(), "recurrent mask entry") {
		t.Fatalf("non-binary recurrent mask: got %v, want recurrent mask error", err)
	}
}

func TestLoadLinearRejectsBadShapes(t *testing.T) {
	w3d := tensor.New(2, 2, 2)
	raw := writeModelStream(t, kindLinear, nil, []*tensor.Tensor{w3d, tensor.New(2)})
	if _, err := LoadLinear(bytes.NewReader(raw)); err == nil || !strings.Contains(err.Error(), "2D") {
		t.Fatalf("3D weight: got %v, want 2D error", err)
	}
	raw = writeModelStream(t, kindLinear, nil, []*tensor.Tensor{tensor.New(2, 3), tensor.New(4)})
	if _, err := LoadLinear(bytes.NewReader(raw)); err == nil || !strings.Contains(err.Error(), "bias") {
		t.Fatalf("bias mismatch: got %v, want bias error", err)
	}
	raw = writeModelStream(t, kindLinear, nil, []*tensor.Tensor{tensor.New(2, 3)})
	if _, err := LoadLinear(bytes.NewReader(raw)); err == nil || !strings.Contains(err.Error(), "1 tensors, want 2") {
		t.Fatalf("missing bias tensor: got %v, want count error", err)
	}
	// Valid kind byte, garbage where the tensor blob should be.
	if _, err := LoadLinear(bytes.NewReader([]byte{kindLinear, 'g', 'a', 'r', 'b'})); err == nil ||
		!strings.Contains(err.Error(), "magic") {
		t.Fatalf("garbage blob: got %v, want magic error", err)
	}
}

func TestLoadLTCRejectsRecurrentMaskShape(t *testing.T) {
	cell := NewLTC(4, 6, nil, 5, rand.New(rand.NewSource(53)))
	ts := ltcTensors(cell)
	ts[1] = tensor.New(5, 5)
	raw := writeModelStream(t, kindLTC, []int{4, 6, 5}, ts)
	if _, err := LoadLTC(bytes.NewReader(raw)); err == nil || !strings.Contains(err.Error(), "recurrent mask shape") {
		t.Fatalf("recurrent mask shape: got %v, want shape error", err)
	}
}
