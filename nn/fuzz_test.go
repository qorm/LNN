// Native Go fuzzing for nn's model-level load paths (doc/persistence.md):
// LoadLTC / LoadCfC / LoadLinear treat their input as an untrusted byte
// stream — every failure is an error, never a panic; the header size limits
// (units/inDim <= 2048, unfolds <= 1024) cannot be bypassed; and a failed load
// leaves the world exactly as it was (no side effects on construction/save).
//
// External test package (nn_test): the targets drive only the exported API and
// forge hostile streams by byte-level surgery on legitimate Save* output, so
// they need none of save_test.go's private helpers.
package nn_test

import (
	"bytes"
	"encoding/binary"
	"math/rand"
	"testing"

	"github.com/qorm/LNN/nn"
)

// Model stream envelope sizes (kind byte + int32 header fields), per
// doc/persistence.md: LTC writes inDim, units, unfolds; CfC writes inDim,
// units; Linear writes none. The serialize "LNNS" blob follows immediately.
const (
	fuzzEnvLTC    = 1 + 3*4
	fuzzEnvCfC    = 1 + 2*4
	fuzzEnvLinear = 1
)

// Load-path limits asserted on every successful load (doc/persistence.md).
const (
	fuzzMaxUnits   = 2048
	fuzzMaxInDim   = 2048
	fuzzMaxUnfolds = 1024
)

// fuzzPatchLE32 returns a copy of b with the little-endian uint32 at off set
// to v — the byte-surgery primitive for forging hostile headers/counts.
func fuzzPatchLE32(b []byte, off int, v uint32) []byte {
	out := append([]byte(nil), b...)
	binary.LittleEndian.PutUint32(out[off:off+4], v)
	return out
}

// fuzzSetByte returns a copy of b with b[off] = v.
func fuzzSetByte(b []byte, off int, v byte) []byte {
	out := append([]byte(nil), b...)
	out[off] = v
	return out
}

// Golden-equivalent cells: the exact documented construction behind the
// committed serialize/testdata golden vectors, so the seeds here are the
// golden streams re-emitted in code (no cross-package file dependency).
func fuzzGoldenLTC() *nn.LTC       { return nn.NewLTC(4, 6, nil, 6, rand.New(rand.NewSource(101))) }
func fuzzGoldenCfC() *nn.CfC       { return nn.NewCfC(4, 6, nil, rand.New(rand.NewSource(202))) }
func fuzzGoldenLinear() *nn.Linear { return nn.NewLinear(6, 3, rand.New(rand.NewSource(303))) }

func fuzzSaveLTC(t testing.TB, c *nn.LTC) []byte {
	var buf bytes.Buffer
	if err := nn.SaveLTC(&buf, c); err != nil {
		t.Fatalf("SaveLTC: %v", err)
	}
	return buf.Bytes()
}

func fuzzSaveCfC(t testing.TB, c *nn.CfC) []byte {
	var buf bytes.Buffer
	if err := nn.SaveCfC(&buf, c); err != nil {
		t.Fatalf("SaveCfC: %v", err)
	}
	return buf.Bytes()
}

func fuzzSaveLinear(t testing.TB, l *nn.Linear) []byte {
	var buf bytes.Buffer
	if err := nn.SaveLinear(&buf, l); err != nil {
		t.Fatalf("SaveLinear: %v", err)
	}
	return buf.Bytes()
}

// fuzzModelSeeds returns the shared seed set for a cell loader: the legitimate
// golden-equivalent stream, the two cross-kind streams (must be rejected on
// the kind byte), empty/tiny streams, and the hand-forged red-team header
// mutations (each comment names the attack intent). makeLegit produces the
// legitimate stream; crossA/crossB are the other two kinds' streams; patch is
// a callback that forges the kind-specific hostile headers from the legit
// stream (it knows the envelope layout).
func fuzzModelSeeds(f *testing.F, legit, crossA, crossB []byte, hostile ...[]byte) {
	f.Add(legit)                                 // legitimate golden-equivalent stream: must load
	f.Add(crossA)                                // wrong-kind stream: must error on the kind byte
	f.Add(crossB)                                // wrong-kind stream: must error on the kind byte
	f.Add([]byte{})                              // empty stream: EOF
	f.Add([]byte{0})                             // kind byte only, then EOF
	f.Add([]byte{99, 1, 2, 3, 4, 5, 6, 7, 8, 9}) // garbage kind byte
	for _, h := range hostile {
		f.Add(h)
	}
}

func FuzzLoadLTC(f *testing.F) {
	legit := fuzzSaveLTC(f, fuzzGoldenLTC())
	cfc := fuzzSaveCfC(f, fuzzGoldenCfC())
	lin := fuzzSaveLinear(f, fuzzGoldenLinear())
	// blob count field sits at envelope + 5 (magic 4 + version 1).
	blobCount := fuzzEnvLTC + 5
	hostile := [][]byte{
		fuzzSetByte(legit, 0, 99),                         // invalid kind byte
		fuzzPatchLE32(legit, 5, 4096),                     // units=4096 (over the 2048 cap)
		fuzzPatchLE32(legit, 5, 0),                        // units=0
		fuzzPatchLE32(legit, 1, 0),                        // inDim=0
		fuzzPatchLE32(legit, 9, 1<<20),                    // unfolds=1<<20 (CPU exhaustion)
		fuzzPatchLE32(legit, 9, 0),                        // unfolds=0
		fuzzPatchLE32(legit, blobCount, 0xFFFFFFFF),       // blob count = 2^32-1
		legit[:len(legit)/2],                              // mid-tensor truncation
		append(append([]byte(nil), legit...), 0xDE, 0xAD), // trailing bytes
	}
	fuzzModelSeeds(f, legit, cfc, lin, hostile...)

	reference := func() []byte {
		return fuzzSaveLTC(f, nn.NewLTC(2, 3, nil, 2, rand.New(rand.NewSource(7))))
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		// Side-effect control: a reference save taken BEFORE the load must be
		// reproduced byte for byte afterwards, so a failing load is provably
		// unable to perturb construction/serialization state.
		refBefore := reference()
		var cell *nn.LTC
		var err error
		func() {
			defer func() {
				if p := recover(); p != nil {
					t.Fatalf("LoadLTC panicked: %v", p)
				}
			}()
			cell, err = nn.LoadLTC(bytes.NewReader(data))
		}()
		if err != nil {
			if cell != nil {
				t.Fatalf("LoadLTC returned a cell with an error")
			}
			if got := reference(); !bytes.Equal(got, refBefore) {
				t.Fatalf("failed load caused a side effect: reference save drifted (%d -> %d bytes)", len(refBefore), len(got))
			}
			return
		}
		// Successful load: the limits must not have been bypassed. Re-save and
		// parse the header, then cross-check against the parameter shapes.
		saved := fuzzSaveLTC(t, cell)
		inDim := int(int32(binary.LittleEndian.Uint32(saved[1:5])))
		units := int(int32(binary.LittleEndian.Uint32(saved[5:9])))
		unfolds := int(int32(binary.LittleEndian.Uint32(saved[9:13])))
		if inDim < 1 || inDim > fuzzMaxInDim {
			t.Fatalf("loaded inDim=%d outside [1, %d]", inDim, fuzzMaxInDim)
		}
		if units < 1 || units > fuzzMaxUnits {
			t.Fatalf("loaded units=%d outside [1, %d]", units, fuzzMaxUnits)
		}
		if unfolds < 1 || unfolds > fuzzMaxUnfolds {
			t.Fatalf("loaded unfolds=%d outside [1, %d]", unfolds, fuzzMaxUnfolds)
		}
		params := cell.Parameters()
		if got := params[0].Data.Shape[0]; got != units { // gleak is [units]
			t.Fatalf("gleak shape %d disagrees with header units=%d", got, units)
		}
		if got := params[9].Data.Shape[0]; got != inDim { // inW is [inDim]
			t.Fatalf("inW shape %d disagrees with header inDim=%d", got, inDim)
		}
		// Re-save idempotence: load the re-saved stream and require identical bytes.
		cell2, err := nn.LoadLTC(bytes.NewReader(saved))
		if err != nil {
			t.Fatalf("re-saved stream does not reload: %v", err)
		}
		if got := fuzzSaveLTC(t, cell2); !bytes.Equal(got, saved) {
			t.Fatalf("LTC save is not idempotent across load (%d -> %d bytes)", len(saved), len(got))
		}
	})
}

func FuzzLoadCfC(f *testing.F) {
	legit := fuzzSaveCfC(f, fuzzGoldenCfC())
	ltc := fuzzSaveLTC(f, fuzzGoldenLTC())
	lin := fuzzSaveLinear(f, fuzzGoldenLinear())
	blobCount := fuzzEnvCfC + 5
	hostile := [][]byte{
		fuzzSetByte(legit, 0, 99),
		fuzzPatchLE32(legit, 5, 4096),       // units=4096
		fuzzPatchLE32(legit, 5, 0),          // units=0
		fuzzPatchLE32(legit, 1, 0),          // inDim=0
		fuzzPatchLE32(legit, 1, 0xFFFFFFFF), // inDim = -1 (as int32)
		fuzzPatchLE32(legit, blobCount, 0xFFFFFFFF),
		legit[:len(legit)/2],
		legit[:2], // mid-header truncation
		append(append([]byte(nil), legit...), 0x00),
	}
	fuzzModelSeeds(f, legit, ltc, lin, hostile...)

	reference := func() []byte {
		return fuzzSaveCfC(f, nn.NewCfC(2, 3, nil, rand.New(rand.NewSource(7))))
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		refBefore := reference()
		var cell *nn.CfC
		var err error
		func() {
			defer func() {
				if p := recover(); p != nil {
					t.Fatalf("LoadCfC panicked: %v", p)
				}
			}()
			cell, err = nn.LoadCfC(bytes.NewReader(data))
		}()
		if err != nil {
			if cell != nil {
				t.Fatalf("LoadCfC returned a cell with an error")
			}
			if got := reference(); !bytes.Equal(got, refBefore) {
				t.Fatalf("failed load caused a side effect: reference save drifted (%d -> %d bytes)", len(refBefore), len(got))
			}
			return
		}
		saved := fuzzSaveCfC(t, cell)
		inDim := int(int32(binary.LittleEndian.Uint32(saved[1:5])))
		units := int(int32(binary.LittleEndian.Uint32(saved[5:9])))
		if inDim < 1 || inDim > fuzzMaxInDim {
			t.Fatalf("loaded inDim=%d outside [1, %d]", inDim, fuzzMaxInDim)
		}
		if units < 1 || units > fuzzMaxUnits {
			t.Fatalf("loaded units=%d outside [1, %d]", units, fuzzMaxUnits)
		}
		params := cell.Parameters()
		if got := params[0].Data.Shape[0]; got != units {
			t.Fatalf("gleak shape %d disagrees with header units=%d", got, units)
		}
		if got := params[9].Data.Shape[0]; got != inDim {
			t.Fatalf("inW shape %d disagrees with header inDim=%d", got, inDim)
		}
		cell2, err := nn.LoadCfC(bytes.NewReader(saved))
		if err != nil {
			t.Fatalf("re-saved stream does not reload: %v", err)
		}
		if got := fuzzSaveCfC(t, cell2); !bytes.Equal(got, saved) {
			t.Fatalf("CfC save is not idempotent across load")
		}
	})
}

func FuzzLoadLinear(f *testing.F) {
	legit := fuzzSaveLinear(f, fuzzGoldenLinear())
	ltc := fuzzSaveLTC(f, fuzzGoldenLTC())
	cfc := fuzzSaveCfC(f, fuzzGoldenCfC())
	blobCount := fuzzEnvLinear + 5
	hostile := [][]byte{
		fuzzSetByte(legit, 0, 99),
		fuzzPatchLE32(legit, blobCount, 0xFFFFFFFF),
		legit[:len(legit)/2],
		append(append([]byte(nil), legit...), 0xAB),
	}
	fuzzModelSeeds(f, legit, ltc, cfc, hostile...)

	reference := func() []byte {
		return fuzzSaveLinear(f, nn.NewLinear(2, 3, rand.New(rand.NewSource(7))))
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		refBefore := reference()
		var layer *nn.Linear
		var err error
		func() {
			defer func() {
				if p := recover(); p != nil {
					t.Fatalf("LoadLinear panicked: %v", p)
				}
			}()
			layer, err = nn.LoadLinear(bytes.NewReader(data))
		}()
		if err != nil {
			if layer != nil {
				t.Fatalf("LoadLinear returned a layer with an error")
			}
			if got := reference(); !bytes.Equal(got, refBefore) {
				t.Fatalf("failed load caused a side effect: reference save drifted (%d -> %d bytes)", len(refBefore), len(got))
			}
			return
		}
		// Successful load: W must be 2D and B must match W's column count.
		wShape := layer.W.Data.Shape
		bShape := layer.B.Data.Shape
		if len(wShape) != 2 {
			t.Fatalf("loaded weight is %dD, want 2D", len(wShape))
		}
		if len(bShape) != 1 || bShape[0] != wShape[1] {
			t.Fatalf("loaded bias shape %v does not match weight columns %d", bShape, wShape[1])
		}
		saved := fuzzSaveLinear(t, layer)
		layer2, err := nn.LoadLinear(bytes.NewReader(saved))
		if err != nil {
			t.Fatalf("re-saved stream does not reload: %v", err)
		}
		if got := fuzzSaveLinear(t, layer2); !bytes.Equal(got, saved) {
			t.Fatalf("Linear save is not idempotent across load")
		}
	})
}
