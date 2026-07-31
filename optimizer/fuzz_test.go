// Native Go fuzzing for optimizer.LoadState (doc/persistence.md, the "LNO1"
// state stream): every failure is an error, never a panic; a kind mismatch is
// always an error; and — the validate-all-then-apply discipline — a failed
// load leaves the destination optimizer bit for bit as it was (proved by
// re-saving it before and after).
//
// External test package (optimizer_test), driving only the exported API and
// forging hostile streams by hand and by byte surgery on legitimate saves.
package optimizer_test

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/optimizer"
	"github.com/qorm/LNN/tensor"
)

// State stream kind tags and header layout (doc/persistence.md): magic "LNO1"
// (4) + version (1) + kind (1) + count (4); the kind byte is therefore at
// offset 5 and the count at offset 6.
const (
	fuzzKindSGD      = 0
	fuzzKindMomentum = 1
	fuzzKindAdam     = 2
)

func fuzzStateHeader(kind byte, count uint32) []byte {
	b := make([]byte, 10)
	copy(b, "LNO1")
	b[4] = 1 // version
	b[5] = kind
	binary.LittleEndian.PutUint32(b[6:], count)
	return b
}

// fuzzParams returns a fixed two-parameter destination ([2] and [3]) with
// gradients set, so Momentum/Adam have state to persist.
func fuzzParams() []*autograd.Variable {
	p0 := autograd.Var(tensor.FromData([]float32{1, -2}, 2))
	p0.Grad = tensor.FromData([]float32{0.5, -0.25}, 2)
	p1 := autograd.Var(tensor.FromData([]float32{3, 4, -5}, 3))
	p1.Grad = tensor.FromData([]float32{1, -1, 0.125}, 3)
	return []*autograd.Variable{p0, p1}
}

// fuzzSaveState saves o's state over params, failing the test on error.
func fuzzSaveState(t testing.TB, o optimizer.Optimizer, params []*autograd.Variable) []byte {
	var buf bytes.Buffer
	if err := optimizer.SaveState(&buf, o, params); err != nil {
		t.Fatalf("SaveState: %v", err)
	}
	return buf.Bytes()
}

// fuzzStepped builds an optimizer of each kind with a few updates accumulated,
// so its state stream is non-trivial. newOpt constructs the optimizer; steps
// runs a handful of updates over params (re-setting gradients each time).
func fuzzStepped(newOpt func() optimizer.Optimizer, params []*autograd.Variable) optimizer.Optimizer {
	o := newOpt()
	for i := 0; i < 3; i++ {
		for _, p := range params {
			if p.Grad == nil {
				p.Grad = tensor.New(p.Data.Shape...)
			}
			for j := range p.Grad.Data {
				p.Grad.Data[j] = float32(i+1) * 0.5
			}
		}
		o.Step(params)
	}
	return o
}

// FuzzLoadState feeds arbitrary byte streams to LoadState against all three
// persisted optimizer kinds. Oracle:
//   - never panics;
//   - if the stream carries a valid magic, version and a kind that differs
//     from the target optimizer's, the load must error (kind mismatch);
//   - on error the destination optimizer is untouched: its re-save before and
//     after the failed load is byte-identical (validate-all-then-apply);
//   - a successful load into an optimizer of the matching kind re-saves
//     without error.
func FuzzLoadState(f *testing.F) {
	params := fuzzParams()
	sgd := fuzzSaveState(f, fuzzStepped(func() optimizer.Optimizer { return optimizer.NewSGD(0.1) }, params), params)
	mom := fuzzSaveState(f, fuzzStepped(func() optimizer.Optimizer { return optimizer.NewMomentum(0.1, 0.9) }, params), params)
	adam := fuzzSaveState(f, fuzzStepped(func() optimizer.Optimizer { return optimizer.NewAdamDefault(0.1) }, params), params)

	// (1) Legitimate streams of all three kinds (each is a cross-kind hostile
	// input for the other two). (2) Empty/tiny. (3) Hand-forged red-team cases.
	f.Add(sgd)
	f.Add(mom)
	f.Add(adam)
	f.Add([]byte{})
	f.Add([]byte("LNO1"))
	f.Add([]byte{'X', 'X', 'X', 'X', 1, 0, 0, 0, 0, 0}) // bad magic
	f.Add(fuzzStateHeader(fuzzKindMomentum, 1))         // header only, then EOF
	// version 99 / 0
	f.Add(func() []byte { b := fuzzStateHeader(fuzzKindAdam, 1); b[4] = 99; return b }())
	f.Add(func() []byte { b := fuzzStateHeader(fuzzKindAdam, 1); b[4] = 0; return b }())
	f.Add(fuzzStateHeader(fuzzKindAdam, 0xFFFFFFFF))       // count = 2^32-1
	f.Add(append(fuzzStateHeader(fuzzKindMomentum, 1), 2)) // presence flag 2 outside {0,1}
	f.Add(func() []byte {                                  // Adam record with t = maxUint32 (over the maxT load limit)
		b := fuzzStateHeader(fuzzKindAdam, 1)
		b = append(b, 1) // present
		var t [4]byte
		binary.LittleEndian.PutUint32(t[:], math.MaxUint32)
		b = append(b, t[:]...)
		// pow1/pow2 must be present so the record read completes and the
		// t-limit check (not a truncation) is what fires.
		b = append(b, 0, 0, 0, 0, 0, 0, 0, 0)
		return b
	}())
	f.Add(func() []byte { // Momentum header + a blob claiming a 1<<62-wide tensor
		b := fuzzStateHeader(fuzzKindMomentum, 1)
		b = append(b, 1) // present
		// blob: "LNNS" magic, version 1, count=1, rank=1 — then a 1<<62 axis.
		blob := []byte{'L', 'N', 'N', 'S', 1, 1, 0, 0, 0, 1}
		var dim [8]byte
		binary.LittleEndian.PutUint64(dim[:], 1<<62)
		b = append(b, blob...)
		return append(b, dim[:]...)
	}())
	f.Add(append(append([]byte(nil), adam...), 0x00)) // trailing byte
	f.Add(mom[:len(mom)/2])                           // mid-stream truncation

	kinds := []struct {
		name string
		kind byte
		new  func() optimizer.Optimizer
	}{
		{"SGD", fuzzKindSGD, func() optimizer.Optimizer { return optimizer.NewSGD(0.1) }},
		{"Momentum", fuzzKindMomentum, func() optimizer.Optimizer { return optimizer.NewMomentum(0.1, 0.9) }},
		{"Adam", fuzzKindAdam, func() optimizer.Optimizer { return optimizer.NewAdamDefault(0.1) }},
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		for _, k := range kinds {
			// Destination with pre-existing state, keyed over params. State is
			// keyed by parameter pointer, so the same params are reused for the
			// before/after saves and the load itself.
			params := fuzzParams()
			dst := fuzzStepped(k.new, params)
			before := fuzzSaveState(t, dst, params)

			var err error
			func() {
				defer func() {
					if p := recover(); p != nil {
						t.Fatalf("%s: LoadState panicked: %v", k.name, p)
					}
				}()
				err = optimizer.LoadState(bytes.NewReader(data), dst, params)
			}()

			// Kind mismatch must be an error whenever the header is readable and
			// its kind differs from the target optimizer's.
			if len(data) >= 6 && string(data[:4]) == "LNO1" && data[4] == 1 && data[5] != k.kind {
				if err == nil {
					t.Fatalf("%s: accepted a stream of kind %d", k.name, data[5])
				}
			}

			if err != nil {
				// Validate-all-then-apply: the destination must be bit for bit
				// as it was, proved by a byte-identical re-save.
				after := fuzzSaveState(t, dst, params)
				if !bytes.Equal(after, before) {
					t.Fatalf("%s: failed load mutated the destination optimizer (%d -> %d bytes)",
						k.name, len(before), len(after))
				}
			}
		}
	})
}
