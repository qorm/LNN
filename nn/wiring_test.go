package nn

import (
	"math"
	"math/rand"
	"testing"

	"lnn/tensor"
)

func maskSum(t *tensor.Tensor) float32 {
	var s float32
	for _, v := range t.Data {
		s += v
	}
	return s
}

func TestFullyConnectedWiring(t *testing.T) {
	w := FullyConnected(3, 5)
	s, r := w.Sensory(), w.Recurrent()
	if s.Rows() != 3 || s.Cols() != 5 {
		t.Fatalf("sensory shape %v, want [3 5]", s.Shape)
	}
	if r.Rows() != 5 || r.Cols() != 5 {
		t.Fatalf("recurrent shape %v, want [5 5]", r.Shape)
	}
	for i, v := range s.Data {
		if v != 1 {
			t.Fatalf("sensory entry %d = %v, want 1", i, v)
		}
	}
	for i, v := range r.Data {
		if v != 1 {
			t.Fatalf("recurrent entry %d = %v, want 1", i, v)
		}
	}
	// Row accessors agree with the masks.
	if row := w.SensoryRow(2); row.Rows() != 1 || row.Cols() != 5 {
		t.Fatalf("SensoryRow shape %v, want [1 5]", row.Shape)
	}
	if row := w.RecurrentRow(4); row.Rows() != 1 || row.Cols() != 5 {
		t.Fatalf("RecurrentRow shape %v, want [1 5]", row.Shape)
	}
}

func TestRandomSparseBoundaryProbabilities(t *testing.T) {
	rng := rand.New(rand.NewSource(2))

	w0 := RandomSparse(4, 6, 0, 0, rng)
	if got := maskSum(w0.Sensory()); got != 0 {
		t.Fatalf("p=0: sensory mask sum = %v, want 0 (all synapses cut)", got)
	}
	if got := maskSum(w0.Recurrent()); got != 0 {
		t.Fatalf("p=0: recurrent mask sum = %v, want 0 (all synapses cut)", got)
	}

	w1 := RandomSparse(4, 6, 1, 1, rng)
	if got := maskSum(w1.Sensory()); got != 24 {
		t.Fatalf("p=1: sensory sum = %v, want 24 (fully connected)", got)
	}
	if got := maskSum(w1.Recurrent()); got != 36 {
		t.Fatalf("p=1: recurrent sum = %v, want 36 (fully connected)", got)
	}

	// Interior probability: every entry is strictly binary.
	w := RandomSparse(8, 8, 0.5, 0.5, rng)
	for _, v := range append(w.Sensory().Data, w.Recurrent().Data...) {
		if v != 0 && v != 1 {
			t.Fatalf("mask entry %v outside {0, 1}", v)
		}
	}
}

func TestRandomSparseRejectsBadProbability(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	bad := []float32{-0.1, 1.1, 2, -100, float32(math.NaN())}
	for _, p := range bad {
		p := p
		for _, use := range []string{"sensory", "recurrent"} {
			t.Run("", func(t *testing.T) {
				sp, rp := float32(0.5), p
				if use == "sensory" {
					sp, rp = p, 0.5
				}
				defer func() {
					if recover() == nil {
						t.Fatalf("RandomSparse with %s p=%v did not panic", use, p)
					}
				}()
				RandomSparse(2, 2, sp, rp, rng)
			})
		}
	}
}

func TestWiringRejectsEmptyDims(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	cases := []func(){
		func() { FullyConnected(0, 4) },
		func() { FullyConnected(4, 0) },
		func() { FullyConnected(-1, 4) },
		func() { RandomSparse(0, 4, 0.5, 0.5, rng) },
		func() { RandomSparse(4, 0, 0.5, 0.5, rng) },
	}
	for i, f := range cases {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatalf("case %d did not panic", i)
				}
			}()
			f()
		}()
	}
}

func TestWiringAccessorsReturnCopies(t *testing.T) {
	w := FullyConnected(2, 3)
	s := w.Sensory()
	s.Data[0] = 42 // mutate the returned copy
	if got := w.Sensory().Data[0]; got != 1 {
		t.Fatalf("mutating Sensory() copy leaked into the wiring (entry = %v)", got)
	}
	r := w.Recurrent()
	r.Data[0] = 42
	if got := w.Recurrent().Data[0]; got != 1 {
		t.Fatalf("mutating Recurrent() copy leaked into the wiring (entry = %v)", got)
	}
	// Row accessors still see the pristine mask.
	if got := w.SensoryRow(0).Data[0]; got != 1 {
		t.Fatalf("SensoryRow sees tampered mask (entry = %v)", got)
	}
}
