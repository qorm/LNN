package nn

import (
	"fmt"
	"math/rand"

	"lnn/tensor"
)

// Wiring describes the synapse topology of an LTC cell as binary masks.
// The sensory mask [inDim, units] gates synapses from inputs to neurons and
// the recurrent mask [units, units] gates synapses between neurons (entry
// [i, j] is the synapse from neuron i to neuron j).
//
// Masks are unexported and only ever exposed as copies, so a Wiring cannot
// be mutated after construction (externally tampered masks used to be a
// silent-corruption hazard). All constructors assert the {0, 1} value
// invariant and reject empty dimensions.
type Wiring struct {
	sensoryMask   *tensor.Tensor
	recurrentMask *tensor.Tensor
}

// FullyConnected returns a wiring where every synapse exists. It panics if
// inDim or units is less than 1.
func FullyConnected(inDim, units int) *Wiring {
	checkWiringDims(inDim, units)
	return newWiring(
		tensor.New(inDim, units).OnesLike(),
		tensor.New(units, units).OnesLike(),
	)
}

// RandomSparse returns a wiring where each synapse exists independently with
// the given connection probability. Both probabilities must lie in [0, 1]
// (NaN is rejected), and inDim and units must be at least 1.
func RandomSparse(inDim, units int, sensoryP, recurrentP float32, rng *rand.Rand) *Wiring {
	checkWiringDims(inDim, units)
	checkProbability("sensoryP", sensoryP)
	checkProbability("recurrentP", recurrentP)
	mask := func(p float32, shape ...int) *tensor.Tensor {
		m := tensor.New(shape...)
		for i := range m.Data {
			if rng.Float32() < p {
				m.Data[i] = 1
			}
		}
		return m
	}
	return newWiring(
		mask(sensoryP, inDim, units),
		mask(recurrentP, units, units),
	)
}

func checkWiringDims(inDim, units int) {
	if inDim < 1 || units < 1 {
		panic(fmt.Sprintf("nn: wiring dimensions must be >= 1, got inDim=%d units=%d", inDim, units))
	}
}

func checkProbability(name string, p float32) {
	// NaN fails both comparisons and is rejected along with out-of-range values.
	if !(p >= 0 && p <= 1) {
		panic(fmt.Sprintf("nn: %s must be in [0, 1], got %v", name, p))
	}
}

// newWiring is the single construction path shared by all wiring constructors;
// it asserts the binary mask invariant.
func newWiring(sensory, recurrent *tensor.Tensor) *Wiring {
	assertBinaryMask("sensory", sensory)
	assertBinaryMask("recurrent", recurrent)
	return &Wiring{sensoryMask: sensory, recurrentMask: recurrent}
}

func assertBinaryMask(name string, m *tensor.Tensor) {
	for _, v := range m.Data {
		if v != 0 && v != 1 {
			panic(fmt.Sprintf("nn: %s mask entry %v outside {0, 1}", name, v))
		}
	}
}

// Sensory returns a copy of the sensory mask [inDim, units]. Mutating the
// returned tensor does not affect the wiring.
func (w *Wiring) Sensory() *tensor.Tensor { return w.sensoryMask.Clone() }

// Recurrent returns a copy of the recurrent mask [units, units]. Mutating the
// returned tensor does not affect the wiring.
func (w *Wiring) Recurrent() *tensor.Tensor { return w.recurrentMask.Clone() }

// SensoryRow returns row i of the sensory mask with shape [1, units].
func (w *Wiring) SensoryRow(i int) *tensor.Tensor { return tensor.SliceRow(w.sensoryMask, i) }

// RecurrentRow returns row i of the recurrent mask with shape [1, units].
func (w *Wiring) RecurrentRow(i int) *tensor.Tensor { return tensor.SliceRow(w.recurrentMask, i) }
