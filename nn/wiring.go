package nn

import (
	"math/rand"

	"lnn/tensor"
)

// Wiring describes the synapse topology of an LTC cell as binary masks.
// SensoryMask[inDim, units] gates synapses from inputs to neurons and
// RecurrentMask[units, units] gates synapses between neurons (entry [i, j]
// is the synapse from neuron i to neuron j).
type Wiring struct {
	SensoryMask   *tensor.Tensor
	RecurrentMask *tensor.Tensor
}

// FullyConnected returns a wiring where every synapse exists.
func FullyConnected(inDim, units int) *Wiring {
	return &Wiring{
		SensoryMask:   tensor.New(inDim, units).OnesLike(),
		RecurrentMask: tensor.New(units, units).OnesLike(),
	}
}

// RandomSparse returns a wiring where each synapse exists independently with
// the given connection probability.
func RandomSparse(inDim, units int, sensoryP, recurrentP float32, rng *rand.Rand) *Wiring {
	mask := func(p float32, shape ...int) *tensor.Tensor {
		m := tensor.New(shape...)
		for i := range m.Data {
			if rng.Float32() < p {
				m.Data[i] = 1
			}
		}
		return m
	}
	return &Wiring{
		SensoryMask:   mask(sensoryP, inDim, units),
		RecurrentMask: mask(recurrentP, units, units),
	}
}

// SensoryRow returns row i of the sensory mask with shape [1, units].
func (w *Wiring) SensoryRow(i int) *tensor.Tensor { return tensor.SliceRow(w.SensoryMask, i) }

// RecurrentRow returns row i of the recurrent mask with shape [1, units].
func (w *Wiring) RecurrentRow(i int) *tensor.Tensor { return tensor.SliceRow(w.RecurrentMask, i) }
