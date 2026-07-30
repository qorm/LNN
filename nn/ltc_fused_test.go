package nn

import (
	"math"
	"math/rand"
	"testing"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/tensor"
)

// legacySynapses reproduces synapses/synapsesRows exactly as they ran before
// the Sigmoid-Hadamard fusion: the identical per-presynaptic structure
// (SliceRow/Col/Sub/Hadamard, ConcatCol, indicator MatMul reductions), with
// the activation block built from the separate Sigmoid and Hadamard nodes
// instead of the fused autograd.SigmoidHadamard node. Differential tests
// below pin the fused path to this oracle.
func legacySynapses(
	pre, mu, sigma, wm, denReduce, numReduce *autograd.Variable,
) (num, den *autograd.Variable) {
	n := pre.Data.Cols()
	muRs := make([]*autograd.Variable, n)
	sigRs := make([]*autograd.Variable, n)
	wmRs := make([]*autograd.Variable, n)
	for i := 0; i < n; i++ {
		muRs[i] = autograd.SliceRow(mu, i)
		sigRs[i] = autograd.SliceRow(sigma, i)
		wmRs[i] = autograd.SliceRow(wm, i)
	}
	blocks := make([]*autograd.Variable, n)
	for i := range blocks {
		preCol := autograd.Col(pre, i)
		z := autograd.Hadamard(sigRs[i], autograd.Sub(preCol, muRs[i]))
		blocks[i] = autograd.Hadamard(autograd.Sigmoid(z), wmRs[i])
	}
	flat := blocks[0]
	if len(blocks) > 1 {
		flat = autograd.ConcatCol(blocks...)
		den = autograd.MatMul(flat, denReduce)
	} else {
		den = flat
	}
	num = autograd.MatMul(flat, numReduce)
	return num, den
}

// TestLTCSynapsesFusedBackwardEquivalence checks that adopting the fused
// SigmoidHadamard node in the synapse hot path leaves the gradients of every
// participating leaf bit-identical to the former Sigmoid+Hadamard pair. The
// fused backward rounds each product at the same spots the two-node chain
// did, so strict Float32bits equality — not tolerance — is the bar, on both
// the multi-source recurrent path and the single-source sensory path (whose
// den skips the indicator MatMul).
func TestLTCSynapsesFusedBackwardEquivalence(t *testing.T) {
	gradsOf := func(
		build func(pre, mu, sigma, wm, denReduce, numReduce *autograd.Variable) (num, den *autograd.Variable),
		preT, muT, sigT, wmT *tensor.Tensor,
		denReduce, numReduce *autograd.Variable,
	) []*tensor.Tensor {
		pre := autograd.Var(preT.Clone())
		mu := autograd.Var(muT.Clone())
		sigma := autograd.Var(sigT.Clone())
		wm := autograd.Var(wmT.Clone())
		num, den := build(pre, mu, sigma, wm, denReduce, numReduce)
		autograd.SumAll(autograd.Add(num, den)).Backward()
		return []*tensor.Tensor{pre.Grad, mu.Grad, sigma.Grad, wm.Grad}
	}
	bitwise := func(name string, got, want []*tensor.Tensor) {
		t.Helper()
		labels := []string{"pre", "mu", "sigma", "wm"}
		for i := range got {
			if got[i] == nil || want[i] == nil {
				t.Fatalf("%s: missing %s gradient (fused %v, legacy %v)", name, labels[i], got[i], want[i])
			}
			if !tensor.SameShape(got[i], want[i]) {
				t.Fatalf("%s: %s gradient shape drift: fused %v vs legacy %v",
					name, labels[i], got[i].Shape, want[i].Shape)
			}
			for k := range got[i].Data {
				if math.Float32bits(got[i].Data[k]) != math.Float32bits(want[i].Data[k]) {
					t.Fatalf("%s: %s gradient elem %d: fused %v (bits %#x) vs legacy %v (bits %#x)",
						name, labels[i], k, got[i].Data[k], math.Float32bits(got[i].Data[k]),
						want[i].Data[k], math.Float32bits(want[i].Data[k]))
				}
			}
		}
	}

	rng := rand.New(rand.NewSource(107))
	const inDim, units = 3, 5
	cell := NewLTC(inDim, units, RandomSparse(inDim, units, 0.6, 0.6, rng), 2, rng)

	// Recurrent path: units presynaptic sources through the indicator MatMuls.
	preR := tensor.Uniform(rng, -1, 1, 4, units)
	muR := tensor.Uniform(rng, 0.3, 0.8, units, units)
	sigR := tensor.Uniform(rng, 3, 8, units, units)
	wmR := tensor.Uniform(rng, 0.001, 1, units, units)
	fused := gradsOf(cell.synapses, preR, muR, sigR, wmR, cell.denReduceR, cell.numReduceR)
	legacy := gradsOf(legacySynapses, preR, muR, sigR, wmR, cell.denReduceR, cell.numReduceR)
	bitwise("recurrent", fused, legacy)

	// Sensory path with a single presynaptic source: den is the raw block
	// (the identity-contraction shortcut), num still goes through the MatMul.
	rng1 := rand.New(rand.NewSource(109))
	cell1 := NewLTC(1, units, RandomSparse(1, units, 0.6, 0.6, rng1), 2, rng1)
	preS := tensor.Uniform(rng1, -1, 1, 4, 1)
	muS := tensor.Uniform(rng1, 0.3, 0.8, 1, units)
	sigS := tensor.Uniform(rng1, 3, 8, 1, units)
	wmS := tensor.Uniform(rng1, 0.001, 1, 1, units)
	fused1 := gradsOf(cell1.synapses, preS, muS, sigS, wmS, cell1.denReduceS, cell1.numReduceS)
	legacy1 := gradsOf(legacySynapses, preS, muS, sigS, wmS, cell1.denReduceS, cell1.numReduceS)
	bitwise("single-source sensory", fused1, legacy1)
}
