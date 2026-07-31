package nn

import (
	"math"
	"math/rand"
	"testing"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/tensor"
)

// fusedLegacySumIndicator and fusedLegacyReversalIndicator rebuild the dense
// [n*m, m] reduction matrices the synapse hot path used to contract against
// (the O(units^3) indicators item #14 removed): R[i*m+j, j] = 1, resp.
// erev[i*m+j]. They live here — not in ltc.go — because only this legacy
// oracle still materializes them, at test scale (units <= 5).
func fusedLegacySumIndicator(n, m int) *tensor.Tensor {
	r := tensor.New(n*m, m)
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			r.Data[(i*m+j)*m+j] = 1
		}
	}
	return r
}

func fusedLegacyReversalIndicator(erev []float32, n, m int) *tensor.Tensor {
	r := tensor.New(n*m, m)
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			r.Data[(i*m+j)*m+j] = erev[i*m+j]
		}
	}
	return r
}

// legacySynapses reproduces synapses/synapsesRows exactly as they ran before
// the Sigmoid-Hadamard fusion AND the item-#14 sparse contraction: the
// identical per-presynaptic structure (SliceRow/Col/Sub/Hadamard, ConcatCol,
// dense indicator MatMul reductions), with the activation block built from
// the separate Sigmoid and Hadamard nodes instead of the fused
// autograd.SigmoidHadamard node. Differential tests below pin the current
// path (fused node + sparse fold contraction) to this oracle.
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

// TestLTCSynapsesFusedBackwardEquivalence checks that the current synapse
// hot path — the fused SigmoidHadamard node on top of the item-#14 sparse
// fold contraction — leaves the gradients of every participating leaf
// bit-identical to the former Sigmoid+Hadamard pair running through the
// dense indicator MatMuls. The fused backward rounds each product at the
// same spots the two-node chain did, and the sparse fold replicates the
// indicator MatMul's accumulation order and zero-skip (see contract), so
// strict Float32bits equality — not tolerance — is the bar, on both the
// multi-source recurrent path and the single-source sensory path (whose den
// skips the contraction entirely).
func TestLTCSynapsesFusedBackwardEquivalence(t *testing.T) {
	gradsOf := func(
		build func(pre, mu, sigma, wm *autograd.Variable) (num, den *autograd.Variable),
		preT, muT, sigT, wmT *tensor.Tensor,
	) []*tensor.Tensor {
		pre := autograd.Var(preT.Clone())
		mu := autograd.Var(muT.Clone())
		sigma := autograd.Var(sigT.Clone())
		wm := autograd.Var(wmT.Clone())
		num, den := build(pre, mu, sigma, wm)
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

	// Recurrent path: units presynaptic sources through the contraction.
	preR := tensor.Uniform(rng, -1, 1, 4, units)
	muR := tensor.Uniform(rng, 0.3, 0.8, units, units)
	sigR := tensor.Uniform(rng, 3, 8, units, units)
	wmR := tensor.Uniform(rng, 0.001, 1, units, units)
	denR := autograd.Const(fusedLegacySumIndicator(units, units))
	numR := autograd.Const(fusedLegacyReversalIndicator(cell.erev.Data.Data, units, units))
	fused := gradsOf(func(pre, mu, sigma, wm *autograd.Variable) (*autograd.Variable, *autograd.Variable) {
		return cell.synapses(pre, mu, sigma, wm, cell.erevRowsR)
	}, preR, muR, sigR, wmR)
	legacy := gradsOf(func(pre, mu, sigma, wm *autograd.Variable) (*autograd.Variable, *autograd.Variable) {
		return legacySynapses(pre, mu, sigma, wm, denR, numR)
	}, preR, muR, sigR, wmR)
	bitwise("recurrent", fused, legacy)

	// Sensory path with a single presynaptic source: den is the raw block
	// (the identity-contraction shortcut), num still contracts.
	rng1 := rand.New(rand.NewSource(109))
	cell1 := NewLTC(1, units, RandomSparse(1, units, 0.6, 0.6, rng1), 2, rng1)
	preS := tensor.Uniform(rng1, -1, 1, 4, 1)
	muS := tensor.Uniform(rng1, 0.3, 0.8, 1, units)
	sigS := tensor.Uniform(rng1, 3, 8, 1, units)
	wmS := tensor.Uniform(rng1, 0.001, 1, 1, units)
	denS := autograd.Const(fusedLegacySumIndicator(1, units))
	numS := autograd.Const(fusedLegacyReversalIndicator(cell1.sErev.Data.Data, 1, units))
	fused1 := gradsOf(func(pre, mu, sigma, wm *autograd.Variable) (*autograd.Variable, *autograd.Variable) {
		return cell1.synapses(pre, mu, sigma, wm, cell1.erevRowsS)
	}, preS, muS, sigS, wmS)
	legacy1 := gradsOf(func(pre, mu, sigma, wm *autograd.Variable) (*autograd.Variable, *autograd.Variable) {
		return legacySynapses(pre, mu, sigma, wm, denS, numS)
	}, preS, muS, sigS, wmS)
	bitwise("single-source sensory", fused1, legacy1)
}
