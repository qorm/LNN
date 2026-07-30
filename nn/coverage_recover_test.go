package nn

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/tensor"
)

// Coverage-recovery tests for the stage-8 fused-operator era: the CfC
// single-presynaptic-source shortcut baked into drive(), and the rank/value
// validation panic contracts of the wiring and cell constructors. Every test
// asserts values, bitwise equality or exact panic messages.

// recoveredMsg runs f and returns the recovered panic message, or "" if f
// did not panic.
func recoveredMsg(f func()) string {
	var msg string
	func() {
		defer func() {
			if r := recover(); r != nil {
				msg = fmt.Sprint(r)
			}
		}()
		f()
	}()
	return msg
}

// TestRecoverCfCSingleSourceShortcutBitExact covers drive()'s n == 1 leg:
// with a single presynaptic source denReduce is the identity, so the
// contraction is the block itself (den = flat) instead of a MatMul. A
// inDim=1 / units=1 cell drives BOTH the sensory and the recurrent
// contraction through the shortcut; each must stay bit-identical to the
// legacy Add-of-Hadamards chain (the same oracle the n>1 regression uses),
// and the full Step/Backward path must run with correctly shaped gradients.
func TestRecoverCfCSingleSourceShortcutBitExact(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	cell := NewCfC(1, 1, nil, rng)
	x := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 1))
	h := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 1))

	inputs := autograd.Add(autograd.Hadamard(x, cell.inW), cell.inB)
	sWPos := autograd.Softplus(cell.sW)
	wPos := autograd.Softplus(cell.w)

	// n = 1 on both drives: sensory (pre cols = inDim) and recurrent
	// (pre cols = units) both take the den = flat shortcut.
	numS, denS := cell.drive(inputs, cell.sMu, cell.sSigma, sWPos, cell.denReduceS, cell.numReduceS, cell.wiring.SensoryRow)
	numR, denR := cell.drive(h, cell.mu, cell.sigma, wPos, cell.denReduceR, cell.numReduceR, cell.wiring.RecurrentRow)
	lNumS, lDenS := legacyCfCDrive(inputs, cell.sMu, cell.sSigma, sWPos, autograd.Var(cell.sErev), cell.wiring.SensoryRow)
	lNumR, lDenR := legacyCfCDrive(h, cell.mu, cell.sigma, wPos, autograd.Var(cell.erev), cell.wiring.RecurrentRow)

	for name, pair := range map[string][2]*autograd.Variable{
		"numS": {numS, lNumS}, "denS": {denS, lDenS},
		"numR": {numR, lNumR}, "denR": {denR, lDenR},
	} {
		if !sameBitsT(pair[0].Data, pair[1].Data) {
			t.Fatalf("%s: n=1 shortcut contraction differs from the legacy Add chain: %v vs %v",
				name, pair[0].Data.Data, pair[1].Data.Data)
		}
	}

	// End-to-end through the public Step: shapes, finiteness, and a
	// backward pass that reaches the input and every parameter.
	out, hNew := cell.Step(x, nil, 0.1)
	if out.Data.Rows() != 2 || out.Data.Cols() != 1 || hNew.Data.Rows() != 2 || hNew.Data.Cols() != 1 {
		t.Fatalf("Step shapes out %v h %v, want [2 1] / [2 1]", out.Data.Shape, hNew.Data.Shape)
	}
	assertFinite(t, "n=1 CfC step", out, hNew)
	autograd.MeanAll(out).Backward()
	if x.Grad == nil || len(x.Grad.Shape) != 2 || x.Grad.Shape[0] != 2 || x.Grad.Shape[1] != 1 {
		t.Fatalf("input gradient shape %v, want [2 1]", x.Grad.Shape)
	}
	for _, p := range cell.Parameters() {
		if p.Grad == nil {
			t.Fatal("n=1 CfC step: parameter with nil gradient")
		}
	}
}

// TestRecoverWiringBinaryMaskContract pins assertBinaryMask: binary masks
// pass and round-trip verbatim through the accessors, while any entry
// outside {0, 1} panics with the entry value in the message — for the
// sensory and the recurrent mask alike.
func TestRecoverWiringBinaryMaskContract(t *testing.T) {
	okS := tensor.FromData([]float32{1, 0, 0, 1}, 2, 2)
	okR := tensor.FromData([]float32{0, 1, 1, 0}, 2, 2)
	w := newWiring(okS, okR)
	if !sameBitsT(w.Sensory(), okS) || !sameBitsT(w.Recurrent(), okR) {
		t.Fatal("newWiring must accept binary masks and expose them verbatim")
	}
	if msg := recoveredMsg(func() {
		newWiring(tensor.FromData([]float32{1, 0.5}, 1, 2), okR)
	}); msg != "nn: sensory mask entry 0.5 outside {0, 1}" {
		t.Fatalf("sensory panic %q", msg)
	}
	if msg := recoveredMsg(func() {
		newWiring(okS, tensor.FromData([]float32{0, 1, 2, 0}, 2, 2))
	}); msg != "nn: recurrent mask entry 2 outside {0, 1}" {
		t.Fatalf("recurrent panic %q", msg)
	}
}

// TestRecoverCellWiringRankPanicContracts drives the mask-shape validation
// of NewLTC/NewCfC through a rank mismatch (a 1D mask against the 2D want),
// which takes the length-comparison arm of shapeIs/cfcShapeEq rather than
// the per-element arm the wrong-dims regressions already exercise.
func TestRecoverCellWiringRankPanicContracts(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	if msg := recoveredMsg(func() {
		NewLTC(2, 3, &Wiring{sensoryMask: tensor.New(6), recurrentMask: tensor.New(3, 3)}, 1, rng)
	}); !strings.Contains(msg, "nn.NewLTC: wiring mask shapes [6] and [3 3] do not match [inDim=2, units=3]") {
		t.Fatalf("NewLTC rank-mismatch panic %q", msg)
	}
	if msg := recoveredMsg(func() {
		NewCfC(2, 2, &Wiring{sensoryMask: tensor.New(2, 2), recurrentMask: tensor.New(4)}, rng)
	}); !strings.Contains(msg, "nn.NewCfC: wiring mask shapes [2 2] and [4] do not match [inDim=2, units=2]") {
		t.Fatalf("NewCfC rank-mismatch panic %q", msg)
	}
}
