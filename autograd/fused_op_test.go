package autograd

import (
	"strings"
	"testing"

	"github.com/qorm/LNN/tensor"
)

// TestFusedOpContract pins the custom-op hook: the node carries the
// caller's forward value and parents, runBackward dispatches the recorded
// closure (and nothing else), and a nil backward panics at construction.
func TestFusedOpContract(t *testing.T) {
	a := Var(tensor.FromData([]float32{1, 2}, 1, 2))
	b := Var(tensor.FromData([]float32{3, 4}, 1, 2))
	var called *Variable
	out := FusedOp(tensor.FromData([]float32{5, 6}, 1, 2), []*Variable{a, b}, func(v *Variable) {
		called = v
		a.addGrad(tensor.New(1, 2).OnesLike())
	})
	if out.Data.Data[0] != 5 || len(out.Parents()) != 2 || out.Parents()[0] != a || out.Parents()[1] != b {
		t.Fatalf("FusedOp wiring: data %v parents %v", out.Data, out.Parents())
	}
	SumAll(out).Backward()
	if called != out {
		t.Fatalf("backward closure did not run on the node")
	}
	if a.Grad == nil || a.Grad.Data[0] != 1 || b.Grad != nil {
		t.Fatalf("closure-delivered gradients: a %v b %v", a.Grad, b.Grad)
	}

	defer func() {
		r := recover()
		if r == nil || !strings.Contains(r.(string), "nil backward") {
			t.Fatalf("nil backward must panic, got %v", r)
		}
	}()
	FusedOp(tensor.New(1), nil, nil)
}
