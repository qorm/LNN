package autograd

import (
	"testing"

	"lnn/tensor"
)

// These tests probe the ownership-transfer regime of addGrad (clone-free
// first contribution) and Backward (private seed propagation). The core risk
// they guard against is aliasing: two leaf gradients, or a leaf gradient and
// a still-readable seed, sharing one buffer so that later accumulation
// silently corrupts earlier results. Every test checks exact values AND
// buffer independence (mutating one gradient must not move another).

func assertGrad(t *testing.T, name string, v *Variable, want []float32) {
	t.Helper()
	if v.Grad == nil {
		t.Fatalf("%s: gradient is nil", name)
	}
	if len(v.Grad.Data) != len(want) {
		t.Fatalf("%s: grad len %d, want %d (%v)", name, len(v.Grad.Data), len(want), v.Grad.Data)
	}
	for i, w := range want {
		if v.Grad.Data[i] != w {
			t.Fatalf("%s: grad[%d] = %v, want %v (full %v)", name, i, v.Grad.Data[i], w, v.Grad.Data)
		}
	}
}

func assertIndependent(t *testing.T, name string, p, q *Variable) {
	t.Helper()
	if p.Grad == nil || q.Grad == nil {
		t.Fatalf("%s: nil gradient", name)
	}
	if len(p.Grad.Data) == 0 || len(q.Grad.Data) == 0 {
		return
	}
	savedP, savedQ := p.Grad.Data[0], q.Grad.Data[0]
	p.Grad.Data[0] = savedP + 12345
	if q.Grad.Data[0] != savedQ {
		t.Fatalf("%s: gradients alias one buffer: mutating p moved q (%v -> %v)", name, savedQ, q.Grad.Data[0])
	}
	p.Grad.Data[0] = savedP
	q.Grad.Data[0] = savedQ + 54321
	if p.Grad.Data[0] != savedP {
		t.Fatalf("%s: gradients alias one buffer: mutating q moved p (%v -> %v)", name, savedP, p.Grad.Data[0])
	}
	q.Grad.Data[0] = savedQ
}

// TestAliasSharedIntermediateTwoLeaves: a shared Add result feeds two
// downstream branches. The a-branch of Add's backward hands its gradient
// buffer to x without cloning; y must still receive its own buffer.
func TestAliasSharedIntermediateTwoLeaves(t *testing.T) {
	x := New([]float32{1, 1, 1, 1}, 2, 2)
	y := New([]float32{2, 2, 2, 2}, 2, 2)
	w1 := New([]float32{1, 2, 3, 4}, 2, 2)
	w2 := New([]float32{5, 6, 7, 8}, 2, 2)

	s := Add(x, y)
	loss := SumAll(Add(Hadamard(s, w1), Hadamard(s, w2)))
	loss.Backward()

	// dL/ds = w1 + w2; both x and y receive it unchanged (Add gradient).
	want := []float32{6, 8, 10, 12}
	assertGrad(t, "x", x, want)
	assertGrad(t, "y", y, want)
	assertIndependent(t, "x/y", x, y)
}

// TestAliasBothLeavesReusedDownstream: x feeds the graph twice, so its
// gradient accumulates from two distinct backward paths; y once. This is the
// exact corruption scenario the cloning b-branch in Add's backward prevents.
func TestAliasBothLeavesReusedDownstream(t *testing.T) {
	x := New([]float32{1, 2, 3, 4}, 2, 2)
	y := New([]float32{5, 6, 7, 8}, 2, 2)

	z1 := Add(x, y)
	z2 := Add(z1, x) // x used again
	loss := SumAll(z2)
	loss.Backward()

	assertGrad(t, "x", x, []float32{2, 2, 2, 2}) // two paths of ones
	assertGrad(t, "y", y, []float32{1, 1, 1, 1})
	assertIndependent(t, "x/y", x, y)
}

// TestAliasHadamardSharedOperand: Hadamard(x, x) doubled and summed. Both
// product branches see the same operand variable; the two per-branch buffers
// must be distinct so the accumulation into x is g*x + g*x, not a self-add
// of one shared buffer twice over (which happens to be arithmetically equal
// here — the independence assertion is what catches aliasing regressions in
// the general fan-out case).
func TestAliasHadamardSharedOperand(t *testing.T) {
	x := New([]float32{1, 2, 3, 4}, 2, 2)
	y := Hadamard(x, x)
	loss := SumAll(Add(y, y))
	loss.Backward()

	assertGrad(t, "x", x, []float32{4, 8, 12, 16}) // d(2x^2)/dx = 4x
}

// TestAliasBroadcastReductionIndependence: a row-vector leaf and a full
// matrix share an Add; the reduced (SumRows) gradient of the row vector must
// be its own buffer, independent of the matrix leaf's handed-through grad.
func TestAliasBroadcastReductionIndependence(t *testing.T) {
	x := New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	b := New([]float32{10, 20, 30}, 3)

	s := Add(x, b)
	loss := SumAll(Hadamard(s, s)) // dL/ds = 2s
	loss.Backward()

	assertGrad(t, "x", x, []float32{22, 44, 66, 28, 50, 72})
	assertGrad(t, "b", b, []float32{50, 94, 138}) // column sums of 2s
	assertIndependent(t, "x/b", x, b)
}

// TestAliasMatMulGradIndependence: the two MatMul backward products must be
// distinct buffers even though they derive from the same out.Grad.
func TestAliasMatMulGradIndependence(t *testing.T) {
	a := New([]float32{1, 2, 3, 4}, 2, 2)
	b := New([]float32{5, 6, 7, 8}, 2, 2)

	loss := SumAll(MatMul(a, b))
	loss.Backward()

	assertGrad(t, "a", a, []float32{11, 15, 11, 15}) // ones · b^T
	assertGrad(t, "b", b, []float32{4, 4, 6, 6})     // a^T · ones
	assertIndependent(t, "a/b", a, b)
}

// TestAliasRepeatedBackwardStaysLinear: three Backward calls on the same
// graph must accumulate exactly linearly (1x, 2x, 3x) — the property the
// Backward seed hand-off/re-seed protocol protects.
func TestAliasRepeatedBackwardStaysLinear(t *testing.T) {
	x := New([]float32{1, 2, 3}, 3)
	s := Scale(x, 2)
	loss := SumAll(Add(s, s)) // dL/dx = 4

	for round, want := range []float32{4, 8, 12} {
		loss.Backward()
		for i := range x.Grad.Data {
			if x.Grad.Data[i] != want {
				t.Fatalf("round %d: x.Grad[%d] = %v, want %v (full %v)",
					round, i, x.Grad.Data[i], want, x.Grad.Data)
			}
		}
	}
	// The root's gradient must remain a pristine all-ones seed, and must not
	// alias the leaf gradient (otherwise the next round would re-propagate a
	// corrupted seed and break linearity).
	assertGrad(t, "loss.Grad", loss, []float32{1})
	assertIndependent(t, "loss/x", loss, x)
}

// TestAliasSeededGradStaysPristine: a manually seeded intermediate gradient
// must survive Backward pointer-identical and unmutated, exactly as when
// addGrad cloned every incoming contribution.
func TestAliasSeededGradStaysPristine(t *testing.T) {
	x := New([]float32{1, 2, 3, 4}, 2, 2)
	y := Hadamard(x, x)
	seed := tensor.FromData([]float32{1, 10, 100, 1000}, 2, 2)
	y.Grad = seed

	y.Backward()

	assertGrad(t, "x", x, []float32{2, 40, 600, 8000}) // 2x ⊙ seed
	if y.Grad != seed {
		t.Fatalf("seeded Grad pointer changed: Backward must restore the caller's buffer")
	}
	assertGrad(t, "seed", y, []float32{1, 10, 100, 1000})
	assertIndependent(t, "y/x", y, x)
}

// TestAliasNonScalarAutoSeedPristine covers the seeded entry path where the
// root's buffer is non-scalar but produced by a previous auto-seed: the
// observable post-Backward state is still all-ones at the root.
func TestAliasNonScalarAutoSeedPristine(t *testing.T) {
	x := New([]float32{1, 2, 3, 4}, 2, 2)
	y := Hadamard(x, x)
	y.Grad = tensor.FromData([]float32{1, 1, 1, 1}, 2, 2) // simulate prior auto-seed

	y.Backward()
	assertGrad(t, "x", x, []float32{2, 4, 6, 8})
	assertGrad(t, "y.Grad", y, []float32{1, 1, 1, 1})
}
