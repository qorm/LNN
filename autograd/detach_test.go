package autograd

import (
	"testing"
)

// These tests pin the Detach contract: a detached leaf carries the source's
// value but none of its history — gradients never flow back into the
// source's graph (TestDetachNoBackflow*), the detached value is unaffected
// by everything the graph does afterwards (TestDetachValueStability), and
// the zero-copy storage alias behaves exactly as documented
// (TestDetachAliasesStorage).

// TestDetachNoBackflow: a loss built on the detached leaf differentiates
// the graph downstream of the cut only; the source's ancestors see nothing.
func TestDetachNoBackflow(t *testing.T) {
	x := New([]float32{1, 2, 3, 4}, 2, 2)
	y := Hadamard(x, x) // y = x^2
	d := Detach(y)
	z := Hadamard(d, d) // z = d^2, downstream of the cut only
	loss := SumAll(z)
	loss.Backward()

	if x.Grad != nil {
		t.Fatalf("gradient flowed back through Detach: x.Grad = %v, want nil", x.Grad.Data)
	}
	if y.Grad != nil {
		t.Fatalf("source node gained a gradient from the detached side: y.Grad = %v, want nil", y.Grad.Data)
	}
	// dz/dd = 2d elementwise, on the detached leaf only.
	assertGrad(t, "d", d, []float32{2, 8, 18, 32})
}

// TestDetachNoBackflowSourceSide: the cut does not disturb the source
// graph's own backward — a loss rooted above the detach point still
// differentiates through the source exactly as if Detach had never been
// called (the detached leaf is an independent handle, not a graph edit).
func TestDetachNoBackflowSourceSide(t *testing.T) {
	x := New([]float32{1, 2, 3, 4}, 2, 2)
	y := Hadamard(x, x)
	d := Detach(y)
	if d.Grad != nil {
		t.Fatalf("fresh Detach leaf has Grad %v, want nil", d.Grad.Data)
	}
	loss := SumAll(y) // rooted in the source graph, above the cut
	loss.Backward()
	// d(loss)/dx = 2x, exactly as without the Detach call.
	assertGrad(t, "x", x, []float32{2, 4, 6, 8})
	if d.Grad != nil {
		t.Fatalf("source-graph backward leaked into the detached leaf: d.Grad = %v, want nil", d.Grad.Data)
	}
}

// TestDetachValueStability: the detached leaf's value equals the source's
// value bit for bit at detach time and stays bit-stable while the source
// graph keeps growing and backpropagating — no library op mutates an
// existing tensor, so sharing the storage is safe.
func TestDetachValueStability(t *testing.T) {
	x := New([]float32{0.5, -1.25, 3, -0.75}, 2, 2)
	y := Tanh(Hadamard(x, x))
	d := Detach(y)
	snapshot := d.Data.Clone()

	// Keep building on the source and run a backward through it.
	z := SumAll(Exp(y))
	z.Backward()

	if !bitsEqual(d.Data, snapshot) {
		t.Fatalf("detached value moved: got %v, want %v", d.Data.Data, snapshot.Data)
	}
	if !bitsEqual(d.Data, y.Data) {
		t.Fatalf("detached value diverged from source: got %v, source %v", d.Data.Data, y.Data.Data)
	}
}

// TestDetachAliasesStorage pins the documented zero-copy contract: the
// detached leaf shares the source's tensor storage (no clone), so a
// caller-side write into that storage — the one mutation the library never
// performs itself — is visible through both, exactly as with Var.
func TestDetachAliasesStorage(t *testing.T) {
	x := New([]float32{1, 2, 3, 4}, 2, 2)
	y := Hadamard(x, x)
	d := Detach(y)
	if d.Data != y.Data {
		t.Fatalf("Detach copied the tensor: got a distinct *tensor.Tensor")
	}
	y.Data.Data[0] = 42
	if d.Data.Data[0] != 42 {
		t.Fatalf("detached leaf does not share storage: got %v, want 42", d.Data.Data[0])
	}
	y.Data.Data[0] = 1 // restore, keeping the probe local
}

// TestDetachOfLeaf: detaching a leaf yields an independent leaf with the
// same value; gradients into the detached copy do not touch the original.
func TestDetachOfLeaf(t *testing.T) {
	p := New([]float32{1, -2}, 1, 2)
	d := Detach(p)
	loss := SumAll(Hadamard(d, d))
	loss.Backward()
	assertGrad(t, "d", d, []float32{2, -4})
	if p.Grad != nil {
		t.Fatalf("gradient flowed from detached copy into the original leaf: %v", p.Grad.Data)
	}
	// Linear reruns accumulate on the detached leaf, as on any leaf.
	loss.Backward()
	assertGrad(t, "d rerun", d, []float32{4, -8})
}
