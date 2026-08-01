package autograd

import (
	"testing"

	"github.com/qorm/LNN/tensor"
)

// TestTopoOrder pins the DFS post-order contract: parents before children,
// parents in construction order, shared subtrees visited once, and the
// whole walk deterministic across calls.
func TestTopoOrder(t *testing.T) {
	x := New([]float32{1, 2}, 1, 2)
	y := New([]float32{3, 4}, 1, 2)
	h := Hadamard(x, y) // [x y]
	a := Add(h, x)      // [h x]
	loss := SumAll(a)   // [a]

	topo := TopoOrder(loss)
	// build(loss): -> build(a): -> build(h): -> x, y, h; -> x visited; a; loss.
	want := []*Variable{x, y, h, a, loss}
	if len(topo) != len(want) {
		t.Fatalf("topo length %d, want %d (%v)", len(topo), len(want), topo)
	}
	for i, v := range want {
		if topo[i] != v {
			t.Fatalf("topo[%d] is not the expected node (position %d of %v)", i, i, want)
		}
	}

	// Parents always precede children, over a wider shared-subtree graph.
	z := Hadamard(a, h) // a and h again: shared
	pos := map[*Variable]int{}
	for i, v := range TopoOrder(z) {
		pos[v] = i
	}
	if len(pos) != 5 { // x, y, h, a, z
		t.Fatalf("shared subtree visited more than once: %d nodes, want 5", len(pos))
	}
	for _, v := range []*Variable{h, a, z} {
		for _, p := range v.parentsSlice() {
			if pos[p] >= pos[v] {
				t.Fatalf("parent appears after its child in the post-order")
			}
		}
	}

	// Determinism: same graph, same order, twice.
	t1 := TopoOrder(loss)
	t2 := TopoOrder(loss)
	if len(t1) != len(t2) {
		t.Fatalf("non-deterministic length: %d vs %d", len(t1), len(t2))
	}
	for i := range t1 {
		if t1[i] != t2[i] {
			t.Fatalf("non-deterministic order at %d", i)
		}
	}

	// A leaf's order is itself.
	leaf := Var(tensor.New(2, 2))
	if got := TopoOrder(leaf); len(got) != 1 || got[0] != leaf {
		t.Fatalf("leaf topo = %v, want just the leaf", got)
	}
}

// TestParents pins the read-only parent accessor: leaves have none, op
// nodes return them in construction order, and the result matches what
// TopoOrder traverses.
func TestParents(t *testing.T) {
	x := New([]float32{1, 2}, 1, 2)
	y := New([]float32{3, 4}, 1, 2)
	if got := x.Parents(); len(got) != 0 {
		t.Fatalf("leaf parents = %v, want none", got)
	}
	h := Hadamard(x, y)
	if got := h.Parents(); len(got) != 2 || got[0] != x || got[1] != y {
		t.Fatalf("Hadamard parents = %v, want [x y]", got)
	}
	a := Add(h, x)
	if got := a.Parents(); len(got) != 2 || got[0] != h || got[1] != x {
		t.Fatalf("Add parents = %v, want [h x]", got)
	}
	c := ConcatCol(x, y, a) // the overflow (>2 parents) path
	if got := c.Parents(); len(got) != 3 || got[0] != x || got[1] != y || got[2] != a {
		t.Fatalf("ConcatCol parents = %v, want [x y a]", got)
	}
}
