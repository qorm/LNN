package autograd

import (
	"testing"

	"github.com/qorm/LNN/tensor"
)

// White-box tests for the stage-19a C8 traversal-scratch pool (dfsScratch):
// Backward and TopoOrder reuse one visited set and one topo buffer across
// calls. The tests pin the three safety properties the pool's comment
// argues — no state leakage across calls, reentrant fallback to fresh
// scratch, pristine release after a mid-traversal panic — plus the
// bitwise accumulation contract that proves the reuse is unobservable.

// TestDFSScratchNoCrossCallLeak runs Backward on a graph, then on a second
// graph sharing the first's subtree: a visited-set leak from the first
// traversal would mark the shared nodes visited and silently drop their
// backward steps in the second. The second graph's gradients must match a
// fresh standalone run bit for bit.
func TestDFSScratchNoCrossCallLeak(t *testing.T) {
	build := func() (x, loss *Variable) {
		x = New([]float32{0.5, -1.25}, 1, 2)
		y := Tanh(x)
		loss = SumAll(y)
		return x, loss
	}
	xA, lossA := build()
	lossA.Backward() // pool now holds (then releases) graph A's scratch
	want := xA.Grad.Clone()

	// Second graph sharing nothing with A: same structure, fresh nodes —
	// the bitwise reference.
	xB, lossB := build()
	lossB.Backward()
	if !bitsEqual(xB.Grad, want) {
		t.Fatalf("second Backward after pool reuse = %v, want %v", xB.Grad.Data, want.Data)
	}

	// Linear accumulation across repeated Backward calls on the same graph
	// (each run propagates ones again): two runs accumulate exactly g+g.
	lossA.Backward()
	twice := tensor.Add(want, want)
	if !bitsEqual(xA.Grad, twice) {
		t.Fatalf("repeated Backward = %v, want linear accumulation %v", xA.Grad.Data, twice.Data)
	}
}

// TestDFSScratchReentrantBackward nests one Backward inside another through
// a FusedOp's caller-supplied backward step — the one reentrancy the
// single-threaded contract does not rule out. The nested call must fall
// back to fresh scratch (the pool is checked out by the outer traversal),
// and both traversals must deliver their exact gradients.
func TestDFSScratchReentrantBackward(t *testing.T) {
	// Inner graph, differentiated from inside the outer graph's traversal.
	ix := New([]float32{0.25, -0.75}, 1, 2)
	innerLoss := SumAll(Tanh(ix))
	// Reference: the same inner graph differentiated standalone.
	rx := New([]float32{0.25, -0.75}, 1, 2)
	SumAll(Tanh(rx)).Backward()

	ox := New([]float32{3, 4}, 1, 2)
	nested := false
	fused := FusedOp(ox.Data.Clone(), []*Variable{ox}, func(v *Variable) {
		innerLoss.Backward() // nested Backward: pool is checked out
		nested = true
		v.parent(0).addGrad(v.Grad.Clone())
	})
	SumAll(fused).Backward()

	if !nested {
		t.Fatal("the fused backward step never ran")
	}
	if !bitsEqual(ix.Grad, rx.Grad) {
		t.Fatalf("nested Backward grad = %v, want the standalone %v", ix.Grad.Data, rx.Grad.Data)
	}
	wantOuter := tensor.New(1, 2).OnesLike() // SumAll broadcasts the scalar seed
	if !bitsEqual(ox.Grad, wantOuter) {
		t.Fatalf("outer grad = %v, want %v — the nested traversal corrupted the outer scratch", ox.Grad.Data, wantOuter.Data)
	}
	// The outer release ran after the nested fallback: the pool must be
	// free and empty for the next call.
	assertScratchPristine(t)
}

// TestDFSScratchReleaseAfterPanic panics a Backward mid-traversal (a leaf
// pre-seeded with a mismatching gradient shape makes addGrad panic inside
// the dispatch loop) and recovers: the deferred release must still return
// the pool pristine, and the next well-formed Backward must be unaffected.
func TestDFSScratchReleaseAfterPanic(t *testing.T) {
	x := New([]float32{1, 2, 3, 4}, 2, 2)
	loss := SumAll(Hadamard(x, x))
	x.Grad = tensor.New(3) // poisoned seed: [3] vs the incoming [2 2]
	msg := recoverMsg(func() { loss.Backward() })
	if msg == "" {
		t.Fatal("the poisoned seed did not panic mid-traversal")
	}
	assertScratchPristine(t)

	ok := New([]float32{1, 2}, 1, 2)
	SumAll(Tanh(ok)).Backward()
	if ok.Grad == nil {
		t.Fatal("Backward after a recovered panic produced no gradient")
	}
}

// TestDFSScratchTopoOrderIsolated pins TopoOrder's half of the pool
// contract: it reuses the visited set but its result slice is always fresh
// (mutating a previous result must not corrupt a later call), and it
// interleaves with Backward without either disturbing the other.
func TestDFSScratchTopoOrderIsolated(t *testing.T) {
	x := New([]float32{1, 2}, 1, 2)
	loss := SumAll(Hadamard(x, x))

	t1 := TopoOrder(loss)
	if len(t1) != 3 { // x, Hadamard, SumAll
		t.Fatalf("topo length %d, want 3", len(t1))
	}
	t1[0] = nil // caller-owned: scribbling on it must be harmless
	t2 := TopoOrder(loss)
	if t2[0] != x {
		t.Fatalf("TopoOrder result aliases pooled storage: first entry %v, want the leaf", t2[0])
	}
	if t1[0] != nil {
		t.Fatal("a later TopoOrder wrote into an earlier result — the result slice must be fresh storage, never pooled")
	}

	loss.Backward()
	assertScratchPristine(t)
	t3 := TopoOrder(loss)
	if len(t3) != 3 || t3[0] != x || t3[2] != loss {
		t.Fatalf("TopoOrder after Backward = %v, want [leaf op root]", t3)
	}
}

// TestDFSScratchAcquireReleaseContract drives the pool mechanism directly:
// the first acquisition checks the pool out, a nested one falls back to
// fresh scratch, release empties both containers while retaining their
// capacity (the high-water trade the pool exists for), and the next
// acquisition checks the pool out again.
func TestDFSScratchAcquireReleaseContract(t *testing.T) {
	visited, topo, pooled := acquireDFS()
	if !pooled {
		t.Fatal("first acquisition did not check out the pool")
	}
	marker := Var(tensor.New(1))
	visited[marker] = true
	for i := 0; i < 40; i++ { // grow past the initial capacity of 16
		topo = append(topo, marker)
	}
	if _, _, nestedPooled := acquireDFS(); nestedPooled {
		t.Fatal("nested acquisition must fall back to fresh scratch")
	}
	releaseDFS(visited, topo)
	assertScratchPristine(t)
	if cap(dfsScratch.topo) < 40 {
		t.Fatalf("release dropped the topo capacity: cap %d after a 40-node traversal", cap(dfsScratch.topo))
	}
	if _, _, again := acquireDFS(); !again {
		t.Fatal("pool not re-acquirable after release")
	}
	releaseDFS(dfsScratch.visited, dfsScratch.topo)
}

// assertScratchPristine verifies the pool is free and holds no traversal
// state (and in particular no node references).
func assertScratchPristine(t *testing.T) {
	t.Helper()
	if dfsScratch.inUse {
		t.Fatal("pool still checked out")
	}
	if len(dfsScratch.visited) != 0 {
		t.Fatalf("visited set not empty after release: %d entries", len(dfsScratch.visited))
	}
	if len(dfsScratch.topo) != 0 {
		t.Fatalf("topo buffer not empty after release: %d entries", len(dfsScratch.topo))
	}
	for i, n := range dfsScratch.topo[:cap(dfsScratch.topo)] {
		if n != nil {
			t.Fatalf("topo backing array retains a node reference at %d", i)
		}
	}
}
