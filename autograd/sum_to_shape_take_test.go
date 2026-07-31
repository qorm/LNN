package autograd

import (
	"reflect"
	"testing"

	"github.com/qorm/LNN/tensor"
)

// TestSumToShapeTakeInternal pins the autograd-internal owning reducer that
// moved here from the tensor package in v0.4.0 (formerly the exported
// tensor.SumToShapeTake). The numeric assertions are carried over unchanged
// from the tensor package's TestSumToShape / TestSumToShapeColVec; the
// additions are the ownership contract that justifies the function's
// existence — the matching-shape arm returns grad itself, not a clone — and
// the irreducible panic contract the tensor coverage test used to drive.
func TestSumToShapeTakeInternal(t *testing.T) {
	fresh := func() *tensor.Tensor {
		return tensor.FromRows([]float32{1, 2, 3}, []float32{4, 5, 6})
	}

	// Matching shape: ownership hand-off, grad returned as-is (same pointer).
	grad := fresh()
	same := sumToShapeTake(grad, []int{2, 3})
	if same != grad {
		t.Fatalf("matching-shape arm cloned: got a distinct buffer, want grad itself")
	}
	if !reflect.DeepEqual(same.Shape, []int{2, 3}) {
		t.Fatalf("matching-shape arm shape = %v, want [2 3]", same.Shape)
	}

	// Scalar target: total sum.
	scalar := sumToShapeTake(fresh(), []int{1})
	if scalar.Scalar() != 21 {
		t.Fatalf("scalar reduce = %v, want 21", scalar.Scalar())
	}

	// Row-vector target [3]: column sums, flattened.
	row := sumToShapeTake(fresh(), []int{3})
	if !reflect.DeepEqual(row.Shape, []int{3}) {
		t.Fatalf("row reduce shape = %v, want [3]", row.Shape)
	}
	if !bitsEqual(row, tensor.FromData([]float32{5, 7, 9}, 3)) {
		t.Fatalf("row reduce = %v, want [5 7 9]", row.Data)
	}

	// Row-vector target [1, 3]: column sums, kept 2D.
	row2 := sumToShapeTake(fresh(), []int{1, 3})
	if !reflect.DeepEqual(row2.Shape, []int{1, 3}) {
		t.Fatalf("row2 reduce shape = %v, want [1 3]", row2.Shape)
	}
	if !bitsEqual(row2, tensor.FromData([]float32{5, 7, 9}, 1, 3)) {
		t.Fatalf("row2 reduce = %v, want [5 7 9]", row2.Data)
	}

	// Column-vector target [2, 1]: row sums.
	col := sumToShapeTake(fresh(), []int{2, 1})
	if !reflect.DeepEqual(col.Shape, []int{2, 1}) {
		t.Fatalf("col reduce shape = %v, want [2 1]", col.Shape)
	}
	if !bitsEqual(col, tensor.FromData([]float32{6, 15}, 2, 1)) {
		t.Fatalf("col reduce = %v, want [6 15]", col.Data)
	}
}

// TestSumToShapeTakeIrreduciblePanic is the migrated panic contract: the
// irreducible default arm must surface the exact historical message so a
// refactor cannot silently reword it (formerly the tensor package's
// "SumToShapeTake irreducible" coverage case).
func TestSumToShapeTakeIrreduciblePanic(t *testing.T) {
	msg := recoverMsg(func() { sumToShapeTake(tensor.New(2, 2), []int{3}) })
	if msg != "tensor.SumToShape: cannot reduce shape [2 2] to [3]" {
		t.Fatalf("irreducible panic = %q, want the historical SumToShape message", msg)
	}
}
