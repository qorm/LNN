// Native Go fuzzing for tensor's constructors (New / FromData / FromRows).
// The tensor package follows the library's panic-on-misuse contract
// (its inputs come from the program, not an external stream), so the oracle is
// NOT "never panics" but "never panics BADLY": every constructor either
// succeeds — in which case Size()==len(Data) and every index is reachable
// through At/Set without going out of bounds — or panics with a descriptive,
// semantic message. A bare runtime crash ("makeslice: len out of range",
// "index out of range") is a failure: the overflow checks in Size must turn
// every gigantic shape into a worded panic BEFORE any allocation happens.
//
// External test package (tensor_test), driving only the exported API.
package tensor_test

import (
	"fmt"
	"math"
	"math/bits"
	"runtime"
	"strings"
	"testing"

	"github.com/qorm/LNN/tensor"
)

// fuzzElemCap bounds the element count the fuzz harness actually allocates,
// so a fuzzed shape like [1<<20, 1<<20] is classified (overflow) without the
// harness itself trying to make a terabyte tensor. Shapes whose product is
// finite but above this cap are skipped for allocation, but their overflow
// classification is still exercised when they DO overflow int64.
const fuzzElemCap = 1 << 16

// fuzzTry runs fn and reports whether it panicked, whether the panic was a
// bare runtime.Error (the crash class the oracle forbids), and the message.
func fuzzTry(fn func()) (panicked, bare bool, msg string) {
	defer func() {
		if r := recover(); r != nil {
			panicked = true
			if _, ok := r.(runtime.Error); ok {
				bare = true
			}
			msg = fmt.Sprint(r)
		}
	}()
	fn()
	return
}

// fuzzClassify computes the element product of shape with the same
// overflow-safe multiplication as tensor.Size, reporting negativity and
// int64 overflow separately so the harness can gate real allocations.
// Negativity is scanned over the WHOLE shape (never short-circuited) because
// tensor.New rejects a negative dimension before it ever computes Size — so a
// shape carrying both a negative and an overflowing axis panics "negative
// dimension", and the harness's expectation must match that precedence.
func fuzzClassify(shape []int) (n uint64, neg, overflow bool) {
	n = 1
	for _, d := range shape {
		if d < 0 {
			neg = true
			continue // keep negatives out of the product
		}
		hi, lo := bits.Mul64(n, uint64(d))
		if hi != 0 || lo > math.MaxInt64 {
			overflow = true
			continue
		}
		if !overflow {
			n = lo
		}
	}
	return n, neg, overflow
}

// fuzzEachIndex enumerates every valid index of shape, calling fn with a
// freshly built index slice (reused storage; do not retain it).
func fuzzEachIndex(shape []int, fn func(idx []int)) {
	for _, d := range shape {
		if d == 0 {
			return // a zero-sized axis admits no valid index
		}
	}
	if len(shape) == 0 {
		fn(nil)
		return
	}
	idx := make([]int, len(shape))
	for {
		fn(idx)
		// increment row-major
		i := len(shape) - 1
		for i >= 0 {
			idx[i]++
			if idx[i] < shape[i] {
				break
			}
			idx[i] = 0
			i--
		}
		if i < 0 {
			return
		}
	}
}

// fuzzAssertDescriptive fails the test if the recovered panic was bare or its
// message lacks one of the expected semantic keywords.
func fuzzAssertDescriptive(t *testing.T, what string, panicked, bare bool, msg string, keywords ...string) {
	t.Helper()
	if !panicked {
		t.Fatalf("%s: expected a panic, got success", what)
	}
	if bare {
		t.Fatalf("%s: bare runtime crash (forbidden): %v", what, msg)
	}
	for _, kw := range keywords {
		if strings.Contains(msg, kw) {
			return
		}
	}
	t.Fatalf("%s: panic message %q carries none of the keywords %v", what, msg, keywords)
}

// fuzzAssertUsable verifies the success contract of a constructed tensor:
// Size()==len(Data), Dims()==len(Shape), and every index is reachable through
// At/Set (the full-index scan is capped to keep the harness fast).
func fuzzAssertUsable(t *testing.T, what string, ts *tensor.Tensor, shape []int) {
	t.Helper()
	if len(ts.Shape) != len(shape) {
		t.Fatalf("%s: Dims=%d, want %d", what, len(ts.Shape), len(shape))
	}
	if ts.Size() != len(ts.Data) {
		t.Fatalf("%s: Size()=%d but len(Data)=%d", what, ts.Size(), len(ts.Data))
	}
	if ts.Size() > fuzzElemCap {
		return // full-index scan only for small tensors
	}
	fuzzEachIndex(shape, func(idx []int) {
		v := ts.At(idx...) // must not panic out of bounds
		ts.Set(v+1.5, idx...)
		if got := ts.At(idx...); got != v+1.5 {
			t.Fatalf("%s: Set/At round trip at %v: got %v, want %v", what, idx, got, v+1.5)
		}
	})
}

func FuzzTensorConstructors(f *testing.F) {
	// (seed) dims d0..d3 — includes zero, negative, giant and overflow-edge
	// values; the harness derives ranks 1..4 from the running prefix.
	f.Add(2, 3, 0, 0)             // ordinary small shapes
	f.Add(0, 0, 0, 0)             // legal empties / rank-0 scalar
	f.Add(1, 1, 1, 1)             // degenerate all-ones
	f.Add(-1, 2, 3, 4)            // negative dimension (descriptive panic)
	f.Add(4, -7, 0, 0)            // negative in a later axis
	f.Add(1<<30, 4, 0, 0)         // product overflows int64 (descriptive panic)
	f.Add(1<<62, 1, 0, 0)         // single giant axis (overflows)
	f.Add(math.MaxInt32, 2, 0, 0) // large-but-finite product (skipped alloc)
	f.Add(8, 8, 8, 8)             // rank-4, 4096 elements (full-index scan)

	f.Fuzz(func(t *testing.T, d0, d1, d2, d3 int) {
		dims := []int{d0, d1, d2, d3}
		for rank := 1; rank <= 4; rank++ {
			shape := dims[:rank]
			n, neg, overflow := fuzzClassify(shape)

			// --- New ---
			switch {
			case neg:
				// New checks negativity before any allocation.
				p, bare, msg := fuzzTry(func() { tensor.New(shape...) })
				fuzzAssertDescriptive(t, fmt.Sprintf("New%v", shape), p, bare, msg, "negative dimension")
			case overflow:
				// Size's overflow check fires before the make.
				p, bare, msg := fuzzTry(func() { tensor.New(shape...) })
				fuzzAssertDescriptive(t, fmt.Sprintf("New%v", shape), p, bare, msg, "overflow")
			case n <= fuzzElemCap:
				fuzzAssertUsable(t, fmt.Sprintf("New%v", shape), tensor.New(shape...), shape)
			}

			// --- FromData: correct length succeeds, wrong length panics ---
			if !neg && !overflow && n <= fuzzElemCap {
				data := make([]float32, n)
				for i := range data {
					data[i] = float32(i)
				}
				fuzzAssertUsable(t, fmt.Sprintf("FromData%v", shape), tensor.FromData(data, shape...), shape)
				// One element too many: a size-mismatch panic, never bare.
				p, bare, msg := fuzzTry(func() { tensor.FromData(make([]float32, n+1), shape...) })
				fuzzAssertDescriptive(t, fmt.Sprintf("FromData%v+1", shape), p, bare, msg, "size", "elements")
			} else if neg {
				p, bare, msg := fuzzTry(func() { tensor.FromData(nil, shape...) })
				fuzzAssertDescriptive(t, fmt.Sprintf("FromData%v", shape), p, bare, msg, "negative dimension")
			} else if overflow {
				p, bare, msg := fuzzTry(func() { tensor.FromData(nil, shape...) })
				fuzzAssertDescriptive(t, fmt.Sprintf("FromData%v", shape), p, bare, msg, "overflow")
			}
		}

		// --- FromRows: built from small clamped dims only ---
		rows := clampDim(d0, 0, 4)
		cols := clampDim(d1, 0, 4)
		rowData := make([][]float32, rows)
		for i := range rowData {
			rowData[i] = make([]float32, cols)
		}
		if rows == 0 {
			fr := tensor.FromRows() // New(0,0)
			if fr.Size() != 0 {
				t.Fatalf("FromRows() size %d, want 0", fr.Size())
			}
		} else {
			fr := tensor.FromRows(rowData...)
			if fr.Size() != rows*cols {
				t.Fatalf("FromRows size %d, want %d", fr.Size(), rows*cols)
			}
			// Ragged rows (two rows of differing length): descriptive panic.
			rag := [][]float32{make([]float32, cols), make([]float32, cols+1)}
			p, bare, msg := fuzzTry(func() { tensor.FromRows(rag...) })
			fuzzAssertDescriptive(t, "FromRows ragged", p, bare, msg, "elements")
		}

		// --- Stack probes removed: Stack deleted in v0.4.0 API hygiene ---
		// (zero in-library callers, sole rank-3 producer, no op supports 3D).
		// The remaining constructor probes (New/FromData/FromRows above) keep
		// the same "descriptive panic, never bare crash" oracle for this
		// target; d2/d3 still feed the rank-1..4 prefix loop above.
	})
}

// clampDim maps an arbitrary fuzzed int into [lo, hi].
func clampDim(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
