package autograd

// Regression tests for the red-team autograd findings F1–F3 (differential
// fuzz vs the 1aab2de baseline). Every expected value below was measured
// against the untouched baseline implementation, not derived by hand.

import (
	"math"
	"math/rand"
	"testing"

	"lnn/tensor"
)

// TestHadamardReduce1DLiftShapeContract is the red team's minimal F1 repro:
// a [1] leaf squared through Hadamard(x, x), differenced against a [2, 1]
// constant, summed and back-propagated. The sameShape fast path used to hand
// the leaf the raw product, which broadcastBinary lifts to [1, 1]; the legacy
// chain's SumToShape flattened the lift away, so the historical gradient is
// [1]·12. The fix keeps the fast path only when the product's own broadcast
// shape equals the target.
func TestHadamardReduce1DLiftShapeContract(t *testing.T) {
	x := New([]float32{3}, 1)
	n := Hadamard(x, x)
	r := Sub(n, Const(tensor.FromData([]float32{1, 1}, 2, 1)))
	SumAll(r).Backward()

	if got := x.Grad.Shape; len(got) != 1 || got[0] != 1 {
		t.Fatalf("x.Grad shape = %v, want [1] (oracle-measured)", got)
	}
	if got := x.Grad.Data[0]; math.Float32bits(got) != 0x41400000 { // 12
		t.Fatalf("x.Grad[0] = %#08x (%v), want 0x41400000 (12)", math.Float32bits(got), got)
	}
}

// TestHadamardReduceMixedContributionsNoPanic is the F1 panic regression: the
// same leaf receives one contribution through the reducing path (shape [1])
// and one through the Hadamard fast path (shape [1, 1] under the old fast
// path), and addGrad's shape assertion exploded — the baseline accumulated
// both into a [1] gradient of value 14 without panicking.
func TestHadamardReduceMixedContributionsNoPanic(t *testing.T) {
	x := New([]float32{3}, 1)
	c := Const(tensor.FromData([]float32{1, 1}, 2, 1))
	r := Add(Sub(Hadamard(x, x), c), Sub(x, c))

	defer func() {
		if p := recover(); p != nil {
			t.Fatalf("Backward panicked (baseline ran clean): %v", p)
		}
	}()
	SumAll(r).Backward()
	if got := x.Grad.Shape; len(got) != 1 || got[0] != 1 {
		t.Fatalf("x.Grad shape = %v, want [1]", got)
	}
	if got := x.Grad.Data[0]; math.Float32bits(got) != 0x41600000 { // 14
		t.Fatalf("x.Grad[0] = %#08x (%v), want 0x41600000 (14)", math.Float32bits(got), got)
	}
}

// TestHadamardReduce1DSeedShape covers the 1D [n] leaf variant: a manually
// seeded 1D gradient on a Hadamard node used to leak the lifted [1, n]
// product shape into the leaf; the baseline reduced it back to [n].
func TestHadamardReduce1DSeedShape(t *testing.T) {
	x := New([]float32{1, 2, 3}, 3)
	y := New([]float32{4, 5, 6}, 3)
	h := Hadamard(x, y)
	h.Grad = tensor.FromData([]float32{1, 1, 1}, 3)
	h.Backward()

	for _, pair := range []struct {
		name string
		g    *tensor.Tensor
		want []uint32
	}{
		{"x", x.Grad, []uint32{0x40800000, 0x40a00000, 0x40c00000}}, // 4, 5, 6
		{"y", y.Grad, []uint32{0x3f800000, 0x40000000, 0x40400000}}, // 1, 2, 3
	} {
		if got := pair.g.Shape; len(got) != 1 || got[0] != 3 {
			t.Fatalf("%s.Grad shape = %v, want [3]", pair.name, got)
		}
		for i, w := range pair.want {
			if b := math.Float32bits(pair.g.Data[i]); b != w {
				t.Fatalf("%s.Grad[%d] = %#08x, want %#08x", pair.name, i, b, w)
			}
		}
	}
}

// TestSignedZeroNormalizedThroughHadamard is the red team's F3 repro: a [4]
// leaf through Tanh → Pow(·,3) → Hadamard(·,·) with a manually seeded 1D
// gradient whose fourth element multiplies a negative seed by a zero tanh
// output. The -0 product used to leak straight into the leaf as 0x80000000;
// the legacy chain's SumToShape reduction accumulated it into a fresh +0
// buffer, normalizing the sign. F1's fix restores the reduction and with it
// the +0 normalization (the seeded 1D shape is what drives the fast path).
func TestSignedZeroNormalizedThroughHadamard(t *testing.T) {
	x := New([]float32{0.5, -0.7, 1.2, 0}, 4)
	p := Pow(Tanh(x), 3)
	h := Hadamard(p, p)
	h.Grad = tensor.FromData([]float32{-1, 2, 3, -4}, 4)
	h.Backward()

	if got := x.Grad.Shape; len(got) != 2 || got[0] != 1 || got[1] != 4 {
		t.Fatalf("x.Grad shape = %v, want [1 4]", got)
	}
	if i := 3; math.Float32bits(x.Grad.Data[i]) != 0x00000000 {
		t.Fatalf("x.Grad[%d] = %#08x, want 0x00000000 (+0, oracle-measured)", i, math.Float32bits(x.Grad.Data[i]))
	}
	// The remaining elements pin the whole gradient to the oracle bits.
	want := []uint32{0xbdcba9a9, 0xbf1d39c4, 0x400d7c41, 0x00000000}
	for i, w := range want {
		if b := math.Float32bits(x.Grad.Data[i]); b != w {
			t.Fatalf("x.Grad[%d] = %#08x, want %#08x", i, b, w)
		}
	}
}

// TestHadamardReduceScalarSeedShape covers the irregular-seed shape corner
// the differential fuzz shook out: a 1D [1] seed over a [1, 1] node. The
// legacy chain materialized Hadamard(g, x) first — broadcast to [1, 1],
// equal to the target, so SumToShape passed it through at [1, 1]; the fused
// scalar branch must not collapse that to [1] (nor flat-index a broadcast
// operand). Oracle-measured bits.
func TestHadamardReduceScalarSeedShape(t *testing.T) {
	x := New([]float32{-0.8392819}, 1, 1)
	h := Hadamard(x, x)
	h.Grad = tensor.FromData([]float32{0.05849433}, 1)
	h.Backward()

	if got := x.Grad.Shape; len(got) != 2 || got[0] != 1 || got[1] != 1 {
		t.Fatalf("x.Grad shape = %v, want [1 1]", got)
	}
	if b := math.Float32bits(x.Grad.Data[0]); b != 0xbdc915fc {
		t.Fatalf("x.Grad[0] = %#08x, want 0xbdc915fc", b)
	}
}

// TestHadamardReduceLiftPassthrough is the graph the 52k-graph fuzz shook
// out after F1: a [1] gradient arriving at a Hadamard whose operand shapes
// are [1, 1] and [1]. The legacy chain materialized Hadamard([1], [1]) —
// lifted to [1, 1] — which equals the target, so SumToShape passed it
// through at [1, 1]; a fused scalar reduction would have collapsed that to
// [1] and later panicked against the [1, 1] contribution GatherRows hands
// the same node. Oracle-measured bits.
func TestHadamardReduceLiftPassthrough(t *testing.T) {
	x := New([]float32{3}, 1)
	b := Scale(Hadamard(x, x), 0.5) // [1, 1]
	c := GatherRows(b, []int{0})    // [1]
	h := Hadamard(b, c)             // [1, 1]
	h.Grad = tensor.FromData([]float32{1}, 1)

	defer func() {
		if p := recover(); p != nil {
			t.Fatalf("Backward panicked (baseline ran clean): %v", p)
		}
	}()
	h.Backward()
	if got := x.Grad.Shape; len(got) != 1 || got[0] != 1 {
		t.Fatalf("x.Grad shape = %v, want [1]", got)
	}
	if got := math.Float32bits(x.Grad.Data[0]); got != 0x41d80000 {
		t.Fatalf("x.Grad[0] = %#08x, want 0x41d80000", got)
	}
}

// TestDivNaNPropagationMatchesNative pins F2 at the graph level: dividing a
// constant by a denominator carrying +NaN must propagate the NaN with the
// hardware multiply's sign/payload (what the legacy Neg(Hadamard(…)) chain
// produced), not the sign-flipped NaN a fused unary minus would emit.
func TestDivNaNPropagationMatchesNative(t *testing.T) {
	nan := math.Float32frombits(0x7fc00000)
	c := Const(tensor.FromData([]float32{1, 2, 3, 4}, 2, 2))
	y := New([]float32{nan, nan, 0.5, 2}, 2, 2)
	SumAll(Div(c, y)).Backward()

	want := []uint32{0x7fc00000, 0x7fc00000, 0xc1400000, 0xbf800000}
	for i, w := range want {
		if b := math.Float32bits(y.Grad.Data[i]); b != w {
			t.Fatalf("y.Grad[%d] = %#08x, want %#08x", i, b, w)
		}
	}
}

// TestMul32MatchesNativeMultiply is the F2 verification at the operator
// level: across 10 million operand pairs — uniform random bit patterns,
// subnormals, ±0, ±Inf and NaN payloads of both signs — mul32 must agree
// bit for bit with a native hardware float32 multiply. Finite agreement is
// the load-bearing invariant (the fused backward loops depend on it); the
// non-finite arm of mul32 extends it to NaN payloads, which the float64
// round-trip used to recanonicalize.
func TestMul32MatchesNativeMultiply(t *testing.T) {
	special := []float32{
		0, float32(math.Copysign(0, -1)),
		float32(math.Inf(1)), float32(math.Inf(-1)),
		math.Float32frombits(0x7FC00000), // quiet NaN
		math.Float32frombits(0xFFC00000), // negative quiet NaN
		math.Float32frombits(0x7F800001), // signaling NaN
		math.Float32frombits(0x7FC00001), // NaN with payload
		math.Float32frombits(0x00000001), // smallest subnormal
		math.Float32frombits(0x80000001), // negative smallest subnormal
		math.MaxFloat32, -math.MaxFloat32,
		math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32,
	}
	rng := rand.New(rand.NewSource(99))
	var mismatches uint64
	const pairs = 10_000_000
	for i := 0; i < pairs; i++ {
		var a, b float32
		switch i % 4 {
		case 0: // fully random bit patterns (hits NaN/Inf/subnormal territory)
			a = math.Float32frombits(rng.Uint32())
			b = math.Float32frombits(rng.Uint32())
		case 1: // special value against a random finite
			a = special[rng.Intn(len(special))]
			b = math.Float32frombits(rng.Uint32() & 0x807FFFFF) // finite (possibly subnormal)
		case 2: // two random finites, biased toward the exponent extremes
			a = math.Float32frombits(rng.Uint32() & 0x807FFFFF)
			b = math.Float32frombits(rng.Uint32() & 0x807FFFFF)
		default: // special × special
			a = special[rng.Intn(len(special))]
			b = special[rng.Intn(len(special))]
		}
		got, want := math.Float32bits(mul32(a, b)), math.Float32bits(a*b)
		if got != want {
			if mismatches < 8 {
				t.Logf("mismatch: mul32(%#08x, %#08x) = %#08x, native %#08x",
					math.Float32bits(a), math.Float32bits(b), got, want)
			}
			mismatches++
		}
	}
	if mismatches > 0 {
		t.Fatalf("%d / %d pairs differ from native float32 multiply", mismatches, pairs)
	}
}
