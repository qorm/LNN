package autograd

import (
	"math"
	"math/rand"
	"testing"

	"github.com/qorm/LNN/tensor"
)

// legacySigmoidHadamard rebuilds the composition SigmoidHadamard fuses, so
// tests can pin the fused node to the exact historical behavior.
func legacySigmoidHadamard(z, w *Variable) *Variable {
	return Hadamard(Sigmoid(z), w)
}

func bitsEqual(a, b *tensor.Tensor) bool {
	if !tensor.SameShape(a, b) {
		return false
	}
	for i := range a.Data {
		if math.Float32bits(a.Data[i]) != math.Float32bits(b.Data[i]) {
			return false
		}
	}
	return true
}

// TestSigmoidHadamardForwardBitwise pins the forward contract: across every
// broadcast combination the library supports and across saturation/extreme
// inputs, the fused node must produce the exact output shape and bit-identical
// values (Float32bits, NaN payloads and ±0 included) of the composition
// Hadamard(Sigmoid(z), w). The forward has no association-order freedom — it
// runs the same two tensor operations — so anything but bitwise equality is a
// bug.
func TestSigmoidHadamardForwardBitwise(t *testing.T) {
	rng := rand.New(rand.NewSource(77))
	cases := []struct {
		name string
		z, w *tensor.Tensor
	}{
		{"same shape", tensor.Uniform(rng, -6, 6, 2, 3), tensor.Uniform(rng, -2, 2, 2, 3)},
		{"same shape larger", tensor.Uniform(rng, -9, 9, 5, 7), tensor.Uniform(rng, -2, 2, 5, 7)},
		{"row broadcast 2D", tensor.Uniform(rng, -6, 6, 4, 5), tensor.Uniform(rng, -2, 2, 1, 5)},
		{"row broadcast 1D", tensor.Uniform(rng, -6, 6, 4, 5), tensor.Uniform(rng, -2, 2, 5)},
		{"col broadcast", tensor.Uniform(rng, -6, 6, 4, 5), tensor.Uniform(rng, -2, 2, 4, 1)},
		{"scalar w", tensor.Uniform(rng, -6, 6, 3, 3), tensor.FromData([]float32{-1.5}, 1)},
		{"outer col/row", tensor.Uniform(rng, -6, 6, 3, 1), tensor.Uniform(rng, -2, 2, 4)},
		{"outer row/col", tensor.Uniform(rng, -6, 6, 4), tensor.Uniform(rng, -2, 2, 3, 1)},
		{"1D over 1D", tensor.Uniform(rng, -6, 6, 6), tensor.Uniform(rng, -2, 2, 6)},
		{"scalar over scalar", tensor.FromData([]float32{2.5}, 1), tensor.FromData([]float32{-3}, 1)},
		{
			"saturation and extremes",
			tensor.FromData([]float32{
				40, -40, 100, -100, 0, float32(math.Copysign(0, -1)),
				math.MaxFloat32, -math.MaxFloat32,
				float32(math.SmallestNonzeroFloat64), // tiny subnormal
				float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()),
			}, 3, 4),
			tensor.FromData([]float32{2, -2, 0, float32(math.Copysign(0, -1)), 1.5, -1.5, 1e20, -1e20, 1, 1, 1, 1}, 3, 4),
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fused := SigmoidHadamard(Var(tc.z.Clone()), Var(tc.w.Clone()))
			legacy := legacySigmoidHadamard(Var(tc.z.Clone()), Var(tc.w.Clone()))
			if !tensor.SameShape(fused.Data, legacy.Data) {
				t.Fatalf("shape drift: fused %v vs legacy %v", fused.Data.Shape, legacy.Data.Shape)
			}
			for i := range fused.Data.Data {
				if math.Float32bits(fused.Data.Data[i]) != math.Float32bits(legacy.Data.Data[i]) {
					t.Fatalf("elem %d: fused %v (bits %#x) vs legacy %v (bits %#x)",
						i, fused.Data.Data[i], math.Float32bits(fused.Data.Data[i]),
						legacy.Data.Data[i], math.Float32bits(legacy.Data.Data[i]))
				}
			}
			// The sigmoid buffer is recorded once on the node for the backward
			// to reuse — it must carry sigmoid(z) exactly (no recomputation).
			if fused.aux == nil {
				t.Fatal("fused node did not record the sigmoid auxiliary tensor")
			}
			if want := tensor.Sigmoid(tc.z); !bitsEqual(fused.aux, want) {
				t.Fatal("auxiliary tensor differs from sigmoid(z)")
			}
		})
	}
}

// signedVar draws magnitudes in [0.5, 2] with alternating signs: gradients of
// σ(z)⊙w scale with |w| and with σ′(z), so w values near zero leave the
// analytic gradient at ~1e-3 where float32 central differences carry ~1e-5
// noise and the relative check degenerates into noise-vs-noise. Keeping |w|
// bounded away from zero (and z inside the non-saturating [-4, 4] range)
// conditions the finite differences without loosening the 2e-2 tolerance.
func signedVar(rng *rand.Rand, shape ...int) *Variable {
	v := tensor.Uniform(rng, 0.5, 2, shape...)
	for i := range v.Data {
		if i%2 == 0 {
			v.Data[i] = -v.Data[i]
		}
	}
	return Var(v)
}

// TestSigmoidHadamardGradCheck validates the fused backward against central
// finite differences for both operands, on the same shape and every broadcast
// combination, plus the shared-input case, across three seeds. Tolerance is
// gradCheck's 2e-2: the backward's multiplication association differs from a
// naive reading of g⊙w⊙s⊙(1−s), so the gate is deliberately tolerance-based.
func TestSigmoidHadamardGradCheck(t *testing.T) {
	f := func(v ...*Variable) *Variable { return SigmoidHadamard(v[0], v[1]) }
	for _, seed := range []int64{83, 84, 85} {
		rng := rand.New(rand.NewSource(seed))
		z := func(shape ...int) *Variable { return randVar(rng, -4, 4, shape...) }

		gradCheck(t, "same shape", f, z(2, 3), signedVar(rng, 2, 3))
		gradCheck(t, "row broadcast 2D", f, z(2, 3), signedVar(rng, 1, 3))
		gradCheck(t, "row broadcast 1D", f, z(2, 3), signedVar(rng, 3))
		gradCheck(t, "col broadcast", f, z(2, 3), signedVar(rng, 2, 1))
		gradCheck(t, "scalar w", f, z(2, 3), signedVar(rng, 1))
		gradCheck(t, "outer product", f, z(2, 1), signedVar(rng, 3))
		gradCheck(t, "1D over 1D", f, z(3), signedVar(rng, 3))
		gradCheck(t, "shared input", func(v ...*Variable) *Variable { return SigmoidHadamard(v[0], v[0]) },
			z(2, 3))
	}
}

// backwardBitwiseVsLegacy runs SumAll(...).Backward() through both the fused
// node and the legacy composition from identical leaves and requires both
// operand gradients to agree bit for bit (Float32bits) — the fused loop
// rounds each product at exactly the spots the two-node chain did.
func backwardBitwiseVsLegacy(t *testing.T, name string, zt, wt *tensor.Tensor) {
	t.Helper()
	z1, w1 := Var(zt.Clone()), Var(wt.Clone())
	z2, w2 := Var(zt.Clone()), Var(wt.Clone())
	SumAll(SigmoidHadamard(z1, w1)).Backward()
	SumAll(legacySigmoidHadamard(z2, w2)).Backward()
	for _, pair := range []struct {
		label    string
		got, ref *tensor.Tensor
	}{{"dz", z1.Grad, z2.Grad}, {"dw", w1.Grad, w2.Grad}} {
		if pair.got == nil || pair.ref == nil {
			t.Fatalf("%s: missing %s gradient (fused %v, legacy %v)", name, pair.label, pair.got, pair.ref)
		}
		if !tensor.SameShape(pair.got, pair.ref) {
			t.Fatalf("%s: %s shape drift: fused %v vs legacy %v", name, pair.label, pair.got.Shape, pair.ref.Shape)
		}
		for i := range pair.got.Data {
			if math.Float32bits(pair.got.Data[i]) != math.Float32bits(pair.ref.Data[i]) {
				t.Fatalf("%s: %s elem %d: fused %v (bits %#x) vs legacy %v (bits %#x)",
					name, pair.label, i, pair.got.Data[i], math.Float32bits(pair.got.Data[i]),
					pair.ref.Data[i], math.Float32bits(pair.ref.Data[i]))
			}
		}
	}
}

// TestSigmoidHadamardBackwardBitwiseVsLegacy pins the stronger-than-required
// property of the regular 2D path: because the fused loop keeps the legacy
// rounding order (rounded g⊙w product first, then mul32(gw, s⊙(1−s)), and
// the same fused reduction for dw), the gradients match the composition
// bitwise — not merely within tolerance — for same-shape operands and every
// row/column broadcast the LTC hot path exercises.
func TestSigmoidHadamardBackwardBitwiseVsLegacy(t *testing.T) {
	rng := rand.New(rand.NewSource(89))
	cases := []struct {
		name string
		z, w *tensor.Tensor
	}{
		{"same shape", tensor.Uniform(rng, -6, 6, 3, 4), tensor.Uniform(rng, -2, 2, 3, 4)},
		{"row broadcast 2D (LTC pattern)", tensor.Uniform(rng, -6, 6, 5, 4), tensor.Uniform(rng, -2, 2, 1, 4)},
		{"col broadcast", tensor.Uniform(rng, -6, 6, 3, 4), tensor.Uniform(rng, -2, 2, 3, 1)},
		{"scalar w", tensor.Uniform(rng, -6, 6, 3, 4), tensor.FromData([]float32{1.25}, 1)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) { backwardBitwiseVsLegacy(t, tc.name, tc.z, tc.w) })
	}
}

// TestSigmoidHadamardSeededAndFallbackPaths covers manually seeded gradients:
// a regular seed, a broadcastable-but-irregular seed (which takes the legacy
// fallback branch), and 1D operands whose gradients must keep the documented
// [1, n] lift quirk. In every case the fused node must reproduce the legacy
// composition bit for bit, shapes included.
func TestSigmoidHadamardSeededAndFallbackPaths(t *testing.T) {
	seeded := func(name string, zt, wt *tensor.Tensor, seed *tensor.Tensor) {
		t.Helper()
		z1, w1 := Var(zt.Clone()), Var(wt.Clone())
		z2, w2 := Var(zt.Clone()), Var(wt.Clone())
		fused := SigmoidHadamard(z1, w1)
		legacy := legacySigmoidHadamard(z2, w2)
		fused.Grad = seed.Clone()
		legacy.Grad = seed.Clone()
		fused.Backward()
		legacy.Backward()
		for _, pair := range []struct {
			label    string
			got, ref *tensor.Tensor
		}{{"dz", z1.Grad, z2.Grad}, {"dw", w1.Grad, w2.Grad}} {
			if pair.got == nil || pair.ref == nil {
				t.Fatalf("%s: missing %s gradient", name, pair.label)
			}
			if !bitsEqual(pair.got, pair.ref) {
				t.Fatalf("%s: %s differs from legacy: fused %v (shape %v) vs %v (shape %v)",
					name, pair.label, pair.got.Data, pair.got.Shape, pair.ref.Data, pair.ref.Shape)
			}
		}
	}
	seeded("regular seed",
		tensor.FromData([]float32{0.5, -1.5, 2, -3}, 2, 2),
		tensor.FromData([]float32{1, 2, -1, 0.5}, 2, 2),
		tensor.FromData([]float32{1, 10, 100, 1000}, 2, 2))
	seeded("irregular broadcastable seed",
		tensor.FromData([]float32{0.5, -1.5, 2, -3}, 2, 2),
		tensor.FromData([]float32{1, 2, -1, 0.5}, 2, 2),
		tensor.FromData([]float32{2, 4}, 1, 2))
	seeded("1D operands keep the lift quirk",
		tensor.FromData([]float32{0.5, -1.5, 2}, 3),
		tensor.FromData([]float32{1, 2, -1}, 3),
		tensor.FromData([]float32{1, 10, 100}, 1, 3))
}

// TestSigmoidHadamardSingleNodeGraph verifies the fusion itself: the fused
// form records one op node wired directly to z and w, where the composition
// recorded two (Hadamard over Sigmoid).
func TestSigmoidHadamardSingleNodeGraph(t *testing.T) {
	countOps := func(root *Variable) int {
		n := 0
		seen := map[*Variable]bool{}
		var walk func(v *Variable)
		walk = func(v *Variable) {
			if seen[v] {
				return
			}
			seen[v] = true
			if v.numParents() > 0 {
				n++
			}
			for _, p := range v.parentsSlice() {
				walk(p)
			}
		}
		walk(root)
		return n
	}
	z := New([]float32{1, 2, 3, 4}, 2, 2)
	w := New([]float32{4, 3, 2, 1}, 2, 2)

	f := SigmoidHadamard(z, w)
	if f.numParents() != 2 || f.parent(0) != z || f.parent(1) != w {
		t.Fatalf("SigmoidHadamard must wire z and w directly, got %d parents", f.numParents())
	}
	if got := countOps(f); got != 1 {
		t.Fatalf("fused graph has %d op nodes, want 1", got)
	}
	if got := countOps(legacySigmoidHadamard(z, w)); got != 2 {
		t.Fatalf("legacy composition has %d op nodes, want 2", got)
	}
}

// TestSigmoidHadamardBackwardTwiceLinear checks the opKind dispatch under the
// repeated-Backward contract (V-09): two runs over the same fused graph must
// exactly double the leaf gradients.
func TestSigmoidHadamardBackwardTwiceLinear(t *testing.T) {
	z := New([]float32{0.5, -1, 1.5, -2}, 2, 2)
	w := New([]float32{1, 2, 3, 4}, 2, 2)
	y := SumAll(SigmoidHadamard(z, w))

	y.Backward()
	zOnce := append([]float32(nil), z.Grad.Data...)
	wOnce := append([]float32(nil), w.Grad.Data...)
	y.Backward()
	for i := range zOnce {
		if got, want := z.Grad.Data[i], 2*zOnce[i]; math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("dz elem %d: after two Backward calls = %v, want %v", i, got, want)
		}
		if got, want := w.Grad.Data[i], 2*wOnce[i]; math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("dw elem %d: after two Backward calls = %v, want %v", i, got, want)
		}
	}
}
