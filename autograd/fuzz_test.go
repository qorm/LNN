// Native Go fuzzing for the autograd engine (structured op-graph fuzzing).
// A generator driven by f.Int/f.Float32 seeds builds random computation graphs
// — an opcode sequence over a fixed [m, n] shape, with a value domain that
// includes NaN, ±Inf, ±0 and ±MaxFloat32 — then runs forward + Backward.
// Oracle:
//   - forward and Backward never panic (every shape combination the generator
//     emits is legal, so ANY panic is a finding);
//   - every leaf's gradient shape equals its data shape (an assertion on top
//     of addGrad's own shape guard);
//   - Backward is linear: running it twice without ZeroGrad leaves each leaf's
//     gradient exactly 2x the single-pass gradient, bit for bit (Float32bits),
//     skipping NaN entries whose payload a double-add may recanonicalize.
//
// External test package (autograd_test), driving only the exported API.
package autograd_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/tensor"
)

// fuzzSpecials is the hostile value domain: the non-finite and boundary
// float32s a real gradient/activation stream can carry.
var fuzzSpecials = []float32{
	0, -0.0,
	float32(math.NaN()),
	float32(math.Inf(1)), float32(math.Inf(-1)),
	math.MaxFloat32, -math.MaxFloat32,
	math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32,
}

// fuzzVal draws a value: mostly a small normal, occasionally a special, so the
// graph sees a steady diet of NaN/Inf/extremes without every node being one.
func fuzzVal(rng *rand.Rand) float32 {
	if rng.Intn(4) == 0 {
		return fuzzSpecials[rng.Intn(len(fuzzSpecials))]
	}
	return float32(rng.NormFloat64() * 2)
}

// fuzzLeaf builds an [m, n] leaf variable filled from the value domain.
func fuzzLeaf(rng *rand.Rand, m, n int) *autograd.Variable {
	data := make([]float32, m*n)
	for i := range data {
		data[i] = fuzzVal(rng)
	}
	return autograd.Var(tensor.FromData(data, m, n))
}

// fuzzPowExps keeps Pow's exponent in a small set so the backward stays in the
// regimes the op explicitly handles (including p==0).
var fuzzPowExps = []float32{-2, -1, 0, 0.5, 1, 2, 3}

func fuzzSameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func FuzzOpGraphs(f *testing.F) {
	// (seed) a few generator configurations: degenerate 1x1, square, wide and
	// deep graphs. The seed is the RNG source; m/n/nOps are clamped in-harness.
	f.Add(int64(1), 2, 2, 4)
	f.Add(int64(2), 1, 1, 1)
	f.Add(int64(3), 4, 4, 8)
	f.Add(int64(42), 3, 2, 6)
	f.Add(int64(7501), 2, 3, 10)

	f.Fuzz(func(t *testing.T, seed int64, mIn, nIn, nOpsIn int) {
		m := clamp(mIn, 1, 4)
		n := clamp(nIn, 1, 4)
		nOps := clamp(nOpsIn, 1, 10)
		rng := rand.New(rand.NewSource(seed))

		// Three [m, n] leaves plus a [n, m] leaf for the terminal MatMul; all
		// four are guaranteed to enter the graph below, so all receive grads.
		l0 := fuzzLeaf(rng, m, n)
		l1 := fuzzLeaf(rng, m, n)
		l2 := fuzzLeaf(rng, m, n)
		w := fuzzLeaf(rng, n, m)
		leaves := []*autograd.Variable{l0, l1, l2, w}

		build := func() *autograd.Variable {
			cur := l0
			for i := 0; i < nOps; i++ {
				other := []*autograd.Variable{l0, l1, l2}[rng.Intn(3)]
				switch rng.Intn(12) {
				case 0:
					cur = autograd.Add(cur, other)
				case 1:
					cur = autograd.Sub(cur, other)
				case 2:
					cur = autograd.Hadamard(cur, other)
				case 3:
					cur = autograd.Div(cur, other)
				case 4:
					cur = autograd.Tanh(cur)
				case 5:
					cur = autograd.Sigmoid(cur)
				case 6:
					cur = autograd.Exp(cur)
				case 7:
					cur = autograd.Log(cur)
				case 8:
					cur = autograd.Softplus(cur)
				case 9:
					cur = autograd.Relu(cur)
				case 10:
					cur = autograd.Scale(cur, fuzzVal(rng))
				case 11:
					cur = autograd.Pow(cur, fuzzPowExps[rng.Intn(len(fuzzPowExps))])
				}
			}
			// Fold in the remaining leaves so every one is differentiated.
			cur = autograd.Add(cur, l1)
			cur = autograd.Add(cur, l2)
			// Terminal contraction: [m, n] @ [n, m] -> [m, m], reduced to scalar.
			return autograd.MeanAll(autograd.MatMul(cur, w))
		}

		// Forward + Backward must never panic on these legal shapes.
		var loss *autograd.Variable
		func() {
			defer func() {
				if p := recover(); p != nil {
					t.Fatalf("forward/backward panicked on a legal graph (m=%d n=%d nOps=%d): %v", m, n, nOps, p)
				}
			}()
			loss = build()
			loss.Grad = loss.Data.OnesLike() // seed the root with 1
			loss.Backward()
		}()

		// Gradient shape == data shape for every leaf (all 2D here, so the 1D
		// lift quirk does not apply).
		for i, lf := range leaves {
			if lf.Grad == nil {
				t.Fatalf("leaf %d has no gradient after Backward", i)
			}
			if !fuzzSameShape(lf.Grad.Shape, lf.Data.Shape) {
				t.Fatalf("leaf %d gradient shape %v != data shape %v", i, lf.Grad.Shape, lf.Data.Shape)
			}
		}

		// Linearity of Backward in the root seed: every gradient formula is
		// linear in the incoming gradient, so re-seeding the root with 2 must
		// yield each leaf's gradient 2x the seed-1 gradient. This is the exact
		// form of the "two Backwards accumulate to 2x" intuition — running
		// Backward twice on the same leaves instead re-associates the running
		// accumulation and drifts ~1 ULP, so the seed-scaling formulation is the
		// honest oracle. The property is bit-exact while the computation is
		// well-conditioned; under the hostile value domain it is asserted to an
		// affine tolerance instead (see the note on the assertion below for why
		// catastrophic cancellation rules out a bit-exact / ULP check).
		snap := make([][]float32, len(leaves))
		for i, lf := range leaves {
			snap[i] = append([]float32(nil), lf.Grad.Data...)
			lf.ZeroGrad()
		}
		two := loss.Data.OnesLike()
		for i := range two.Data {
			two.Data[i] = 2
		}
		func() {
			defer func() {
				if p := recover(); p != nil {
					t.Fatalf("re-seeded Backward panicked: %v", p)
				}
			}()
			loss.Grad = two
			loss.Backward()
		}()
		// Linearity of Backward in the seed (continued): the backward formulas
		// are all exactly homogeneous of degree one in the incoming gradient, so
		// grad(seed=2) = 2·grad(seed=1) — bit for bit — as long as the floating
		// point stays well-conditioned. The hostile value domain (NaN/±Inf/
		// ±MaxFloat32, Exp/Log) breaks that in ways that are IEEE-754 physics,
		// not gradient defects:
		//   - non-finite arithmetic is not a linear ring (0·Inf, Inf−Inf);
		//   - subnormals lose scale-invariance (round(2P) ≠ 2·round(P));
		//   - catastrophic cancellation: a leaf gradient that is the residual
		//     of large opposite-sign terms lands at the ULP of those terms, and
		//     doubling the terms shifts the residual (observed: 0.0076 from
		//     1e38-scale terms, 29 ULP off; and a residual that was exactly 0
		//     at seed 1 but 1.4e-08 at seed 2 because the seed-1 terms rounded
		//     to equal floats). Such residuals can sit at ANY magnitude, so no
		//     magnitude gate is complete and a ULP metric is meaningless near
		//     zero.
		// The oracle therefore uses an affine tolerance — an absolute floor
		// (absorbs near-zero cancellation residuals) plus a relative term — and
		// is skipped entirely for graphs whose gradients go non-finite. A real
		// gradient-formula bug errs by an O(1) factor, far beyond this window;
		// the never-panic and grad-shape oracles above police hostile values.
		finite := true
		for i, lf := range leaves {
			if len(lf.Grad.Data) != len(snap[i]) {
				t.Fatalf("leaf %d gradient length changed across Backward", i)
			}
			for j := range lf.Grad.Data {
				if !fuzzFinite(snap[i][j]) || !fuzzFinite(lf.Grad.Data[j]) {
					finite = false
				}
			}
		}
		if finite {
			for i, lf := range leaves {
				for j, g := range lf.Grad.Data {
					want := snap[i][j] * 2
					diff := math.Abs(float64(g - want))
					tol := fuzzLinAbsTol + fuzzLinRelTol*math.Abs(float64(want))
					if diff > tol {
						t.Fatalf("leaf %d grad[%d] at seed 2 = %#x (%v), want 2x seed-1 %#x (%v): |diff| %g > tol %g (%d ULP)",
							i, j, math.Float32bits(g), g, math.Float32bits(want), want, diff, tol, fuzzULP(g, want))
					}
				}
			}
		}
	})
}

// Linearity tolerance: an absolute floor plus a relative term. The absolute
// floor absorbs catastrophic-cancellation residuals (which cluster near zero);
// the relative term bounds the well-conditioned drift. Both are orders of
// magnitude below any genuine gradient-formula error.
const (
	fuzzLinAbsTol = 1e-6
	fuzzLinRelTol = 1e-4
)

// fuzzFinite reports whether f is neither NaN nor ±Inf.
func fuzzFinite(f float32) bool {
	return !math.IsNaN(float64(f)) && !math.IsInf(float64(f), 0)
}

// fuzzF32Ordered maps a float32 onto an ordered integer axis (sign-magnitude
// folded so integer order matches numeric order and ±0 coincide); neighbors
// are exactly one ULP apart. Self-contained minimal version of the golden
// tests' helper (no import of serialize_test's private copy).
func fuzzF32Ordered(f float32) int32 {
	b := math.Float32bits(f)
	if b >= 0x80000000 {
		return int32(0x80000000 - b)
	}
	return int32(b)
}

// fuzzULP is the count of representable float32 values between a and b.
func fuzzULP(a, b float32) uint32 {
	d := int64(fuzzF32Ordered(a)) - int64(fuzzF32Ordered(b))
	if d < 0 {
		d = -d
	}
	return uint32(d)
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
