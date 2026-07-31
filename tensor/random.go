package tensor

import (
	"math"
	"math/rand"
)

// Uniform returns a tensor of the given shape with elements drawn from
// U(lo, hi) using rng. If lo > hi the interval is mirrored rather than
// rejected: values fall in [hi, lo]. This is deliberate legacy behavior
// kept for backward compatibility; callers that rely on interval bounds
// should pass lo <= hi. Panics if rng is nil, if any dimension is
// negative (via New), or if the element count overflows int64.
func Uniform(rng *rand.Rand, lo, hi float32, shape ...int) *Tensor {
	t := New(shape...)
	for i := range t.Data {
		t.Data[i] = lo + (hi-lo)*rng.Float32()
	}
	return t
}

// Randn returns a tensor of the given shape with elements drawn from
// N(0, 1) using Box-Muller. Panics if rng is nil, if any dimension is
// negative (via New), or if the element count overflows int64.
//
// The u1 uniform is clamped away from zero at 1e-12 (keeping log(u1) finite
// for deterministic same-seed runs), which hard-truncates the distribution's
// tails at sqrt(-2*ln(1e-12)) ≈ 7.43 standard deviations: no sample ever
// exceeds that magnitude. The omitted tail mass is ~1e-13, so this is
// immaterial for weight/parameter initialization, but it makes Randn
// unsuitable as a general-purpose normal sampler (e.g. Monte Carlo work that
// relies on tail events). Raising or removing the clamp would change the
// output stream under a fixed seed, so the truncation is kept and documented
// rather than fixed.
func Randn(rng *rand.Rand, shape ...int) *Tensor {
	t := New(shape...)
	for i := 0; i+1 < len(t.Data); i += 2 {
		u1 := float64(rng.Float32())
		if u1 < 1e-12 {
			u1 = 1e-12
		}
		u2 := float64(rng.Float32())
		r := math.Sqrt(-2 * math.Log(u1))
		t.Data[i] = float32(r * math.Cos(2*math.Pi*u2))
		t.Data[i+1] = float32(r * math.Sin(2*math.Pi*u2))
	}
	if len(t.Data)%2 == 1 {
		u1 := float64(rng.Float32())
		if u1 < 1e-12 {
			u1 = 1e-12
		}
		u2 := float64(rng.Float32())
		t.Data[len(t.Data)-1] = float32(math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2))
	}
	return t
}
