package tensor

import (
	"math"
	"math/rand"
)

// Uniform returns a tensor with elements drawn from U(lo, hi). If lo > hi the
// interval is mirrored rather than rejected: values fall in [hi, lo]. This is
// deliberate legacy behavior kept for backward compatibility; callers that
// rely on interval bounds should pass lo <= hi.
func Uniform(rng *rand.Rand, lo, hi float32, shape ...int) *Tensor {
	t := New(shape...)
	for i := range t.Data {
		t.Data[i] = lo + (hi-lo)*rng.Float32()
	}
	return t
}

// Randn returns a tensor with elements drawn from N(0, 1) using Box-Muller.
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
