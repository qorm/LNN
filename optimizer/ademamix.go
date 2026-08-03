package optimizer

import (
	"fmt"
	"math"

	"github.com/qorm/LNN/autograd"
)

// AdEMAMix implements the AdEMAMix update of Pagliardini, Ablin &
// Grangier, "The AdEMAMix Optimizer: Better, Faster, Older"
// (arXiv:2409.03137, ICLR 2025): Adam augmented with a second, slow EMA
// of the gradient, mixed in with coefficient Alpha, so that very old
// gradients keep a non-negligible influence while the fast EMA stays
// reactive. Per element of each parameter, at update count t:
//
//	t  += 1
//	m1  = Beta1*m1 + (1-Beta1)*g        // fast EMA
//	m2  = Beta3(t)*m2 + (1-Beta3(t))*g  // slow EMA, NO bias correction
//	v   = Beta2*v + (1-Beta2)*g*g
//	m1' = m1 / (1 - Beta1^t)
//	v'  = v / (1 - Beta2^t)
//	p  -= LR * (m1' + Alpha(t)*m2) / (sqrt(v') + Eps)
//
// The slow EMA m2 deliberately carries no bias correction: with
// Beta3 = 0.9999 the correction factor 1-Beta3^t stays tiny for
// thousands of steps and dividing by it would blow up the early
// updates (the paper's Fig. 27 divergence). Instead the buffer fills
// slowly and the warmup schedulers below ramp its influence.
//
// A corollary worth knowing before long zero-gradient pauses (a frozen
// loss, a data gap, gradient accumulation with the graph paused):
// while g = 0, m2 decays as Beta3^t and v as Beta2^t, so the m2
// contribution to the update magnitude evolves as
// (Beta3/sqrt(Beta2))^t — with the defaults (0.9999, 0.999) that ratio
// is about 1.0004 per step, i.e. the update GROWS exponentially during
// the pause until float32 limits bottom out. This is inherent to the
// paper's design (an uncorrected slow EMA over a decaying second
// moment), not an implementation defect; avoid long zero-gradient
// stretches, or expect a transient when they end.
//
// Like this package's Adam, the update has no decoupled weight-decay
// term (the paper's lambda / the official implementation's
// weight_decay): users porting a recipe that calls for one must apply
// it themselves.
//
// # Warmup schedulers
//
// Alpha and Beta3 are warmed up over the first Warmup steps (the
// paper's T_{alpha,beta3}, usually set to the total iteration count);
// with Warmup = 0 both stay at their final values from the first step.
// The schedules, evaluated at the update count t = 1..Warmup-1 exactly
// as in the paper's equations f_alpha and f_beta3 (and the official
// apple/ml-ademamix implementation):
//
//	Alpha(t) = t/T * Alpha                                     (linear from 0)
//	Beta3(t) = fInv((1-t/T)*f(Beta1) + (t/T)*f(Beta3))
//	  with f(b) = log(0.5)/log(b+1e-8) - 1, fInv(x) = 0.5^(1/(x+1))
//
// The Beta3 schedule is linear in the EMA half-life, not in Beta3
// itself — a fixed increment of Beta3 matters far more near 1 than
// near 0.9, so the paper interpolates the half-life instead (its
// Appendix A.1). From t >= Warmup on, Alpha and Beta3 are used as
// written and no scheduler arithmetic runs. The schedules are computed
// in float64 (one scalar per step, via math.Log/math.Pow, which are
// deterministic in Go) and cast to float32.
//
// All moment state is float32, per the library convention — the same
// self-normalizing argument as Adam applies to the m1'/sqrt(v') part,
// while Alpha*m2/sqrt(v') is bounded near Alpha in steady state.
type AdEMAMix struct {
	// LR is the learning rate, Beta1/Beta2 the fast EMA and second
	// moment decays, Beta3 the slow EMA decay, Alpha the slow-EMA
	// mixing coefficient, Eps the divisor guard. Warmup is the number
	// of steps T_{alpha,beta3} over which Alpha and Beta3 ramp from
	// 0 and Beta1 to their written values (0 disables the schedulers).
	// NewAdEMAMix requires LR > 0, Beta1/Beta2/Beta3 in [0, 1),
	// Alpha >= 0, Warmup >= 0 and Eps > 0; Step uses the fields as
	// written — editing Warmup mid-training restarts the schedules
	// against the current update count, exactly the arithmetic asked
	// for (the same trust model as Adam's exported fields).
	LR, Beta1, Beta2, Beta3, Alpha, Eps float32
	Warmup                              int

	// state maps each parameter pointer to its moment buffers and
	// update count. The map pins every parameter it has seen, so use a
	// fresh optimizer per model.
	state map[*autograd.Variable]*ademamixState
}

// ademamixState holds one parameter's AdEMAMix state. t counts the
// updates applied to this parameter (parameters skipped because Grad
// was nil do not advance it, so the bias corrections and scheduler
// lookups stay exact); pow1 and pow2 are Beta1^t and Beta2^t kept as
// running float32 products, the same O(1) bias-correction discipline
// as Adam. The scheduled Alpha(t)/Beta3(t) are pure functions of t and
// the exported fields, so they are recomputed per step rather than
// stored.
type ademamixState struct {
	m1, m2, v  []float32
	t          int
	pow1, pow2 float32
}

// NewAdEMAMix returns an AdEMAMix optimizer. It panics unless lr > 0,
// 0 <= beta1 < 1, 0 <= beta2 < 1, 0 <= beta3 < 1, alpha >= 0,
// warmup >= 0 and eps > 0 (NaN fails the checks and panics too).
// warmup is the paper's T_{alpha,beta3}: the number of initial steps
// over which Alpha ramps linearly from 0 and Beta3 from Beta1 (the
// paper always sets the two warmup times equal, and typically to the
// total iteration count); warmup = 0 keeps both constant.
func NewAdEMAMix(lr, beta1, beta2, beta3, alpha float32, warmup int, eps float32) *AdEMAMix {
	if !(lr > 0) {
		panic(fmt.Sprintf("optimizer.NewAdEMAMix: learning rate must be positive, got %v", lr))
	}
	if !(beta1 >= 0 && beta1 < 1) {
		panic(fmt.Sprintf("optimizer.NewAdEMAMix: beta1 must be in [0, 1), got %v", beta1))
	}
	if !(beta2 >= 0 && beta2 < 1) {
		panic(fmt.Sprintf("optimizer.NewAdEMAMix: beta2 must be in [0, 1), got %v", beta2))
	}
	if !(beta3 >= 0 && beta3 < 1) {
		panic(fmt.Sprintf("optimizer.NewAdEMAMix: beta3 must be in [0, 1), got %v", beta3))
	}
	if !(alpha >= 0) {
		panic(fmt.Sprintf("optimizer.NewAdEMAMix: alpha must be non-negative, got %v", alpha))
	}
	if warmup < 0 {
		panic(fmt.Sprintf("optimizer.NewAdEMAMix: warmup must be non-negative, got %v", warmup))
	}
	if !(eps > 0) {
		panic(fmt.Sprintf("optimizer.NewAdEMAMix: eps must be positive, got %v", eps))
	}
	return &AdEMAMix{
		LR: lr, Beta1: beta1, Beta2: beta2, Beta3: beta3, Alpha: alpha, Eps: eps,
		Warmup: warmup,
		state:  make(map[*autograd.Variable]*ademamixState),
	}
}

// NewAdEMAMixDefault returns AdEMAMix with the hyperparameters the
// paper uses for its language-model experiments: Beta1 = 0.9,
// Beta2 = 0.999, Beta3 = 0.9999, Alpha = 5, Eps = 1e-8. The paper
// reports Alpha in [4, 10] working well in practice and uses 5 in its
// language-model runs (e.g. the 1.3B constant-η experiment of its
// Fig. 3(b)); the official repo's usage example uses 8. warmup is
// T_{alpha,beta3} — the paper typically sets it to the total number of
// training iterations; pass 0 to disable the schedulers.
func NewAdEMAMixDefault(lr float32, warmup int) *AdEMAMix {
	return NewAdEMAMix(lr, 0.9, 0.999, 0.9999, 5, warmup, 1e-8)
}

// ademamixAlpha returns the scheduled mixing coefficient at update
// count t (1-based, t < warmup): the paper's f_alpha, a linear ramp
// from 0 to alpha — alpha*(t/warmup), computed in float64.
func ademamixAlpha(t, warmup int, alpha float32) float32 {
	return float32(float64(t) / float64(warmup) * float64(alpha))
}

// ademamixBeta3 returns the scheduled slow-EMA decay at update count t
// (1-based, t < warmup): the paper's f_beta3, linear interpolation of
// the EMA half-life between betaStart (always Beta1 in the paper's
// experiments) and betaEnd. In the official implementation's form:
//
//	f(b)    = log(0.5)/log(b+1e-8) - 1   (half-life of an EMA with decay b)
//	fInv(x) = 0.5^(1/(x+1))
//	beta3(t) = fInv((1-a)*f(betaStart) + a*f(betaEnd)), a = t/warmup
//
// computed in float64 — one scalar per step, so the transcendentals
// cost nothing next to a graph backward, and Go's math package is
// platform-deterministic.
func ademamixBeta3(t, warmup int, betaStart, betaEnd float32) float32 {
	f := func(beta float64) float64 { return math.Log(0.5)/math.Log(beta+1e-8) - 1 }
	fInv := func(x float64) float64 { return math.Pow(0.5, 1/(x+1)) }
	a := float64(t) / float64(warmup)
	return float32(fInv((1-a)*f(float64(betaStart)) + a*f(float64(betaEnd))))
}

// Step updates in place every parameter with a non-nil Grad, carrying
// its moment buffers and update count across calls. A parameter whose
// Data is resized between steps panics: its stored moments no longer
// match. See Optimizer for the nil-Grad and ZeroGrad contract.
func (o *AdEMAMix) Step(params []*autograd.Variable) {
	for _, p := range params {
		if p.Grad == nil {
			continue
		}
		d, g := p.Data.Data, p.Grad.Data
		st := o.state[p]
		if st == nil {
			st = &ademamixState{
				m1:   make([]float32, len(d)),
				m2:   make([]float32, len(d)),
				v:    make([]float32, len(d)),
				pow1: 1,
				pow2: 1,
			}
			o.state[p] = st
		} else if len(st.m1) != len(d) {
			panic(fmt.Sprintf("optimizer.AdEMAMix.Step: parameter %p changed size: state has %d elements, Data has %d", p, len(st.m1), len(d)))
		}
		st.t++
		st.pow1 *= o.Beta1
		st.pow2 *= o.Beta2
		bc1 := 1 - st.pow1 // bias-correction denominators 1 - Beta^t
		bc2 := 1 - st.pow2
		one1 := 1 - o.Beta1
		one2 := 1 - o.Beta2
		alpha, beta3 := o.Alpha, o.Beta3
		if o.Warmup > 0 && st.t < o.Warmup {
			alpha = ademamixAlpha(st.t, o.Warmup, o.Alpha)
			beta3 = ademamixBeta3(st.t, o.Warmup, o.Beta1, o.Beta3)
		}
		one3 := 1 - beta3
		for i := range d {
			st.m1[i] = o.Beta1*st.m1[i] + one1*g[i]
			st.m2[i] = beta3*st.m2[i] + one3*g[i]
			st.v[i] = o.Beta2*st.v[i] + one2*g[i]*g[i]
			m1hat := st.m1[i] / bc1
			vhat := st.v[i] / bc2
			d[i] -= o.LR * (m1hat + alpha*st.m2[i]) / (float32(math.Sqrt(float64(vhat))) + o.Eps)
		}
	}
}
