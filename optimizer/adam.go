package optimizer

import (
	"fmt"
	"math"

	"lnn/autograd"
)

// Adam implements the Adam update of Kingma & Ba, "Adam: A Method for
// Stochastic Optimization" (arXiv:1412.6980, 2014), with bias-corrected
// moment estimates, applied per element of each parameter:
//
//	t  += 1
//	m   = Beta1*m + (1-Beta1)*g
//	v   = Beta2*v + (1-Beta2)*g*g
//	m'  = m / (1 - Beta1^t)
//	v'  = v / (1 - Beta2^t)
//	p  -= LR * m' / (sqrt(v') + Eps)
//
// All state is float32, per the library's float32-everywhere
// convention. That is safe here because the update is
// self-normalizing — |m'|/sqrt(v') stays bounded near 1 regardless of
// gradient scale — so no wide-magnitude sum is ever formed, unlike the
// global gradient norm in doc/training.md's clipping section, which
// accumulates in float64 for exactly that reason. The square root goes
// through math.Sqrt: the standard library has no float32 sqrt, and a
// single correctly-rounded conversion per element is not an
// accumulation, so it cannot lose precision the way a long float32
// sum would.
type Adam struct {
	// LR is the learning rate, Beta1/Beta2 the moment decay rates, Eps
	// the divisor guard. NewAdam requires LR > 0, Beta1 and Beta2 in
	// [0, 1), Eps > 0; Step uses the fields as written.
	LR, Beta1, Beta2, Eps float32

	// state maps each parameter pointer to its moment buffers and
	// update count. The map pins every parameter it has seen, so use a
	// fresh optimizer per model.
	state map[*autograd.Variable]*adamState
}

// adamState holds one parameter's Adam state. t counts the updates
// applied to this parameter (parameters skipped because Grad was nil do
// not advance it, so their bias correction stays exact); pow1 and pow2
// are Beta1^t and Beta2^t kept as running float32 products, which keeps
// the bias correction O(1) per step without leaving float32.
type adamState struct {
	m, v       []float32
	t          int
	pow1, pow2 float32
}

// NewAdam returns an Adam optimizer. It panics unless lr > 0,
// 0 <= beta1 < 1, 0 <= beta2 < 1 and eps > 0 (NaN fails the checks and
// panics too).
func NewAdam(lr, beta1, beta2, eps float32) *Adam {
	if !(lr > 0) {
		panic(fmt.Sprintf("optimizer.NewAdam: learning rate must be positive, got %v", lr))
	}
	if !(beta1 >= 0 && beta1 < 1) {
		panic(fmt.Sprintf("optimizer.NewAdam: beta1 must be in [0, 1), got %v", beta1))
	}
	if !(beta2 >= 0 && beta2 < 1) {
		panic(fmt.Sprintf("optimizer.NewAdam: beta2 must be in [0, 1), got %v", beta2))
	}
	if !(eps > 0) {
		panic(fmt.Sprintf("optimizer.NewAdam: eps must be positive, got %v", eps))
	}
	return &Adam{LR: lr, Beta1: beta1, Beta2: beta2, Eps: eps, state: make(map[*autograd.Variable]*adamState)}
}

// NewAdamDefault returns Adam with the hyperparameters recommended by
// Kingma & Ba: Beta1 = 0.9, Beta2 = 0.999, Eps = 1e-8.
func NewAdamDefault(lr float32) *Adam { return NewAdam(lr, 0.9, 0.999, 1e-8) }

// Step updates in place every parameter with a non-nil Grad, carrying
// its moment buffers and update count across calls. A parameter whose
// Data is resized between steps panics: its stored moments no longer
// match. See Optimizer for the nil-Grad and ZeroGrad contract.
func (o *Adam) Step(params []*autograd.Variable) {
	for _, p := range params {
		if p.Grad == nil {
			continue
		}
		d, g := p.Data.Data, p.Grad.Data
		st := o.state[p]
		if st == nil {
			st = &adamState{
				m:    make([]float32, len(d)),
				v:    make([]float32, len(d)),
				pow1: 1,
				pow2: 1,
			}
			o.state[p] = st
		} else if len(st.m) != len(d) {
			panic(fmt.Sprintf("optimizer.Adam.Step: parameter %p changed size: state has %d elements, Data has %d", p, len(st.m), len(d)))
		}
		st.t++
		st.pow1 *= o.Beta1
		st.pow2 *= o.Beta2
		bc1 := 1 - st.pow1 // bias-correction denominators 1 - Beta^t
		bc2 := 1 - st.pow2
		one1 := 1 - o.Beta1
		one2 := 1 - o.Beta2
		for i := range d {
			st.m[i] = o.Beta1*st.m[i] + one1*g[i]
			st.v[i] = o.Beta2*st.v[i] + one2*g[i]*g[i]
			mhat := st.m[i] / bc1
			vhat := st.v[i] / bc2
			d[i] -= o.LR * mhat / (float32(math.Sqrt(float64(vhat))) + o.Eps)
		}
	}
}
