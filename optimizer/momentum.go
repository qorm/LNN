package optimizer

import (
	"fmt"

	"github.com/qorm/LNN/autograd"
)

// Momentum is classical heavy-ball SGD with one velocity buffer per
// parameter — exactly the update documented in doc/training.md
// ("Optional: momentum"):
//
//	v = Mu*v + g // velocity stores UNSCALED gradients
//	p -= LR * v
//
// Mu is the damping factor; Mu = 0 reduces the rule to plain SGD with
// the same LR. Because the velocity accumulates unscaled gradients, if
// you combine Momentum with global norm clipping, apply the same scale
// to the gradient before it enters the velocity, or clip the velocity
// itself — pick one and be consistent, as doc/training.md advises.
type Momentum struct {
	// LR is the learning rate, Mu the damping factor. NewMomentum
	// requires LR > 0 and Mu in [0, 1); Step uses the fields as written.
	LR, Mu float32

	// velocity maps each parameter pointer to its velocity buffer, a
	// flat []float32 shaped like the parameter's Data. The first Step
	// that sees a parameter allocates a zero buffer; the map pins every
	// parameter it has seen, so use a fresh optimizer per model.
	velocity map[*autograd.Variable][]float32
}

// NewMomentum returns a heavy-ball Momentum optimizer. It panics unless
// lr > 0 and 0 <= mu < 1 (NaN fails the checks and panics too).
func NewMomentum(lr, mu float32) *Momentum {
	if !(lr > 0) {
		panic(fmt.Sprintf("optimizer.NewMomentum: learning rate must be positive, got %v", lr))
	}
	if !(mu >= 0 && mu < 1) {
		panic(fmt.Sprintf("optimizer.NewMomentum: mu must be in [0, 1), got %v", mu))
	}
	return &Momentum{LR: lr, Mu: mu, velocity: make(map[*autograd.Variable][]float32)}
}

// Step updates in place every parameter with a non-nil Grad, carrying
// its velocity buffer across calls. A parameter whose Data is resized
// between steps panics: its stored velocity no longer matches. See
// Optimizer for the nil-Grad and ZeroGrad contract.
func (o *Momentum) Step(params []*autograd.Variable) {
	for _, p := range params {
		if p.Grad == nil {
			continue
		}
		d, g := p.Data.Data, p.Grad.Data
		v := o.velocity[p]
		if v == nil {
			v = make([]float32, len(d))
			o.velocity[p] = v
		} else if len(v) != len(d) {
			panic(fmt.Sprintf("optimizer.Momentum.Step: parameter %p changed size: velocity has %d elements, Data has %d", p, len(v), len(d)))
		}
		for i := range d {
			v[i] = o.Mu*v[i] + g[i]
			d[i] -= o.LR * v[i]
		}
	}
}
