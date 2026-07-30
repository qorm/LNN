package optimizer

import (
	"fmt"

	"github.com/qorm/LNN/autograd"
)

// SGD is plain gradient descent: p -= LR * p.Grad for every element of
// every parameter. It is exactly the hand-rolled update of
// doc/training.md ("The loop", phase 4) packaged as a Step method:
//
//	for i := range p.Data.Data {
//		p.Data.Data[i] -= lr * p.Grad.Data[i]
//	}
//
// There is no built-in stabilization: use modest learning rates and
// clip the global gradient norm on larger problems, as doc/training.md
// recommends.
type SGD struct {
	// LR is the learning rate. Must be positive; NewSGD validates, Step
	// uses the field as written.
	LR float32
}

// NewSGD returns a plain SGD optimizer. It panics unless lr > 0 (NaN
// fails the check and panics too). The constructor check is a convenience,
// not a full invariant: literal +Inf satisfies lr > 0 and is accepted,
// after which every Step produces ±Inf updates — or NaN where a gradient
// element is exactly zero, since Inf*0 is NaN. Step likewise trusts the
// LR field as written, so writing an invalid value into SGD.LR after
// construction yields exactly the arithmetic asked for. This is the same
// trust model as a hand-rolled loop trusting its lr constant: validate
// once at the boundary, then run without re-checking.
func NewSGD(lr float32) *SGD {
	if !(lr > 0) {
		panic(fmt.Sprintf("optimizer.NewSGD: learning rate must be positive, got %v", lr))
	}
	return &SGD{LR: lr}
}

// Step updates in place every parameter with a non-nil Grad. See
// Optimizer for the nil-Grad and ZeroGrad contract.
func (o *SGD) Step(params []*autograd.Variable) {
	for _, p := range params {
		if p.Grad == nil {
			continue
		}
		d, g := p.Data.Data, p.Grad.Data
		for i := range d {
			d[i] -= o.LR * g[i]
		}
	}
}
