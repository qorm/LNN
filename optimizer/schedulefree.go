package optimizer

import (
	"fmt"
	"math"

	"github.com/qorm/LNN/autograd"
)

// ScheduleFreeAdamW implements Schedule-Free AdamW of Defazio, Yang,
// Mehta, Mishchenko, Khaled & Cutkosky, "The Road Less Scheduled"
// (arXiv:2405.15682, NeurIPS 2024 oral; winner of the MLCommons 2024
// AlgoPerf self-tuning track): AdamW's learning-rate schedule replaced
// by iterate averaging, so no stopping time T and no decay schedule
// are ever needed. Three sequences run per parameter (the paper's
// equations 3-5, with AdamW as the base update, exactly as the
// official facebookresearch/schedule_free implementation):
//
//	y = (1-Beta1)*z + Beta1*x      // gradients are evaluated at y
//	v = Beta2*v + (1-Beta2)*g*g    // g = grad at y, second moment
//	g' = g / (sqrt(v/(1-Beta2^t)) + Eps)  (+ WeightDecay*y)
//	z -= lr * g'                   // base AdamW step on the z sequence
//	x = (1-c)*x + c*z              // x: weighted average of z, c = w_t/Sum(w)
//
// y is the gradient-evaluation point, z the fast base sequence, and x
// — a polynomially weighted average of z — the evaluation/export
// sequence: the weights a caller should actually deploy or measure.
// With a constant LR the average weights are uniform and c = 1/(k+1)
// at step k (0-based); with WarmupSteps > 0 the learning rate ramps
// linearly, lr_t = LR*(k+1)/WarmupSteps, and each step's average
// weight is lr_max^2 where lr_max is the largest lr seen so far (the
// official r = 0, weight_lr_power = 2 form), which keeps c = 1/(k+1)
// once the warmup is done.
//
// The bias correction follows the CURRENT official implementation
// (schedulefree v1.3+): v is divided by 1-Beta2^t inside the square
// root, as above. The repo also ships the paper-published variant
// (AdamWScheduleFreePaper), which folds sqrt(1-Beta2^t) into the
// learning rate instead and is documented upstream as less stable;
// this implementation does not reproduce that variant.
//
// # The y/x contract — read this before using Step
//
// Gradients must be evaluated at y, but the deployable weights are x.
// Like the official implementation, this optimizer keeps the
// PARAMETER'S Data holding y during training and converts in place:
//
//	opt := optimizer.NewScheduleFreeAdamWDefault(0.01)
//	for ... {              // fresh optimizer starts in train mode
//		// params hold y: build the graph, backward — nothing special
//		opt.Step(params) // updates z and x in place; params keep holding y
//	}
//	opt.Eval(params)  // params now hold x: evaluate or export here
//	opt.Train(params) // params back to y before the next Step
//
// Every stepped parameter carries its OWN mode bit in its optimizer
// state, and Step checks it per parameter: stepping a parameter whose
// Data holds x panics naming that parameter's index — training at x
// instead of y is the one misuse that silently ruins the method's
// guarantees (it is what the official implementation raises on), so it
// is caught cheaply rather than documented away. The per-parameter
// bits also close the subset-desynchronization trap: Eval(all)
// followed by Train(subset) leaves the unconverted parameters in eval
// mode, and the next Step panics on the first of them instead of
// silently training it at x. Eval and Train are idempotent and only
// touch parameters that already carry optimizer state; a parameter
// never stepped has x = y = z and needs no conversion. Parameters
// without state are gated by the optimizer-level mode flag instead
// (fresh optimizers start in train mode): Eval-then-Step on a
// never-stepped optimizer panics until Train, matching the official
// contract gate. Unlike the official implementation, a fresh optimizer
// starts in train mode: at construction x = y = z, so the officially
// mandatory initial train() call converts nothing.
//
// Checkpointing: SaveState/LoadState persist z, v, the counters and
// each parameter's mode bit; the parameter values ride the model
// streams as usual. Checkpoint in TRAIN mode for a bit-exact training
// resume (params hold y, z rides the state stream); checkpoint in EVAL
// mode to export x — the eval weights — for inference, OR to pause
// training: after the load every restored parameter is back in eval
// mode, Step panics, and Train performs the x->y conversion before
// training resumes — bit-identical to the same-instance Eval -> Train
// -> resume path (the resume tests pin this). The y->x->y conversion
// round trip is not bit-exact (each conversion rounds), so a resume
// through an eval-mode checkpoint continues from a y a few ulps off
// the uninterrupted one. The optimizer-level mode flag alone is
// transient and never serialized; a fresh optimizer is always in train
// mode.
//
// # Hyperparameters
//
// Beta1 is the momentum interpolation between z and x (0.9 default;
// the official guidance notes training is more sensitive to Beta1 than
// EMA-momentum users expect, and that 0.95-0.98 can be necessary for
// very long runs — the paper's NanoGPT experiment used 0.98). Beta1 = 0
// (Polyak-Ruppert averaging, y = z) is REJECTED by the constructor:
// the in-place y/x conversion needs 1/Beta1. Beta2 and Eps are Adam's
// usual second-moment decay and guard. WeightDecay is decoupled,
// applied at y as in the official implementation (0 default). LR wants
// values 1x-10x larger than schedule-based AdamW (official guidance);
// the official default is 0.0025. Do not layer a learning-rate decay
// schedule on top — replacing it is the point — but writing the
// exported fields mid-training is, as for every optimizer here, used
// as written.
type ScheduleFreeAdamW struct {
	// LR is the learning rate, Beta1 the y-interpolation momentum,
	// Beta2 the second-moment decay, Eps the divisor guard,
	// WeightDecay the decoupled decay coefficient (applied at y),
	// WarmupSteps the linear learning-rate warmup length (0 = none).
	// NewScheduleFreeAdamW requires LR > 0, Beta1 in (0, 1), Beta2 in
	// [0, 1) and Eps > 0; Step uses the fields as written.
	LR, Beta1, Beta2, Eps, WeightDecay float32
	WarmupSteps                        int

	// trainMode is the optimizer-level mode, gating parameters that carry
	// no state yet (never stepped): Eval-then-Step on a fresh optimizer
	// panics until Train, the official contract gate. Parameters WITH
	// state are gated by their own per-state mode bit instead. Fresh
	// optimizers start in train mode; Eval/Train flip it. It is
	// deliberately not part of the serialized state.
	trainMode bool

	// state maps each parameter pointer to its z/v buffers and
	// counters. The map pins every parameter it has seen, so use a
	// fresh optimizer per model.
	state map[*autograd.Variable]*scheduleFreeState
}

// scheduleFreeState holds one parameter's Schedule-Free state. z is
// the base sequence (x is never materialized: the parameter's Data
// holds y, and x is recovered from y and z by Eval); v is the second
// moment. k counts the updates applied to this parameter (parameters
// skipped because Grad was nil do not advance it); pow2 is Beta2^k as
// a running float32 product, the same bias-correction discipline as
// Adam. weightSum accumulates the float64 average weights (a scalar
// sum over the whole run, so it does not go through float32 the way
// the elementwise buffers do) and lrMax tracks the largest scheduled
// learning rate seen, for the average weight w_k = lrMax^2. trainMode
// is the parameter's own mode bit: true while its Data holds y, false
// while it holds x — Step checks it per parameter, so a subset Train
// cannot desynchronize the unconverted parameters into a silent
// train-at-x. It is serialized with the rest of the state.
type scheduleFreeState struct {
	z, v      []float32
	k         int
	pow2      float32
	weightSum float64
	lrMax     float32
	trainMode bool
}

// NewScheduleFreeAdamW returns a Schedule-Free AdamW optimizer. It
// panics unless lr > 0, 0 < beta1 < 1, 0 <= beta2 < 1 and eps > 0
// (NaN fails the checks and panics too). beta1 = 0 is refused — not
// because the paper excludes it (there it is Polyak-Ruppert averaging)
// but because the in-place y/x conversion this implementation uses
// divides by beta1; the official implementation fails there too, only
// later and less clearly.
func NewScheduleFreeAdamW(lr, beta1, beta2, eps float32) *ScheduleFreeAdamW {
	if !(lr > 0) {
		panic(fmt.Sprintf("optimizer.NewScheduleFreeAdamW: learning rate must be positive, got %v", lr))
	}
	if !(beta1 > 0 && beta1 < 1) {
		panic(fmt.Sprintf("optimizer.NewScheduleFreeAdamW: beta1 must be in (0, 1), got %v", beta1))
	}
	if !(beta2 >= 0 && beta2 < 1) {
		panic(fmt.Sprintf("optimizer.NewScheduleFreeAdamW: beta2 must be in [0, 1), got %v", beta2))
	}
	if !(eps > 0) {
		panic(fmt.Sprintf("optimizer.NewScheduleFreeAdamW: eps must be positive, got %v", eps))
	}
	return &ScheduleFreeAdamW{
		LR: lr, Beta1: beta1, Beta2: beta2, Eps: eps,
		trainMode: true,
		state:     make(map[*autograd.Variable]*scheduleFreeState),
	}
}

// NewScheduleFreeAdamWDefault returns Schedule-Free AdamW with the
// official defaults: Beta1 = 0.9, Beta2 = 0.999, Eps = 1e-8. The
// official implementation's default learning rate is 0.0025, and its
// guidance is 1x-10x the learning rate of schedule-based AdamW.
func NewScheduleFreeAdamWDefault(lr float32) *ScheduleFreeAdamW {
	return NewScheduleFreeAdamW(lr, 0.9, 0.999, 1e-8)
}

// Train switches the given parameters from the evaluation sequence x
// back to the gradient-evaluation sequence y, converting in place:
// p += (1-Beta1)*(z - p), i.e. y = Beta1*x + (1-Beta1)*z. It is a
// per-parameter no-op for parameters already in train mode and for
// parameters that carry no optimizer state. Call it after Eval before
// the next Step; a fresh optimizer is already in train mode. It panics
// on a nil parameter or one without Data (a programmer error, not
// untrusted input): the guard is unconditional.
func (o *ScheduleFreeAdamW) Train(params []*autograd.Variable) {
	for i, p := range params {
		if p == nil || p.Data == nil {
			panic(fmt.Sprintf("optimizer.ScheduleFreeAdamW.Train: parameter %d has no data", i))
		}
	}
	w := 1 - o.Beta1
	for _, p := range params {
		st := o.state[p]
		if st == nil || st.trainMode {
			continue // never stepped (x = y = z) or already at y: nothing to convert
		}
		d, z := p.Data.Data, st.z
		if len(z) != len(d) {
			panic(fmt.Sprintf("optimizer.ScheduleFreeAdamW.Train: parameter %p changed size: state has %d elements, Data has %d", p, len(z), len(d)))
		}
		for j := range d {
			d[j] += w * (z[j] - d[j])
		}
		st.trainMode = true
	}
	o.trainMode = true
}

// Eval switches the given parameters from the gradient-evaluation
// sequence y to the evaluation sequence x, converting in place:
// p += (1-1/Beta1)*(z - p), the exact inverse of Train's conversion.
// Evaluate, benchmark and export (nn.Save*/serialize) the parameters
// in this mode — x is the averaged weight sequence the method exists
// to produce. It is a per-parameter no-op for parameters already in
// eval mode and for parameters that carry no optimizer state. Call
// Train before resuming Step. It panics on a nil parameter or one
// without Data: the guard is unconditional.
func (o *ScheduleFreeAdamW) Eval(params []*autograd.Variable) {
	for i, p := range params {
		if p == nil || p.Data == nil {
			panic(fmt.Sprintf("optimizer.ScheduleFreeAdamW.Eval: parameter %d has no data", i))
		}
	}
	w := 1 - 1/o.Beta1
	for _, p := range params {
		st := o.state[p]
		if st == nil || !st.trainMode {
			continue // never stepped (x = y = z) or already at x: nothing to convert
		}
		d, z := p.Data.Data, st.z
		if len(z) != len(d) {
			panic(fmt.Sprintf("optimizer.ScheduleFreeAdamW.Eval: parameter %p changed size: state has %d elements, Data has %d", p, len(z), len(d)))
		}
		for j := range d {
			d[j] += w * (z[j] - d[j])
		}
		st.trainMode = false
	}
	o.trainMode = false
}

// Step applies one Schedule-Free AdamW update to every parameter with
// a non-nil Grad, keeping each parameter's Data holding y (the next
// forward pass therefore evaluates at the fresh y with no caller
// action). It panics on any parameter that is in eval mode — after
// Eval and before Train — naming the parameter's index in the slice:
// a gradient computed at x fed into the y-contract update is the
// silent failure this API exists to prevent, and the per-parameter
// check also catches a subset Train that left some parameters
// unconverted. The mode gate runs as an atomic pre-pass: the panic
// fires before ANY parameter is touched, so recover + Train + Step
// cannot double-step the parameters ahead of the violating one. A
// parameter whose Data is resized between steps panics: its stored z
// no longer matches. See Optimizer for the nil-Grad and ZeroGrad
// contract.
func (o *ScheduleFreeAdamW) Step(params []*autograd.Variable) {
	for i, p := range params {
		if p.Grad == nil {
			continue
		}
		st := o.state[p]
		if st == nil {
			// No per-parameter state yet: the optimizer-level mode
			// gates (a fresh optimizer starts in train mode; Eval
			// before the first Step locks Step until Train).
			if !o.trainMode {
				panic(fmt.Sprintf("optimizer.ScheduleFreeAdamW.Step: optimizer is in eval mode and parameter %d carries no state: gradients must be evaluated at y, not x; call Train(params) before resuming training", i))
			}
		} else if !st.trainMode {
			panic(fmt.Sprintf("optimizer.ScheduleFreeAdamW.Step: parameter %d is in eval mode (its Data holds the evaluation weights x): gradients must be evaluated at y; call Train(params) before resuming training", i))
		}
	}
	for _, p := range params {
		if p.Grad == nil {
			continue
		}
		d, g := p.Data.Data, p.Grad.Data
		st := o.state[p]
		if st == nil {
			st = &scheduleFreeState{
				z:         append([]float32(nil), d...), // z starts at the initial point
				v:         make([]float32, len(d)),
				pow2:      1,
				lrMax:     -1,
				trainMode: true,
			}
			o.state[p] = st
		} else if len(st.z) != len(d) {
			panic(fmt.Sprintf("optimizer.ScheduleFreeAdamW.Step: parameter %p changed size: state has %d elements, Data has %d", p, len(st.z), len(d)))
		}
		st.pow2 *= o.Beta2
		bc2 := 1 - st.pow2 // bias-correction denominator 1 - Beta2^(k+1)
		one2 := 1 - o.Beta2
		// Linear warmup on the learning rate only; the average weight
		// uses the running maximum so warmup steps do not down-weight
		// themselves in x (the official weight_lr_power = 2 form).
		lr := o.LR
		if st.k < o.WarmupSteps {
			lr *= float32(st.k+1) / float32(o.WarmupSteps)
		}
		if lr > st.lrMax {
			st.lrMax = lr
		}
		weight := float64(st.lrMax) * float64(st.lrMax)
		st.weightSum += weight
		c := float32(weight / st.weightSum) // c_{k+1} = w_k / Sum(w_0..k)
		lrc := lr * (o.Beta1*(1-c) - 1)
		for i := range d {
			st.v[i] = o.Beta2*st.v[i] + one2*g[i]*g[i]
			vhat := st.v[i] / bc2
			gn := g[i] / (float32(math.Sqrt(float64(vhat))) + o.Eps)
			if o.WeightDecay != 0 {
				gn += o.WeightDecay * d[i] // decoupled weight decay, at y
			}
			zi := st.z[i]
			yi := d[i] + c*(zi-d[i]) // y <- (1-c)*y + c*z
			d[i] = yi + lrc*gn
			st.z[i] = zi - lr*gn
		}
		st.k++
	}
}
