package nn

import (
	"fmt"
	"math"
	"math/rand"

	"lnn/autograd"
	"lnn/tensor"
)

// LTC is a Liquid Time-Constant cell following Hasani et al. (2021) and the
// reference implementation in mlech26l/ncps. The membrane ODE
//
//	dv/dt = -(1/tau + f(v, I)) * v + f(v, I) * A
//
// is integrated with the semi-implicit Euler scheme of the reference:
//
//	num = cm/dt * v + gleak * vleak + sum_j(act_j * erev_j)
//	den = cm/dt     + gleak        + sum_j(act_j)
//	v  <- num / (den + eps)
//
// where each synapse activates as act = w * sigmoid(sigma * (v_pre - mu)).
// Positivity of cm, gleak and synaptic weights is enforced implicitly with a
// softplus (the reference's implicit_param_constraints mode), so no weight
// clipping hook is needed inside the optimizer.
type LTC struct {
	inDim, units int
	unfolds      int
	eps          float32

	gleak, vleak, cm *autograd.Variable // [units]
	mu, sigma, w     *autograd.Variable // [units, units] recurrent synapses
	sMu, sSigma, sW  *autograd.Variable // [inDim, units] sensory synapses
	inW, inB         *autograd.Variable // [inDim]
	outW, outB       *autograd.Variable // [units]

	// Reversal potentials are fixed random +/-1 constants drawn once at
	// construction time, mirroring the reference wiring. They are NOT
	// trainable and are deliberately absent from Parameters(): learning them
	// would flip synapses between excitatory and inhibitory polarity and
	// degrade the LTC into an ordinary plastic network.
	erev  *autograd.Variable // [units, units]
	sErev *autograd.Variable // [inDim, units]

	wiring *Wiring
}

// NewLTC creates an LTC cell. wiring may be nil, meaning FullyConnected;
// otherwise its sensory mask must have shape [inDim, units] and its
// recurrent mask [units, units]. unfolds is the number of ODE solver steps
// per RNN step (6 in the reference). Parameter init ranges follow the
// reference implementation.
func NewLTC(inDim, units int, wiring *Wiring, unfolds int, rng *rand.Rand) *LTC {
	if inDim < 1 || units < 1 {
		panic(fmt.Sprintf("nn.NewLTC: invalid dims in=%d units=%d", inDim, units))
	}
	if unfolds < 1 {
		panic(fmt.Sprintf("nn.NewLTC: unfolds must be >= 1, got %d", unfolds))
	}
	if wiring == nil {
		wiring = FullyConnected(inDim, units)
	}
	// Shape-only validation: comparing Shape fields allocates nothing, unlike
	// building reference tensors to diff against (which used to allocate an
	// inDim*units tensor just to say "no").
	if !shapeIs(wiring.sensoryMask.Shape, inDim, units) ||
		!shapeIs(wiring.recurrentMask.Shape, units, units) {
		panic(fmt.Sprintf("nn.NewLTC: wiring mask shapes %v and %v do not match [inDim=%d, units=%d]",
			wiring.sensoryMask.Shape, wiring.recurrentMask.Shape, inDim, units))
	}
	uniform := func(lo, hi float32, shape ...int) *autograd.Variable {
		return autograd.Var(tensor.Uniform(rng, lo, hi, shape...))
	}
	// Reversal potentials are random +/- 1, as in the reference wiring.
	erevInit := func(shape ...int) *autograd.Variable {
		t := tensor.New(shape...)
		for i := range t.Data {
			if rng.Intn(2) == 0 {
				t.Data[i] = -1
			} else {
				t.Data[i] = 1
			}
		}
		return autograd.Var(t)
	}
	return &LTC{
		inDim: inDim, units: units, unfolds: unfolds, eps: 1e-8,
		gleak:  uniform(0.001, 1, units),
		vleak:  uniform(-0.2, 0.2, units),
		cm:     uniform(0.4, 0.6, units),
		mu:     uniform(0.3, 0.8, units, units),
		sigma:  uniform(3, 8, units, units),
		w:      uniform(0.001, 1, units, units),
		sMu:    uniform(0.3, 0.8, inDim, units),
		sSigma: uniform(3, 8, inDim, units),
		sW:     uniform(0.001, 1, inDim, units),
		inW:    autograd.Var(tensor.New(inDim).OnesLike()),
		inB:    autograd.Var(tensor.New(inDim)),
		outW:   autograd.Var(tensor.New(units).OnesLike()),
		outB:   autograd.Var(tensor.New(units)),
		erev:   erevInit(units, units),
		sErev:  erevInit(inDim, units),
		wiring: wiring,
	}
}

// shapeIs reports whether sh equals want, without allocating.
func shapeIs(sh []int, want ...int) bool {
	if len(sh) != len(want) {
		return false
	}
	for i := range sh {
		if sh[i] != want[i] {
			return false
		}
	}
	return true
}

// StateSize returns the hidden state dimension.
func (c *LTC) StateSize() int { return c.units }

// Parameters returns the trainable variables of the cell. The reversal
// potentials erev/sErev are fixed constants and intentionally excluded.
func (c *LTC) Parameters() []*autograd.Variable {
	return []*autograd.Variable{
		c.gleak, c.vleak, c.cm,
		c.mu, c.sigma, c.w,
		c.sMu, c.sSigma, c.sW,
		c.inW, c.inB, c.outW, c.outB,
	}
}

// Step advances the cell by one RNN step, integrating the ODE over the time
// span ts (which must be positive and finite; NaN and +/-Inf are rejected).
// x is [batch, inDim], h is [batch, units] or nil for a zero initial state.
// It returns the (affinely mapped) output and the new raw state.
func (c *LTC) Step(x, h *autograd.Variable, ts float64) (out, hNew *autograd.Variable) {
	// NaN-aware positivity check: NaN > 0 is false, so NaN panics here too.
	// +Inf passes `ts > 0` but would silently integrate over an infinite time
	// span (the "infinite-time steady state"), which callers rarely intend;
	// reject both infinities explicitly, on the same panic path.
	if !(ts > 0) || math.IsInf(ts, 0) {
		panic(fmt.Sprintf("nn.LTC.Step: ts must be positive and finite, got %v", ts))
	}
	batch := x.Data.Rows()
	if h == nil {
		h = autograd.Var(tensor.New(batch, c.units))
	}

	// Affine input mapping.
	inputs := autograd.Add(autograd.Hadamard(x, c.inW), c.inB)

	// Positivity-constrained parameters (softplus).
	cmT := c.scaledCapacitance(ts)
	gleak := autograd.Softplus(c.gleak)
	wPos := autograd.Softplus(c.w)
	sWPos := autograd.Softplus(c.sW)

	// Sensory (input) synapses are loop-invariant over the ODE unfolds.
	numS, denS := c.synapses(inputs, c.sMu, c.sSigma, sWPos, c.sErev, c.wiring.SensoryRow)

	v := h
	epsV := autograd.Const(tensor.FromData([]float32{c.eps}, 1))
	for t := 0; t < c.unfolds; t++ {
		numR, denR := c.synapses(v, c.mu, c.sigma, wPos, c.erev, c.wiring.RecurrentRow)
		// num = cm_t .* v + gleak .* vleak + synapses
		num := autograd.Add(autograd.Add(autograd.Hadamard(cmT, v), autograd.Hadamard(gleak, c.vleak)), numS)
		num = autograd.Add(num, numR)
		den := autograd.Add(autograd.Add(cmT, gleak), denS)
		den = autograd.Add(den, denR)
		v = autograd.Div(num, autograd.Add(den, epsV))
	}

	out = autograd.Add(autograd.Hadamard(v, c.outW), c.outB)
	return out, v
}

// scaledCapacitance builds cm_t = softplus(cm) * unfolds/ts, the ODE's cm/dt
// term with dt = ts/unfolds. The scalar time scale is computed in float64
// and clamped to the finite float32 domain before being converted back: a
// tiny ts (e.g. 1e-40, the hallmark of the LTC's variable-step regime) used
// to overflow the float32 division to +Inf, and Inf*0 in the state update
// then turned every output NaN.
//
// Clamping the scale alone is not enough: softplus(cm)*scale can still
// exceed MaxFloat32 elementwise. We therefore cap the product with a smooth
// differentiable min, cap(sp) = sp - softplus(sp - hi), hi = MaxFloat32/scale.
// While sp << hi (every sane ts) softplus(sp - hi) underflows to exactly 0,
// so cap(sp) is bit-identical to sp and the ODE algebra is untouched; the
// cap only engages where the unscaled product would overflow. The tiny
// relative headroom on hi absorbs float32 rounding so cm_t is always finite.
func (c *LTC) scaledCapacitance(ts float64) *autograd.Variable {
	scale64 := float64(c.unfolds) / ts
	if scale64 > math.MaxFloat32 {
		scale64 = math.MaxFloat32
	}
	hi64 := math.MaxFloat32 / scale64 / 1.0001
	if hi64 > math.MaxFloat32 {
		hi64 = math.MaxFloat32
	}
	hiV := autograd.Const(tensor.FromData([]float32{float32(hi64)}, 1))
	sp := autograd.Softplus(c.cm)
	capped := autograd.Sub(sp, autograd.Softplus(autograd.Sub(sp, hiV)))
	return autograd.Scale(capped, float32(scale64))
}

// synapses accumulates numerator and denominator synaptic currents from a
// presynaptic source (inputs or previous state). The four parameter matrices
// are passed whole; row i, extracted inside the loop with SliceRow,
// parameterizes the synapses of presynaptic neuron i, whose wiring mask row
// comes from maskRow(i). The sensory and recurrent paths share this single
// calling convention.
func (c *LTC) synapses(
	pre, mu, sigma, w, erev *autograd.Variable,
	maskRow func(i int) *tensor.Tensor,
) (num, den *autograd.Variable) {
	n := pre.Data.Cols()
	for i := 0; i < n; i++ {
		muR := autograd.SliceRow(mu, i)
		sigR := autograd.SliceRow(sigma, i)
		wR := autograd.SliceRow(w, i)
		erevR := autograd.SliceRow(erev, i)
		preCol := autograd.Col(pre, i) // [batch, 1]
		act := autograd.Sigmoid(autograd.Hadamard(sigR, autograd.Sub(preCol, muR)))
		act = autograd.Hadamard(act, wR)
		act = autograd.Hadamard(act, autograd.Const(maskRow(i)))
		rev := autograd.Hadamard(act, erevR)
		if i == 0 {
			num, den = rev, act
		} else {
			num = autograd.Add(num, rev)
			den = autograd.Add(den, act)
		}
	}
	return num, den
}
