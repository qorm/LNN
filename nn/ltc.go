package nn

import (
	"fmt"
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
	mu, sigma        *autograd.Variable // [units, units]
	w, erev          *autograd.Variable // [units, units]
	sMu, sSigma      *autograd.Variable // [inDim, units]
	sW, sErev        *autograd.Variable // [inDim, units]
	inW, inB         *autograd.Variable // [inDim]
	outW, outB       *autograd.Variable // [units]

	wiring *Wiring
}

// NewLTC creates an LTC cell. wiring may be nil, meaning FullyConnected.
// unfolds is the number of ODE solver steps per RNN step (6 in the
// reference). Parameter init ranges follow the reference implementation.
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
	if !tensor.SameShape(wiring.SensoryMask, tensor.New(inDim, units)) ||
		!tensor.SameShape(wiring.RecurrentMask, tensor.New(units, units)) {
		panic("nn.NewLTC: wiring mask shapes do not match (inDim, units)")
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
		gleak: uniform(0.001, 1, units),
		vleak: uniform(-0.2, 0.2, units),
		cm:    uniform(0.4, 0.6, units),
		mu:    uniform(0.3, 0.8, units, units),
		sigma: uniform(3, 8, units, units),
		w:     uniform(0.001, 1, units, units),
		erev:  erevInit(units, units),
		sMu:   uniform(0.3, 0.8, inDim, units),
		sSigma: uniform(3, 8, inDim, units),
		sW:    uniform(0.001, 1, inDim, units),
		sErev: erevInit(inDim, units),
		inW:   autograd.Var(tensor.New(inDim).OnesLike()),
		inB:   autograd.Var(tensor.New(inDim)),
		outW:  autograd.Var(tensor.New(units).OnesLike()),
		outB:  autograd.Var(tensor.New(units)),
		wiring: wiring,
	}
}

// StateSize returns the hidden state dimension.
func (c *LTC) StateSize() int { return c.units }

// Parameters returns the trainable variables of the cell.
func (c *LTC) Parameters() []*autograd.Variable {
	return []*autograd.Variable{
		c.gleak, c.vleak, c.cm,
		c.mu, c.sigma, c.w, c.erev,
		c.sMu, c.sSigma, c.sW, c.sErev,
		c.inW, c.inB, c.outW, c.outB,
	}
}

// Step advances the cell by one RNN step, integrating the ODE over the time
// span ts (which must be > 0). x is [batch, inDim], h is [batch, units] or
// nil for a zero initial state. It returns the (affinely mapped) output and
// the new raw state.
func (c *LTC) Step(x, h *autograd.Variable, ts float32) (out, hNew *autograd.Variable) {
	if ts <= 0 {
		panic(fmt.Sprintf("nn.LTC.Step: ts must be > 0, got %v", ts))
	}
	batch := x.Data.Rows()
	if h == nil {
		h = autograd.Var(tensor.New(batch, c.units))
	}

	// Affine input mapping.
	inputs := autograd.Add(autograd.Hadamard(x, c.inW), c.inB)

	// Positivity-constrained parameters (softplus).
	cmT := autograd.Scale(autograd.Softplus(c.cm), float32(c.unfolds)/ts)
	gleak := autograd.Softplus(c.gleak)
	wPos := autograd.Softplus(c.w)
	sWPos := autograd.Softplus(c.sW)

	// Sensory (input) synapses are loop-invariant over the ODE unfolds.
	numS, denS := c.synapses(inputs, c.sMu, c.sSigma, sWPos, c.sErev, c.wiring.SensoryRow)

	// Hoist presynaptic parameter rows out of the unfold loop.
	muRows := make([]*autograd.Variable, c.units)
	sigRows := make([]*autograd.Variable, c.units)
	wRows := make([]*autograd.Variable, c.units)
	erevRows := make([]*autograd.Variable, c.units)
	for i := 0; i < c.units; i++ {
		muRows[i] = autograd.SliceRow(c.mu, i)
		sigRows[i] = autograd.SliceRow(c.sigma, i)
		wRows[i] = autograd.SliceRow(wPos, i)
		erevRows[i] = autograd.SliceRow(c.erev, i)
	}

	v := h
	epsV := autograd.Const(tensor.FromData([]float32{c.eps}, 1))
	for t := 0; t < c.unfolds; t++ {
		numR, denR := c.synapses(v, muRows, sigRows, wRows, erevRows, c.wiring.RecurrentRow)
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

// synapses accumulates numerator and denominator synaptic currents from a
// presynaptic source (inputs or previous state). Param rows are extracted
// per source neuron i via SliceRow; maskRow yields the wiring mask rows.
func (c *LTC) synapses(
	pre *autograd.Variable,
	muRows, sigRows, wRows, erevRows []*autograd.Variable,
	maskRow func(i int) *tensor.Tensor,
) (num, den *autograd.Variable) {
	n := pre.Data.Cols()
	for i := 0; i < n; i++ {
		muR, sigR, wR, erevR := muRows[i], sigRows[i], wRows[i], erevRows[i]
		if muR.Data.Dims() == 2 && muR.Data.Shape[0] != 1 {
			// Params given as full matrices: slice row i.
			muR = autograd.SliceRow(muRows[i], i)
			sigR = autograd.SliceRow(sigRows[i], i)
			wR = autograd.SliceRow(wRows[i], i)
			erevR = autograd.SliceRow(erevRows[i], i)
		}
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
