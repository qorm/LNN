package nn

import "github.com/qorm/LNN/autograd"

// Cell is a recurrent cell that can be stepped over time.
//
// Step advances the cell by one RNN step spanning ts time units: x is
// the [batch, inDim] input and h the [batch, StateSize()] state (nil
// means a zero state). It returns the step output [batch, StateSize()]
// and the new raw state. ts must be positive and finite: the provided
// cells (LTC, CfC) panic on NaN, +/-Inf, zero and negative ts, and a
// wrong x/h rank or width panics in the tensor layer.
//
// StateSize returns the hidden state dimension (the units count), the
// column count of both the state and the step output.
type Cell interface {
	Step(x, h *autograd.Variable, ts float64) (out, hNew *autograd.Variable)
	StateSize() int
}

// LTC satisfies the Cell interface.
var _ Cell = (*LTC)(nil)

// Unroll drives cell over the input sequence xs (one Step per element,
// all at time span ts), threading the hidden state through. h0 may be
// nil for a zero initial state. It returns the per-step outputs (one
// [batch, StateSize()] variable per input) and the final state; the
// whole sequence stays in the graph, so a loss built on ys
// differentiates through time with a single Backward. For an empty xs,
// ys is an empty (non-nil) slice and hN is h0 unchanged (callers must
// handle a nil final state). Every xs element must be non-nil with the
// cell's input width, and ts is validated by each Step (panic contract
// above).
func Unroll(cell Cell, xs []*autograd.Variable, h0 *autograd.Variable, ts float64) (ys []*autograd.Variable, hN *autograd.Variable) {
	h := h0
	ys = make([]*autograd.Variable, len(xs))
	for i, x := range xs {
		var y *autograd.Variable
		y, h = cell.Step(x, h, ts)
		ys[i] = y
	}
	return ys, h
}
