package nn

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/tensor"
)

// These red-team cases run UnrollRemat against CUSTOM Cell
// implementations — not just the library's LTC/CfC — to pin the claim
// that the fold classification generalizes: any cell whose per-step graph
// keeps each leaf's consumers at one structural position per step must be
// rematerialized bit-exactly.

// rnnCell is a vanilla RNN: hNew = tanh(hW + xU + b), out = hNew. All
// parameters sit inside the state subgraph (state-rest class): the
// simplest possible fold structure.
type rnnCell struct {
	w, u, b *autograd.Variable
	units   int
}

func newRNNCell(inDim, units int, rng *rand.Rand) *rnnCell {
	return &rnnCell{
		w:     autograd.Var(tensor.Uniform(rng, -0.4, 0.4, units, units)),
		u:     autograd.Var(tensor.Uniform(rng, -0.4, 0.4, inDim, units)),
		b:     autograd.Var(tensor.New(units)),
		units: units,
	}
}

func (c *rnnCell) StateSize() int { return c.units }

func (c *rnnCell) Step(x, h *autograd.Variable, ts float64) (out, hNew *autograd.Variable) {
	if h == nil {
		h = autograd.Var(tensor.New(x.Data.Rows(), c.units))
	}
	v := autograd.Tanh(autograd.Add(autograd.Add(autograd.MatMul(h, c.w), autograd.MatMul(x, c.u)), c.b))
	return v, v
}

func (c *rnnCell) Parameters() []*autograd.Variable { return []*autograd.Variable{c.w, c.u, c.b} }

// spineRNNCell deliberately hangs a parameter on the DFS descent: the
// gate g enters as Hadamard(g, h) with g FIRST, so the build pass appends
// g's chain before descending the state — the same spine position the
// LTC's cm occupies, reached by a cell the library never heard of. g is
// used ONLY there: the bitwise guarantee requires each trainable leaf to
// have consumers in exactly one fold class (the LTC and CfC satisfy it;
// see UnrollRemat's doc comment).
type spineRNNCell struct {
	g, w, u *autograd.Variable
	units   int
}

func newSpineRNNCell(inDim, units int, rng *rand.Rand) *spineRNNCell {
	return &spineRNNCell{
		g:     autograd.Var(tensor.Uniform(rng, 0.5, 1.5, units)),
		w:     autograd.Var(tensor.Uniform(rng, -0.4, 0.4, units, units)),
		u:     autograd.Var(tensor.Uniform(rng, -0.4, 0.4, inDim, units)),
		units: units,
	}
}

func (c *spineRNNCell) StateSize() int { return c.units }

func (c *spineRNNCell) Step(x, h *autograd.Variable, ts float64) (out, hNew *autograd.Variable) {
	if h == nil {
		h = autograd.Var(tensor.New(x.Data.Rows(), c.units))
	}
	gated := autograd.Hadamard(c.g, h) // g first: spine position
	v := autograd.Tanh(autograd.Add(autograd.MatMul(gated, c.w), autograd.MatMul(x, c.u)))
	out = autograd.Scale(v, 2)
	return out, v
}

func (c *spineRNNCell) Parameters() []*autograd.Variable { return []*autograd.Variable{c.g, c.w, c.u} }

// TestUnrollRematCustomCells differentiates remat against the plain
// Unroll+Backward reference over custom cells, across loss shapes
// (ascending, last-only, middle-seeded, out-of-order) and chunk sizes.
func TestUnrollRematCustomCells(t *testing.T) {
	cells := map[string]func(rng *rand.Rand) (Cell, Module){
		"rnn": func(rng *rand.Rand) (Cell, Module) {
			c := newRNNCell(3, 4, rng)
			return c, c
		},
		"spineRNN": func(rng *rand.Rand) (Cell, Module) {
			c := newSpineRNNCell(3, 4, rng)
			return c, c
		},
	}
	for name, mk := range cells {
		for _, loss := range []int{0, 1, 7, 8} {
			for _, chunk := range []int{1, 2, 5, 17} {
				tc := rematCase{name: name, inDim: 3, units: 4, batch: 2, T: 9, chunkSize: chunk, h0: true, ts: 1.0, loss: loss}
				t.Run(name+"-loss"+itoa(loss)+"-c"+itoa(chunk), func(t *testing.T) {
					rng := rand.New(rand.NewSource(int64(1000 + chunk + 10*loss)))
					cell, module := mk(rng)
					readout := NewLinear(4, 1, rng)
					params := ParametersOf(module, readout)
					xs := make([]*autograd.Variable, tc.T)
					targets := make([]*autograd.Variable, tc.T)
					for i := range xs {
						xs[i] = autograd.Var(tensor.Uniform(rng, -1, 1, tc.batch, tc.inDim))
						targets[i] = autograd.Var(tensor.Uniform(rng, -0.5, 0.5, tc.batch, 1))
					}
					h0 := autograd.Var(tensor.Uniform(rng, -0.5, 0.5, tc.batch, tc.units))
					want := runRematReference(cell, params, readout, xs, targets, h0, tc)
					got := runRematCandidate(cell, params, readout, xs, targets, h0, tc)
					for key, wantBits := range want {
						diffBits(t, name+"/"+key, got[key], wantBits)
					}
				})
			}
		}
	}
}

// mixedCell consumes one gate g at TWO structural positions per step:
// Hadamard(g, h) with g first (spine — the DFS descends into g's chain
// before the state) and Add(out, g) on the output branch (output class).
// No sweep decomposition reproduces that leaf's fold; the classification
// probe sees both consumers and must panic instead of drifting.
type mixedCell struct {
	g, w, u *autograd.Variable
	units   int
}

func newMixedCell(inDim, units int, rng *rand.Rand) *mixedCell {
	return &mixedCell{
		g:     autograd.Var(tensor.Uniform(rng, 0.5, 1.5, units)),
		w:     autograd.Var(tensor.Uniform(rng, -0.4, 0.4, units, units)),
		u:     autograd.Var(tensor.Uniform(rng, -0.4, 0.4, inDim, units)),
		units: units,
	}
}

func (c *mixedCell) StateSize() int { return c.units }

func (c *mixedCell) Step(x, h *autograd.Variable, ts float64) (out, hNew *autograd.Variable) {
	if h == nil {
		h = autograd.Var(tensor.New(x.Data.Rows(), c.units))
	}
	gated := autograd.Hadamard(c.g, h) // g FIRST: spine position
	v := autograd.Tanh(autograd.Add(autograd.MatMul(gated, c.w), autograd.MatMul(x, c.u)))
	out = autograd.Add(autograd.Scale(v, 2), c.g) // g ALSO on the output branch
	return out, v
}

func (c *mixedCell) Parameters() []*autograd.Variable { return []*autograd.Variable{c.g, c.w, c.u} }

// TestUnrollRematMultiClassPanic: a leaf consumed in more than one fold
// class per step is reported by the structural probe (RT-R Medium-4) —
// the alternative is a silent rounding-order drift on that leaf.
func TestUnrollRematMultiClassPanic(t *testing.T) {
	rng := rand.New(rand.NewSource(4242))
	cell := newMixedCell(3, 4, rng)
	xs := make([]*autograd.Variable, 4)
	for i := range xs {
		xs[i] = autograd.Var(tensor.Uniform(rng, -1, 1, 2, 3))
	}
	h0 := autograd.Var(tensor.Uniform(rng, -0.5, 0.5, 2, 4))
	lossFn := func(ys []*autograd.Variable) *autograd.Variable {
		return autograd.MeanAll(autograd.Hadamard(ys[len(ys)-1], ys[len(ys)-1]))
	}
	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("multi-class leaf: no panic")
		}
		if msg := fmt.Sprint(r); !strings.Contains(msg, "fold classes") {
			t.Fatalf("multi-class leaf: panic %q, want the fold-classes message", msg)
		}
	}()
	UnrollRemat(cell, cell.Parameters(), xs, h0, 1.0, 3, lossFn)
}

// recompute unit LONGER than chunkSize (several consecutive
// non-record-high seeds forbid every internal boundary) and pins that the
// result is still bit-exact — the documented price of fidelity.
func TestUnrollRematLongMergeLoss(t *testing.T) {
	tc := rematCase{name: "longmerge", inDim: 2, units: 4, unfolds: 3, batch: 2, T: 12, chunkSize: 3, h0: true, ts: 1.0, loss: 9}
	cell, params, readout, xs, targets, h0 := rematSetup(tc, 4242)
	want := runRematReference(cell, params, readout, xs, targets, h0, tc)
	got := runRematCandidate(cell, params, readout, xs, targets, h0, tc)
	for key, wantBits := range want {
		diffBits(t, key, got[key], wantBits)
	}
}

// TestUnrollRematGateLossBitExact pins the consumer-position rule of the
// loss-order check: a learnable output gate loss MeanAll((g⊙y_last−t)²)
// visits the parameter g BEFORE the seeded output when g is the first
// Hadamard operand, yet its only loss-side consumer appends after the
// output's subtree — the whole-graph backward still delivers loss-side
// first, the sweep's replay order. Both spellings must pass and stay
// bit-exact against the whole-graph reference (RT-R Critical-1).
func TestUnrollRematGateLossBitExact(t *testing.T) {
	for _, pFirst := range []bool{true, false} {
		name := "gate-second"
		if pFirst {
			name = "gate-first"
		}
		t.Run(name, func(t *testing.T) {
			build := func() (*rnnCell, []*autograd.Variable, *autograd.Variable, *autograd.Variable) {
				rng := rand.New(rand.NewSource(815))
				c := newRNNCell(3, 4, rng)
				for i := range c.b.Data.Data {
					c.b.Data.Data[i] = 0.5 + 0.1*float32(i) // an active gate
				}
				xs := make([]*autograd.Variable, 2)
				for i := range xs {
					xs[i] = autograd.Var(tensor.Uniform(rng, -1, 1, 2, 3))
				}
				h0 := autograd.Var(tensor.Uniform(rng, -0.5, 0.5, 2, 4))
				target := autograd.Var(tensor.Uniform(rng, -0.5, 0.5, 2, 4))
				return c, xs, h0, target
			}
			lossOf := func(g, target *autograd.Variable) func(ys []*autograd.Variable) *autograd.Variable {
				return func(ys []*autograd.Variable) *autograd.Variable {
					y := ys[len(ys)-1]
					gated := autograd.Hadamard(y, g)
					if pFirst {
						gated = autograd.Hadamard(g, y)
					}
					diff := autograd.Sub(gated, target)
					return autograd.MeanAll(autograd.Hadamard(diff, diff))
				}
			}
			cA, xsA, h0A, tA := build()
			cB, xsB, h0B, tB := build()
			zeroRematLeaves(cA.Parameters(), xsA, h0A)
			ysA, _ := Unroll(cA, xsA, h0A, 1.0)
			lossOf(cA.b, tA)(ysA).Backward()
			zeroRematLeaves(cB.Parameters(), xsB, h0B)
			UnrollRemat(cB, cB.Parameters(), xsB, h0B, 1.0, 2, lossOf(cB.b, tB))
			cmp := func(name string, a, b *autograd.Variable) {
				_, want := gradBits(a)
				_, got := gradBits(b)
				diffBits(t, name, got, want)
			}
			for i, p := range cA.Parameters() {
				cmp("p"+itoa(i), p, cB.Parameters()[i])
			}
			for i := range xsA {
				cmp("x"+itoa(i), xsA[i], xsB[i])
			}
			cmp("h0", h0A, h0B)
		})
	}
}

// skipCell taps x twice per step: inside the state subgraph (u) and on
// the output branch (u2, a skip connection). x is a single-step leaf —
// exempt from the multi-class panic (RT-R Medium-3) — but its gradient
// must stay bit-exact, verified here against the whole-graph reference.
type skipCell struct {
	w, u, u2 *autograd.Variable
	units    int
}

func newSkipCell(inDim, units int, rng *rand.Rand) *skipCell {
	return &skipCell{
		w:     autograd.Var(tensor.Uniform(rng, -0.4, 0.4, units, units)),
		u:     autograd.Var(tensor.Uniform(rng, -0.4, 0.4, inDim, units)),
		u2:    autograd.Var(tensor.Uniform(rng, -0.4, 0.4, inDim, units)),
		units: units,
	}
}

func (c *skipCell) StateSize() int { return c.units }

func (c *skipCell) Step(x, h *autograd.Variable, ts float64) (out, hNew *autograd.Variable) {
	if h == nil {
		h = autograd.Var(tensor.New(x.Data.Rows(), c.units))
	}
	v := autograd.Tanh(autograd.Add(autograd.MatMul(h, c.w), autograd.MatMul(x, c.u)))
	o := autograd.Add(v, autograd.MatMul(x, c.u2)) // skip: x on the output branch too
	return o, v
}

func (c *skipCell) Parameters() []*autograd.Variable { return []*autograd.Variable{c.w, c.u, c.u2} }

// TestUnrollRematSkipInputCell differentiates remat against the
// whole-graph reference over the skip-connection cell across loss shapes
// — ascending, last-only, even-steps and out-of-order (the affine pass
// also accumulates into xs under out-of-order visits; the pick must keep
// the rest sweep's value, exactly) — and chunk sizes.
func TestUnrollRematSkipInputCell(t *testing.T) {
	const T = 5
	lossOf := func(kind int) func(ys []*autograd.Variable) *autograd.Variable {
		term := func(y *autograd.Variable, s float32) *autograd.Variable {
			return autograd.SumAll(autograd.Scale(y, s))
		}
		switch kind {
		case 0: // ascending
			return func(ys []*autograd.Variable) *autograd.Variable {
				var acc *autograd.Variable
				for i, y := range ys {
					if i == 0 {
						acc = term(y, 1)
					} else {
						acc = autograd.Add(acc, term(y, float32(i+1)))
					}
				}
				return acc
			}
		case 1: // last only
			return func(ys []*autograd.Variable) *autograd.Variable { return term(ys[T-1], 1) }
		case 2: // even steps
			return func(ys []*autograd.Variable) *autograd.Variable {
				var acc *autograd.Variable
				for i := 0; i < T; i += 2 {
					if acc == nil {
						acc = term(ys[i], float32(i+1))
					} else {
						acc = autograd.Add(acc, term(ys[i], float32(i+1)))
					}
				}
				return acc
			}
		default: // 3: out of order — forces the affine pass
			return func(ys []*autograd.Variable) *autograd.Variable {
				return autograd.Add(autograd.Add(term(ys[T/2], 1), term(ys[T/4], 2)), term(ys[3*T/4], 3))
			}
		}
	}
	for kind := 0; kind < 4; kind++ {
		for _, chunk := range []int{1, 2, 5} {
			t.Run("loss"+itoa(kind)+"-c"+itoa(chunk), func(t *testing.T) {
				build := func() (*skipCell, []*autograd.Variable, *autograd.Variable) {
					rng := rand.New(rand.NewSource(51))
					c := newSkipCell(3, 4, rng)
					xs := make([]*autograd.Variable, T)
					for i := range xs {
						xs[i] = autograd.Var(tensor.Uniform(rng, -1, 1, 2, 3))
					}
					return c, xs, autograd.Var(tensor.Uniform(rng, -0.5, 0.5, 2, 4))
				}
				cA, xsA, h0A := build()
				cB, xsB, h0B := build()
				zeroRematLeaves(cA.Parameters(), xsA, h0A)
				ysA, _ := Unroll(cA, xsA, h0A, 1.0)
				lossOf(kind)(ysA).Backward()
				zeroRematLeaves(cB.Parameters(), xsB, h0B)
				UnrollRemat(cB, cB.Parameters(), xsB, h0B, 1.0, chunk, lossOf(kind))
				cmp := func(name string, a, b *autograd.Variable) {
					_, want := gradBits(a)
					_, got := gradBits(b)
					diffBits(t, name, got, want)
				}
				for i, p := range cA.Parameters() {
					cmp("p"+itoa(i), p, cB.Parameters()[i])
				}
				for i := range xsA {
					cmp("x"+itoa(i), xsA[i], xsB[i])
				}
				cmp("h0", h0A, h0B)
			})
		}
	}
}
