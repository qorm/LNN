# LNN cookbook

> English | [中文](zh/cookbook.md)

**Purpose:** task-oriented recipes — "I want to do X, how?" Each recipe is
one scenario in one sentence, a complete program you can copy, the key
lines explained, the verbatim measured output, and pointers into the
concept guides ([training.md](training.md), [persistence.md](persistence.md),
[ltc.md](ltc.md), [cfc.md](cfc.md), [pitfalls.md](pitfalls.md),
[shapes-and-broadcasting.md](shapes-and-broadcasting.md),
[architecture.md](architecture.md)) where the *why* lives.

**Every program below was compiled and run against this repository**
(Go 1.26.5, `darwin/arm64`); the output blocks are the real measured
output. To run them yourself, put each recipe in its own directory as a
`package main` inside a scratch module that points at the library:

```
module lnncook

go 1.26.5

require github.com/qorm/LNN v0.0.0

replace github.com/qorm/LNN => /path/to/LNN   // your checkout
```

(With a published version instead, `go get github.com/qorm/LNN@latest`
and drop the `replace`.) All seeds are fixed, so on `arm64` you will see
exactly these numbers; on other architectures the trailing digits may
differ by floating-point contraction — see [faq.md](faq.md) ("do
last-digit differences across machines matter?").

## Index

| # | Recipe | In one line |
|---|---|---|
| 1 | [Minimal training loop](#1-minimal-training-loop) | The four-phase loop with pure autograd: build graph → `ZeroGrad` → `Backward` → update. |
| 2 | [Training with an optimizer: Adam and norm clipping](#2-training-with-an-optimizer-adam-and-norm-clipping) | `NewAdamDefault` + the caller-owned global gradient-norm clip, `Step` form. |
| 3 | [Gradient accumulation](#3-gradient-accumulation) | Zero once every N backwards, step once: effective large batch for free. |
| 4 | [Checkpoint and resume, bit-exact](#4-checkpoint-and-resume-bit-exact) | `SaveCfC` + `SaveState` (Adam) → load into fresh objects → resume identical to uninterrupted training. |
| 5 | [Event-driven sequences with variable ts](#5-event-driven-sequences-with-variable-ts) | `Unroll` takes one `ts`; hand-roll the `Step` loop when the sampling gap varies. |
| 6 | [Inspecting and debugging a model](#6-inspecting-and-debugging-a-model) | Parameter inventory, per-parameter gradient norms, NaN/Inf detection. |
| 7 | [Custom losses](#7-custom-losses) | Masked MSE and L1: a loss is just an op graph. |
| 8 | [Choosing between LTC and CfC](#8-choosing-between-ltc-and-cfc) | Decision table plus the same training loop driving both cells. |
| 9 | [Composing multi-module models](#9-composing-multi-module-models) | Cell → Linear → Linear, `ParametersOf` aggregation, forward chaining. |
| 10 | [Loading untrusted model files safely](#10-loading-untrusted-model-files-safely) | Error classification pattern: kind / version / limits / truncation. |
| 11 | [Annealing the learning rate mid-training](#11-annealing-the-learning-rate-mid-training) | Hyperparameters are exported fields: write `opt.LR`, in effect from the next `Step`. |
| 12 | [Deterministic reproduction](#12-deterministic-reproduction) | Seed discipline: same seeds ⇒ bit-identical runs. |
| 13 | [Long-sequence training: chunked BPTT with UnrollRemat](#13-long-sequence-training-chunked-bptt-with-unrollremat) | Bit-identical gradients at O(chunk) peak graph memory: rematerialization for sequences that outgrow the whole-graph model. |

---

## 1. Minimal training loop

**Scenario:** fit `y = 2x + 1` with plain gradient descent using only
`tensor` + `autograd` — the canonical four-phase loop everything else in
the library is built on.

```go
package main

import (
	"fmt"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/tensor"
)

func main() {
	rng := rand.New(rand.NewSource(42))

	// Data: y = 2x + 1, stored as [n,1] matrices.
	const n = 32
	xs, ys := make([]float32, n), make([]float32, n)
	for i := range xs {
		xs[i] = rng.Float32()*2 - 1
		ys[i] = 2*xs[i] + 1
	}
	x := autograd.Const(tensor.FromData(xs, n, 1))
	y := autograd.Const(tensor.FromData(ys, n, 1))

	w := autograd.Var(tensor.Randn(rng, 1, 1))
	b := autograd.Var(tensor.New(1))

	const epochs, lr = 200, 0.1
	for epoch := 0; epoch < epochs; epoch++ {
		// 1. forward: build a fresh graph.
		pred := autograd.Add(autograd.MatMul(x, w), b)
		diff := autograd.Sub(pred, y)
		loss := autograd.MeanAll(autograd.Hadamard(diff, diff))

		// 2. zero grads, 3. backward.
		w.ZeroGrad()
		b.ZeroGrad()
		loss.Backward()

		// 4. update in place.
		for i := range w.Data.Data {
			w.Data.Data[i] -= lr * w.Grad.Data[i]
		}
		for i := range b.Data.Data {
			b.Data.Data[i] -= lr * b.Grad.Data[i]
		}

		if epoch%40 == 0 || epoch == epochs-1 {
			fmt.Printf("epoch %3d  loss=%.6f  w=%.4f  b=%.4f\n",
				epoch, loss.Value(), w.Data.Data[0], b.Data.Data[0])
		}
	}
}
```

Measured output (seed 42 — deterministic):

```
epoch   0  loss=1.398637  w=0.5503  b=0.1674
epoch  40  loss=0.006162  w=1.8658  b=0.9795
epoch  80  loss=0.000047  w=1.9883  b=0.9982
epoch 120  loss=0.000000  w=1.9990  b=0.9998
epoch 160  loss=0.000000  w=1.9999  b=1.0000
epoch 199  loss=0.000000  w=2.0000  b=1.0000
```

**Key lines:**

- `autograd.Var` marks trainable leaves; `autograd.Const` documents
  "not trainable" (same function, intent-carrying alias).
- The graph is rebuilt every iteration — ops are eager and record
  themselves; there is no tape object.
- `ZeroGrad` **before** `Backward`, never after: leaf gradients
  accumulate across calls by design (recipe 3 exploits exactly this).
- The update writes `p.Data.Data` directly — parameters are plain
  `float32` buffers you own.

**See also:** [training.md](training.md) derives this loop and its three
disciplines; the repository root `README.md` carries the same program as
the quick start; [pitfalls.md](pitfalls.md) §3–4 for repeated-`Backward`
and forward/backward mutation hazards.

---

## 2. Training with an optimizer: Adam and norm clipping

**Scenario:** train a recurrent model the recommended production way —
`optimizer.NewAdamDefault` plus caller-owned global gradient-norm
clipping, on the bounded-accumulator sequence task (needs cross-step
memory).

```go
package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/nn"
	"github.com/qorm/LNN/optimizer"
	"github.com/qorm/LNN/tensor"
)

const (
	inDim   = 1
	units   = 8
	seqLen  = 12
	batch   = 16
	iters   = 250
	maxNorm = 1.0 // global gradient-norm clip
	ts      = 1.0
)

func main() {
	rng := rand.New(rand.NewSource(42))

	cell := nn.NewCfC(inDim, units, nil, rng)
	readout := nn.NewLinear(units, 1, rng)
	params := nn.ParametersOf(cell, readout)
	opt := optimizer.NewAdamDefault(0.01)

	clips := 0
	var maxObserved float64
	var first, last float64
	for it := 0; it < iters; it++ {
		xs := make([]*autograd.Variable, seqLen)
		targets := make([]*autograd.Variable, seqLen)
		state := make([]float32, batch)
		for t := 0; t < seqLen; t++ {
			xb := make([]float32, batch)
			yb := make([]float32, batch)
			for b := 0; b < batch; b++ {
				u := float32(1)
				if rng.Intn(2) == 0 {
					u = -1
				}
				xb[b] = u
				s := state[b] + 0.25*u
				if s > 1 {
					s = 1
				} else if s < -1 {
					s = -1
				}
				state[b] = s
				yb[b] = s
			}
			xs[t] = autograd.Var(tensor.FromData(xb, batch, inDim))
			targets[t] = autograd.Var(tensor.FromData(yb, batch, 1))
		}

		ys, _ := nn.Unroll(cell, xs, nil, ts)
		var acc *autograd.Variable
		for t, y := range ys {
			diff := autograd.Sub(readout.Forward(y), targets[t])
			sq := autograd.Hadamard(diff, diff)
			if t == 0 {
				acc = sq
			} else {
				acc = autograd.Add(acc, sq)
			}
		}
		loss := autograd.Scale(autograd.MeanAll(acc), 1/float32(seqLen))

		for _, p := range params {
			p.ZeroGrad()
		}
		loss.Backward()

		// ----- verbatim from doc/training.md "Gradient clipping" -----
		var norm2 float64
		for _, p := range params {
			if p.Grad == nil {
				continue
			}
			for _, g := range p.Grad.Data {
				norm2 += float64(g) * float64(g)
			}
		}
		if norm := math.Sqrt(norm2); norm > maxNorm {
			s := float32(maxNorm / norm)
			for _, p := range params {
				if p.Grad == nil {
					continue
				}
				for i := range p.Grad.Data {
					p.Grad.Data[i] *= s
				}
			}
		}
		opt.Step(params)
		// ----- end verbatim block -----

		norm := math.Sqrt(norm2)
		if norm > maxNorm {
			clips++
		}
		if norm > maxObserved {
			maxObserved = norm
		}
		if it == 0 {
			first = float64(loss.Value())
		}
		last = float64(loss.Value())
		if it%50 == 0 || it == iters-1 {
			fmt.Printf("iter %3d  loss=%.6f  gnorm=%.3f\n", it, loss.Value(), norm)
		}
	}
	fmt.Printf("first=%.6f last=%.6f\n", first, last)
	fmt.Printf("clip engaged on %d/%d iterations, max observed norm %.3f\n", clips, iters, maxObserved)
}
```

Measured output (seed 42; each `loss` measured *before* that
iteration's update):

```
iter   0  loss=0.620651  gnorm=2.196
iter  50  loss=0.008872  gnorm=0.178
iter 100  loss=0.007931  gnorm=0.095
iter 150  loss=0.004200  gnorm=0.116
iter 200  loss=0.004664  gnorm=0.147
iter 249  loss=0.004286  gnorm=0.090
first=0.620651 last=0.004286
clip engaged on 14/250 iterations, max observed norm 2.196
```

**Key lines:**

- The block between the `verbatim` markers is copied character for
  character from the `optimizer.Step` clipping snippet in
  [training.md](training.md#gradient-clipping-do-it): accumulate the
  global norm in `float64`, and when it exceeds `maxNorm`, rescale the
  **gradients** (not the step — `Step` applies its own learning rate)
  *before* calling `opt.Step(params)`.
- `NewAdamDefault(lr)` is Kingma & Ba's `0.9 / 0.999 / 1e-8`; `Step`
  skips parameters with nil `Grad` and never zeroes gradients — that is
  your contract ([training.md](training.md#the-step-contract)).
- Adam's update is self-normalizing, so with Adam (unlike plain SGD on
  the LTC) the clip engages mostly early: 14/250 iterations here, the
  first spike at norm `2.196`. Keep it anyway — it is the difference
  between converging and diverging when a spike does come.

**See also:** [training.md](training.md#gradient-clipping-do-it) for why
clipping is non-optional on the LTC; `examples/cfc-sequence` in the
repository is this pattern with plain SGD, loss `0.620651 → 0.029091`.

---

## 3. Gradient accumulation

**Scenario:** you want an effective batch of `N` micro-batches but must
run them one at a time (memory, or streamed data). Leaf gradients
accumulate across `Backward` calls *by design* — so accumulation needs
no library support at all: `ZeroGrad` once, backward N times, `Step`
once.

```go
package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/optimizer"
	"github.com/qorm/LNN/tensor"
)

// buildGraph builds the MSE graph of y = x*w + b over the rows given.
func buildGraph(x, y, w, b *autograd.Variable) *autograd.Variable {
	pred := autograd.Add(autograd.MatMul(x, w), b)
	diff := autograd.Sub(pred, y)
	return autograd.MeanAll(autograd.Hadamard(diff, diff))
}

func main() {
	rng := rand.New(rand.NewSource(42))
	const n, micro = 32, 4 // 4 micro-batches of 8 rows

	xs, ys := make([]float32, n), make([]float32, n)
	for i := range xs {
		xs[i] = rng.Float32()*2 - 1
		ys[i] = 2*xs[i] + 1
	}

	// --- (a) one full-batch backward: the reference gradient ---
	w := autograd.Var(tensor.Randn(rng, 1, 1))
	b := autograd.Var(tensor.New(1))
	xFull := autograd.Const(tensor.FromData(xs, n, 1))
	yFull := autograd.Const(tensor.FromData(ys, n, 1))
	buildGraph(xFull, yFull, w, b).Backward()
	gwFull, gbFull := w.Grad.Data[0], b.Grad.Data[0]

	// --- (b) N micro-batch backwards, ZeroGrad'd once up front ---
	// Each micro loss is scaled by 1/micro so the accumulated gradient
	// equals the mean over the whole effective batch (each micro MeanAll
	// divides by its own rows; the 1/micro factor turns the sum of those
	// per-batch means into the grand mean).
	w.ZeroGrad()
	b.ZeroGrad()
	rows := n / micro
	for m := 0; m < micro; m++ {
		xm := autograd.Const(tensor.FromData(xs[m*rows:(m+1)*rows], rows, 1))
		ym := autograd.Const(tensor.FromData(ys[m*rows:(m+1)*rows], rows, 1))
		lm := autograd.Scale(buildGraph(xm, ym, w, b), 1/float32(micro))
		lm.Backward() // grads accumulate into w, b
	}
	fmt.Printf("full-batch    grad: dw=%+.8f db=%+.8f\n", gwFull, gbFull)
	fmt.Printf("4 micro-batch grad: dw=%+.8f db=%+.8f\n", w.Grad.Data[0], b.Grad.Data[0])
	fmt.Printf("max |diff| = %.3e (float32 addition order only)\n",
		math.Max(math.Abs(float64(w.Grad.Data[0]-gwFull)), math.Abs(float64(b.Grad.Data[0]-gbFull))))

	// --- one Step on the accumulated gradients ---
	opt := optimizer.NewSGD(0.1)
	wBefore := w.Data.Data[0]
	opt.Step([]*autograd.Variable{w, b})
	fmt.Printf("one Step on accumulated grads: w %.6f -> %.6f\n", wBefore, w.Data.Data[0])

	// --- a real accumulation loop: effective batch 4x32 online data ---
	rng2 := rand.New(rand.NewSource(7))
	w2 := autograd.Var(tensor.Randn(rng2, 1, 1))
	b2 := autograd.Var(tensor.New(1))
	opt2 := optimizer.NewSGD(0.1)
	const epochs, accum = 200, 4
	for e := 0; e < epochs; e++ {
		w2.ZeroGrad()
		b2.ZeroGrad() // zero once per `accum` micro-batches
		var lossSum float64
		for m := 0; m < accum; m++ {
			xb := make([]float32, n)
			yb := make([]float32, n)
			for i := range xb {
				xb[i] = rng2.Float32()*2 - 1
				yb[i] = 2*xb[i] + 1
			}
			l := buildGraph(autograd.Const(tensor.FromData(xb, n, 1)),
				autograd.Const(tensor.FromData(yb, n, 1)), w2, b2)
			autograd.Scale(l, 1/float32(accum)).Backward()
			lossSum += float64(l.Value())
		}
		opt2.Step([]*autograd.Variable{w2, b2}) // step once per window
		if e%40 == 0 || e == epochs-1 {
			fmt.Printf("epoch %3d  avg loss=%.6f  w=%.4f  b=%.4f\n",
				e, lossSum/accum, w2.Data.Data[0], b2.Data.Data[0])
		}
	}
}
```

Measured output (seed 42 / seed 7 — deterministic):

```
full-batch    grad: dw=-0.73727763 db=-1.67409205
4 micro-batch grad: dw=-0.73727751 db=-1.67409170
max |diff| = 3.576e-07 (float32 addition order only)
one Step on accumulated grads: w 0.476583 -> 0.550311
epoch   0  avg loss=2.813679  w=0.2184  b=0.2294
epoch  40  avg loss=0.005552  w=1.8851  b=1.0031
epoch  80  avg loss=0.000021  w=1.9925  b=0.9998
epoch 120  avg loss=0.000000  w=1.9995  b=1.0000
epoch 160  avg loss=0.000000  w=2.0000  b=1.0000
epoch 199  avg loss=0.000000  w=2.0000  b=1.0000
```

**Key lines:**

- `w.ZeroGrad()` runs **once**, before the four `Backward` calls — the
  four per-batch gradients sum into the same leaf buffers. This is the
  documented flip side of "gradients accumulate"
  ([training.md](training.md#the-step-contract): "zero once every N
  iterations … and you get gradient accumulation over N micro-batches
  for free").
- `autograd.Scale(l, 1/micro)` matters: each micro loss is a mean over
  its *own* rows, so without the `1/micro` factor the accumulated
  gradient would be `micro` times the grand-mean gradient. With it, the
  sum equals the full-batch gradient up to `float32` addition order
  (`3.6e-7` above).
- `Step` runs once per window, after all backwards — `Step` never
  zeroes, so it happily consumes accumulated gradients.
- If you clip (recipe 2), clip once over the *accumulated* gradients,
  after the last micro-batch backward and before the single `Step`.

**See also:** [faq.md](faq.md) "why are my gradients still growing after
`Backward`?" for the accumulation semantics; [pitfalls.md](pitfalls.md)
§3 for the exactly-linear repeated-`Backward` guarantee this relies on.

---

## 4. Checkpoint and resume, bit-exact

**Scenario:** train, checkpoint (model + readout + optimizer state),
load into *fresh* objects later, resume — and prove the resumed
trajectory is bit-identical to uninterrupted training. This is the
compact task version of the full worked example in
[persistence.md](persistence.md).

```go
package main

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"os"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/nn"
	"github.com/qorm/LNN/optimizer"
	"github.com/qorm/LNN/serialize"
	"github.com/qorm/LNN/tensor"
)

const (
	inDim, units  = 1, 8
	seqLen, batch = 12, 16
	lr, ts        = 0.01, 1.0
	total, split  = 100, 50
)

func main() {
	// Reference: 100 uninterrupted iterations.
	cellA := nn.NewCfC(inDim, units, nil, rand.New(rand.NewSource(7)))
	readA := nn.NewLinear(units, 1, rand.New(rand.NewSource(7)))
	paramsA := nn.ParametersOf(cellA, readA)
	ref := train(rand.New(rand.NewSource(42)), cellA, readA, paramsA,
		optimizer.NewAdamDefault(lr), total)

	// Checkpointed run, phase 1: identical construction, 50 iterations.
	cellB := nn.NewCfC(inDim, units, nil, rand.New(rand.NewSource(7)))
	readB := nn.NewLinear(units, 1, rand.New(rand.NewSource(7)))
	paramsB := nn.ParametersOf(cellB, readB)
	optB := optimizer.NewAdamDefault(lr)
	dataB := rand.New(rand.NewSource(42))
	first := train(dataB, cellB, readB, paramsB, optB, split)

	// Checkpoint: model + readout parameters + Adam state.
	var modelBuf, paramBuf, stateBuf bytes.Buffer
	must(nn.SaveCfC(&modelBuf, cellB))
	must(serialize.WriteParameters(&paramBuf, readB.Parameters()))
	must(optimizer.SaveState(&stateBuf, optB, paramsB))
	fmt.Printf("checkpoint at step %d: model %d B, readout %d B, Adam state %d B\n",
		split, modelBuf.Len(), paramBuf.Len(), stateBuf.Len())

	// Phase 2: fresh objects, state restored from the streams.
	loaded, err := nn.LoadCfC(bytes.NewReader(modelBuf.Bytes()))
	must(err)
	readC := nn.NewLinear(units, 1, rand.New(rand.NewSource(123))) // seed irrelevant
	must(serialize.LoadParameters(bytes.NewReader(paramBuf.Bytes()), readC.Parameters()))
	paramsC := nn.ParametersOf(loaded, readC)
	optC := optimizer.NewAdamDefault(lr)
	must(optimizer.LoadState(bytes.NewReader(stateBuf.Bytes()), optC, paramsC))
	second := train(dataB, loaded, readC, paramsC, optC, total-split)

	// Compare against the uninterrupted run, bit for bit.
	resumed := append(first, second...)
	same := true
	for i := range ref {
		if resumed[i] != ref[i] {
			same = false
		}
	}
	fmt.Printf("steps 0..%d loss bits identical to uninterrupted run: %v\n", total-1, same)
	sameP := true
	for i := range paramsA {
		a, c := paramsA[i].Data.Data, paramsC[i].Data.Data
		for j := range a {
			if math.Float32bits(a[j]) != math.Float32bits(c[j]) {
				sameP = false
			}
		}
	}
	fmt.Printf("final parameters bit-identical: %v\n", sameP)
	fmt.Printf("loss: iter %d = %.6f, iter %d = %.6f\n",
		split-1, math.Float32frombits(ref[split-1]), total-1, math.Float32frombits(ref[total-1]))
}

// train runs iters iterations of the bounded-accumulator task, returning the
// loss measured before each iteration's update as float32 bit patterns.
func train(rng *rand.Rand, cell nn.Cell, readout *nn.Linear,
	params []*autograd.Variable, opt optimizer.Optimizer, iters int) []uint32 {
	losses := make([]uint32, 0, iters)
	for it := 0; it < iters; it++ {
		xs := make([]*autograd.Variable, seqLen)
		targets := make([]*autograd.Variable, seqLen)
		state := make([]float32, batch)
		for t := 0; t < seqLen; t++ {
			xb := make([]float32, batch)
			yb := make([]float32, batch)
			for b := 0; b < batch; b++ {
				u := float32(1)
				if rng.Intn(2) == 0 {
					u = -1
				}
				xb[b] = u
				s := state[b] + 0.25*u
				if s > 1 {
					s = 1
				} else if s < -1 {
					s = -1
				}
				state[b] = s
				yb[b] = s
			}
			xs[t] = autograd.Var(tensor.FromData(xb, batch, inDim))
			targets[t] = autograd.Var(tensor.FromData(yb, batch, 1))
		}
		ys, _ := nn.Unroll(cell, xs, nil, ts)
		var acc *autograd.Variable
		for t, y := range ys {
			diff := autograd.Sub(readout.Forward(y), targets[t])
			sq := autograd.Hadamard(diff, diff)
			if t == 0 {
				acc = sq
			} else {
				acc = autograd.Add(acc, sq)
			}
		}
		loss := autograd.Scale(autograd.MeanAll(acc), 1/float32(seqLen))
		losses = append(losses, math.Float32bits(loss.Value()))
		for _, p := range params {
			p.ZeroGrad()
		}
		loss.Backward()
		opt.Step(params)
	}
	return losses
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "FATAL:", err)
		os.Exit(1)
	}
}
```

Measured output (deterministic):

```
checkpoint at step 50: model 1859 B, readout 71 B, Adam state 2732 B
steps 0..99 loss bits identical to uninterrupted run: true
final parameters bit-identical: true
loss: iter 49 = 0.024681, iter 99 = 0.011424
```

**Key lines:**

- Three streams, three savers: `nn.SaveCfC` (the cell — for an LTC use
  `nn.SaveLTC`), `serialize.WriteParameters` (the readout's `[]*autograd.Variable`),
  `optimizer.SaveState` (Adam's moments, step counts and bias-correction
  powers — `"LNO1"` format).
- Load into **fresh** objects; every RNG seed on the load side is
  irrelevant because `Load` overwrites every RNG-derived field.
  `LoadParameters` copies values back *in place*, so the variable
  pointers keep their identity.
- The model checkpoint alone is not enough for Adam/Momentum: without
  `SaveState`/`LoadState` the resumed run silently restarts bias
  correction at `t = 0`. With it, all 100 per-step losses agree bit for
  bit (`Float32bits`) with the uninterrupted run, as do the final
  parameters.
- **Hyperparameters are not in the stream:** construct the destination
  optimizer with the same `LR`/betas you trained with. `LoadState`
  verifies `Beta1^t`/`Beta2^t` against the saved powers, so a betas
  mismatch fails loudly as corruption.
- **Parameter order is the key:** `LoadState` attaches record *i* to
  `params[i]` — pass the same module order to Save and Load (here
  `nn.ParametersOf(loaded, readC)` mirrors `nn.ParametersOf(cellB, readB)`).

**See also:** [persistence.md](persistence.md) — the full train→save→
load→resume program (with hostile-stream demos), the `"LNO1"` wire
format byte by byte, and the bit-exact resume contract for all three
optimizers; [faq.md](faq.md) "how do I resume training with Adam?".

---

## 5. Event-driven sequences with variable ts

**Scenario:** your data is event-streamed with irregular gaps (sensor
events, transactions, spikes), and the time between events should drive
the cell dynamics. `nn.Unroll` takes a *single* `ts` for the whole
sequence — for per-step time spans, write the three-line `Step` loop
yourself.

```go
package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/nn"
	"github.com/qorm/LNN/tensor"
)

// unrollVariableTs is the hand-rolled loop nn.Unroll would run, with a
// different ts per step — the event-driven / variable-step form.
func unrollVariableTs(cell nn.Cell, xs []*autograd.Variable, h0 *autograd.Variable, dts []float64) ([]*autograd.Variable, *autograd.Variable) {
	h := h0
	ys := make([]*autograd.Variable, len(xs))
	for i, x := range xs {
		var y *autograd.Variable
		y, h = cell.Step(x, h, dts[i]) // one time span per event
		ys[i] = y
	}
	return ys, h
}

func main() {
	rng := rand.New(rand.NewSource(42))
	cell := nn.NewCfC(1, 6, nil, rng) // LTC drives identically: same Cell interface

	// Irregularly sampled sensor: events arrive after gaps of 0.2, 1.0,
	// 3.0, 0.05, 1.5 time units. The gap since the previous event IS ts.
	gaps := []float64{0.2, 1.0, 3.0, 0.05, 1.5}
	xs := make([]*autograd.Variable, len(gaps))
	for i := range xs {
		xs[i] = autograd.Var(tensor.Uniform(rng, -1, 1, 2, 1)) // batch 2
	}

	// 1. event-driven loop: ts follows the sampling gaps.
	ys, h := unrollVariableTs(cell, xs, nil, gaps)
	fmt.Println("variable-ts unroll (ts = inter-event gap):")
	for i, y := range ys {
		fmt.Printf("  step %d  ts=%.2f  out[0]=%+.6f\n", i, gaps[i], y.Data.Data[0])
	}
	fmt.Printf("  final state finite: %v\n", allFinite(h.Data.Data))

	// 2. same inputs at one fixed ts, for contrast.
	ysFix, _ := nn.Unroll(cell, xs, nil, 1.0)
	fmt.Println("fixed-ts unroll (ts = 1.0 every step):")
	for i, y := range ysFix {
		fmt.Printf("  step %d            out[0]=%+.6f\n", i, y.Data.Data[0])
	}

	// 3. the whole variable-ts sequence is one graph: one Backward.
	target := autograd.Const(tensor.New(2, 1))
	var acc *autograd.Variable
	for i, y := range ys {
		d := autograd.Sub(y, target)
		sq := autograd.Hadamard(d, d)
		if i == 0 {
			acc = sq
		} else {
			acc = autograd.Add(acc, sq)
		}
	}
	loss := autograd.MeanAll(acc)
	for _, p := range cell.Parameters() {
		p.ZeroGrad()
	}
	loss.Backward()
	finite := true
	for _, p := range cell.Parameters() {
		if !allFinite(p.Grad.Data) {
			finite = false
		}
	}
	fmt.Printf("loss=%.6f  all 13 parameter grads finite through variable-ts BPTT: %v\n",
		loss.Value(), finite)

	// 4. the ts contract: positive and finite, else panic.
	for _, bad := range []float64{0, -1, math.Inf(1), math.NaN()} {
		func() {
			defer func() {
				if r := recover(); r != nil {
					fmt.Printf("Step with ts=%v -> panic: %v\n", bad, r)
				}
			}()
			cell.Step(xs[0], nil, bad)
		}()
	}
}

func allFinite(s []float32) bool {
	for _, v := range s {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return false
		}
	}
	return true
}
```

Measured output (seed 42):

```
variable-ts unroll (ts = inter-event gap):
  step 0  ts=0.20  out[0]=+0.034594
  step 1  ts=1.00  out[0]=+0.111270
  step 2  ts=3.00  out[0]=+0.134097
  step 3  ts=0.05  out[0]=+0.135209
  step 4  ts=1.50  out[0]=+0.168719
  final state finite: true
fixed-ts unroll (ts = 1.0 every step):
  step 0            out[0]=+0.111878
  step 1            out[0]=+0.120359
  step 2            out[0]=+0.136230
  step 3            out[0]=+0.149414
  step 4            out[0]=+0.171723
loss=0.090011  all 13 parameter grads finite through variable-ts BPTT: true
Step with ts=0 -> panic: nn.CfC.Step: ts must be positive and finite, got 0
Step with ts=-1 -> panic: nn.CfC.Step: ts must be positive and finite, got -1
Step with ts=+Inf -> panic: nn.CfC.Step: ts must be positive and finite, got +Inf
Step with ts=NaN -> panic: nn.CfC.Step: ts must be positive and finite, got NaN
```

**Key lines:**

- The loop is exactly what `nn.Unroll` does internally, with `dts[i]`
  replacing the single `ts` — thread the state `h` from step to step,
  `nil` for a zero start.
- **Choosing `ts`:** set it to the inter-event gap in the ODE's time
  units. A small `ts` barely advances the membrane (step 3, gap 0.05:
  the output almost doesn't move); a large `ts` relaxes it toward
  steady state. Anchor one time unit to something physical (one
  second, one sampling interval) and express gaps in those units.
- The whole variable-`ts` sequence is still one graph: a single
  `Backward` differentiates through all steps and all time spans.
- `ts` must be positive and finite — `0`, negatives, `±Inf` and `NaN`
  panic (demonstrated above).

**See also:** [ltc.md](ltc.md#the-time-span-ts) — the full `ts` contract
and finiteness domains (`ts ≳ 1e-3` keeps full physical fidelity; below
`≈ 1e-38` is a finiteness-only domain); [cfc.md](cfc.md) for the CfC's
identical contract; [faq.md](faq.md) "how do I choose `ts`?".

---

## 6. Inspecting and debugging a model

**Scenario:** before trusting a model — or when a run misbehaves —
enumerate its parameters, check values for non-finite entries, and look
at per-parameter gradient norms after one backward.

```go
package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/nn"
	"github.com/qorm/LNN/tensor"
)

// hasNaNInf scans a buffer for non-finite values.
func hasNaNInf(s []float32) bool {
	for _, v := range s {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return true
		}
	}
	return false
}

// gradNorm returns the L2 norm of a gradient buffer, accumulated in float64.
func gradNorm(g []float32) float64 {
	var n2 float64
	for _, v := range g {
		n2 += float64(v) * float64(v)
	}
	return math.Sqrt(n2)
}

func main() {
	rng := rand.New(rand.NewSource(42))
	cell := nn.NewLTC(2, 6, nil, 4, rng)
	readout := nn.NewLinear(6, 1, rng)
	params := nn.ParametersOf(cell, readout)

	// Parameter inventory: index, shape, element count. The LTC order is
	// documented in doc/ltc.md's parameter table.
	names := []string{
		"gleak", "vleak", "cm", "mu", "sigma", "w", "sMu", "sSigma", "sW",
		"inW", "inB", "outW", "outB", "readout.W", "readout.B",
	}
	total := 0
	fmt.Println("parameter inventory:")
	for i, p := range params {
		fmt.Printf("  [%2d] %-10s shape %-7s %4d elems\n", i, names[i], fmt.Sprint(p.Data.Shape), len(p.Data.Data))
		total += len(p.Data.Data)
		if hasNaNInf(p.Data.Data) {
			fmt.Printf("       !! non-finite values at init\n")
		}
	}
	fmt.Printf("  total %d params, %d trainable elements\n", len(params), total)

	// One forward/backward pass, then per-parameter diagnostics.
	x := autograd.Var(tensor.Uniform(rng, -1, 1, 3, 2))
	y := autograd.Var(tensor.Uniform(rng, -1, 1, 3, 1))
	for _, p := range params {
		p.ZeroGrad()
	}
	ys, _ := nn.Unroll(cell, []*autograd.Variable{x, x, x, x}, nil, 1.0)
	diff := autograd.Sub(readout.Forward(ys[3]), y)
	loss := autograd.MeanAll(autograd.Hadamard(diff, diff))
	loss.Backward()

	fmt.Printf("loss=%.6f (finite: %v)\n", loss.Value(), !hasNaNInf([]float32{loss.Value()}))
	fmt.Println("per-parameter gradient diagnostics:")
	var global float64
	for i, p := range params {
		switch {
		case p.Grad == nil:
			fmt.Printf("  [%2d] %-10s grad=nil  (parameter unused in this graph)\n", i, names[i])
		case hasNaNInf(p.Grad.Data):
			fmt.Printf("  [%2d] %-10s GRAD HAS NaN/Inf — clip/inspect before stepping\n", i, names[i])
		default:
			n := gradNorm(p.Grad.Data)
			global += n * n
			fmt.Printf("  [%2d] %-10s |grad|=%.6f\n", i, names[i], n)
		}
	}
	fmt.Printf("global gradient norm: %.6f\n", math.Sqrt(global))
}
```

Measured output (seed 42):

```
parameter inventory:
  [ 0] gleak      shape [6]        6 elems
  [ 1] vleak      shape [6]        6 elems
  [ 2] cm         shape [6]        6 elems
  [ 3] mu         shape [6 6]     36 elems
  [ 4] sigma      shape [6 6]     36 elems
  [ 5] w          shape [6 6]     36 elems
  [ 6] sMu        shape [2 6]     12 elems
  [ 7] sSigma     shape [2 6]     12 elems
  [ 8] sW         shape [2 6]     12 elems
  [ 9] inW        shape [2]        2 elems
  [10] inB        shape [2]        2 elems
  [11] outW       shape [6]        6 elems
  [12] outB       shape [6]        6 elems
  [13] readout.W  shape [6 1]      6 elems
  [14] readout.B  shape [1]        1 elems
  total 15 params, 185 trainable elements
loss=0.546932 (finite: true)
per-parameter gradient diagnostics:
  [ 0] gleak      |grad|=0.100045
  [ 1] vleak      |grad|=0.703703
  [ 2] cm         |grad|=0.027202
  [ 3] mu         |grad|=1.205924
  [ 4] sigma      |grad|=0.051544
  [ 5] w          |grad|=0.215859
  [ 6] sMu        |grad|=0.226093
  [ 7] sSigma     |grad|=0.014812
  [ 8] sW         |grad|=0.108447
  [ 9] inW        |grad|=0.168447
  [10] inB        |grad|=0.218458
  [11] outW       |grad|=0.157862
  [12] outB       |grad|=1.150598
  [13] readout.W  |grad|=0.676386
  [14] readout.B  |grad|=0.989655
global gradient norm: 2.221342
```

**Key lines:**

- `ParametersOf` returns plain `*autograd.Variable`s: `p.Data.Shape`
  and `p.Data.Data` give shape and values; the index→name mapping is
  the documented parameter order ([ltc.md](ltc.md)'s parameter table —
  the cells' `Parameters()` order is frozen by the persistence format).
- `p.Grad == nil` means the parameter did not take part in the last
  graph — expected for unused modules, a bug if the module is wired in.
- The global norm here (`2.22`) exceeds a typical `maxNorm = 1.0` clip
  threshold — exactly what recipe 2's clip would scale down.
- `hasNaNInf` is the one-line health check: scan `p.Data.Data` after a
  suspicious step and `p.Grad.Data` before stepping.

**Loss not falling? A checklist, in order of frequency:**

1. Learning rate too large (loss → `NaN`) or too small (slow creep) —
   [training.md](training.md#why-did-my-training-diverge)'s symptom
   table.
2. Missing `ZeroGrad` or double `Backward` on one graph — gradients
   exactly doubled ([pitfalls.md](pitfalls.md) §3).
3. No gradient clipping on an LTC — the `1/(den+eps)` division can
   spike gradients ~`1e16`× ([training.md](training.md#gradient-clipping-do-it)).
4. Mutated parameter `Data` between forward and `Backward` — finite
   but wrong gradients ([pitfalls.md](pitfalls.md) §4).
5. Shape bugs — the loss averages over the wrong axis, or a broadcast
   silently did something unintended
   ([shapes-and-broadcasting.md](shapes-and-broadcasting.md)).

**See also:** [faq.md](faq.md) "I got a `NaN` loss" and "why are my
gradients still growing after `Backward`?".

---

## 7. Custom losses

**Scenario:** your task needs a non-standard loss — labels with missing
entries (masked MSE) or outliers you want to be robust to (L1/MAE).
There is no loss API to learn: a loss is an ordinary op graph ending in
a scalar, and anything in [shapes-and-broadcasting.md](shapes-and-broadcasting.md)'s
op tables works.

```go
package main

import (
	"fmt"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/tensor"
)

// fit runs 300 plain-SGD epochs; lossFn builds the loss from (pred, y).
func fit(x, y *autograd.Variable, w0 float32, lossFn func(pred, y *autograd.Variable) *autograd.Variable) (float32, float32, float32) {
	w := autograd.Var(tensor.FromData([]float32{w0}, 1, 1))
	b := autograd.Var(tensor.New(1))
	const lr = 0.05
	var loss *autograd.Variable
	for epoch := 0; epoch < 300; epoch++ {
		pred := autograd.Add(autograd.MatMul(x, w), b)
		loss = lossFn(pred, y)
		w.ZeroGrad()
		b.ZeroGrad()
		loss.Backward()
		w.Data.Data[0] -= lr * w.Grad.Data[0]
		b.Data.Data[0] -= lr * b.Grad.Data[0]
	}
	return loss.Value(), w.Data.Data[0], b.Data.Data[0]
}

func main() {
	rng := rand.New(rand.NewSource(42))
	const n = 32
	xs, ys := make([]float32, n), make([]float32, n)
	mask := make([]float32, n)
	valid := 0
	for i := range xs {
		xs[i] = rng.Float32()*2 - 1
		ys[i] = 2*xs[i] + 1
		mask[i] = 1
		valid++
		if i%4 == 0 { // every 4th label is corrupt: a +50 outlier
			ys[i] += 50
			mask[i] = 0
			valid--
		}
	}
	x := autograd.Const(tensor.FromData(xs, n, 1))
	y := autograd.Const(tensor.FromData(ys, n, 1))
	m := autograd.Const(tensor.FromData(mask, n, 1)) // mask: a graph constant
	w0 := tensor.Randn(rng, 1, 1).Data[0]            // one common start for all fits
	fmt.Printf("data: y = 2x + 1, %d/%d labels corrupted by +50 outliers; w0 = %.4f\n", n-valid, n, w0)

	// (1) masked MSE — the mask is an ordinary tensor in the graph:
	// zero the corrupt rows, rescale to a mean over valid rows.
	loss, w, b := fit(x, y, w0, func(pred, y *autograd.Variable) *autograd.Variable {
		diff := autograd.Sub(pred, y)
		sq := autograd.Hadamard(diff, diff)
		return autograd.Scale(autograd.MeanAll(autograd.Hadamard(m, sq)), float32(n)/float32(valid))
	})
	fmt.Printf("masked MSE : final loss=%.6f  w=%.4f  b=%.4f   <- recovers (2, 1)\n", loss, w, b)

	// (2) control: plain MSE over all rows.
	loss, w, b = fit(x, y, w0, func(pred, y *autograd.Variable) *autograd.Variable {
		diff := autograd.Sub(pred, y)
		return autograd.MeanAll(autograd.Hadamard(diff, diff))
	})
	fmt.Printf("plain MSE  : final loss=%.6f  w=%.4f  b=%.4f   <- outliers pull the fit\n", loss, w, b)

	// (3) L1 / MAE: |pred - y| is just another op graph.
	loss, w, b = fit(x, y, w0, func(pred, y *autograd.Variable) *autograd.Variable {
		return autograd.MeanAll(autograd.Abs(autograd.Sub(pred, y)))
	})
	fmt.Printf("L1 / MAE   : final loss=%.6f  w=%.4f  b=%.4f   <- robust to outliers\n", loss, w, b)
}
```

Measured output (seed 42; the true relationship is `y = 2x + 1`):

```
data: y = 2x + 1, 8/32 labels corrupted by +50 outliers; w0 = 0.4766
masked MSE : final loss=0.000000  w=1.9999  b=1.0000   <- recovers (2, 1)
plain MSE  : final loss=461.407043  w=-2.9410  b=12.9715   <- outliers pull the fit
L1 / MAE   : final loss=12.506966  w=1.9891  b=0.9969   <- robust to outliers
```

**Key lines:**

- The mask is a `Const` tensor; `Hadamard(m, sq)` zeroes the corrupt
  rows' squared errors, and `Scale(..., n/valid)` turns the all-rows
  mean into a mean over valid rows. Masked rows receive exactly zero
  gradient — no parameter ever sees the `+50` outliers.
- L1 is `MeanAll(Abs(diff))` — `Abs`'s backward is `sign` (subgradient
  0 at exactly 0), bounded gradients regardless of outlier size: the
  L1 fit lands at `(1.9891, 0.9969)` while plain MSE is dragged to
  `(−2.9410, 12.9715)`.
- Nothing changes in the training loop: whatever scalar graph
  `lossFn` returns, `.Backward()` differentiates it.

**See also:** [shapes-and-broadcasting.md](shapes-and-broadcasting.md)
for the full op/shape tables; [pitfalls.md](pitfalls.md) §2 for the
unguarded `Log`/`Div` domains if your loss uses them (clamp `Log`
inputs, keep divisors away from 0).

---

## 8. Choosing between LTC and CfC

**Scenario:** which cell should you use? They discretize the *same*
membrane ODE with different integrators — the decision is about graph
cost, precision and convenience, not expressiveness.

| | LTC (`nn.NewLTC`) | CfC (`nn.NewCfC`) |
|---|---|---|
| integrator | semi-implicit Euler over `unfolds` substeps | Lemma 1 closed form, one analytic step |
| graph nodes per RNN step | one fused node since stage 16 (substeps no longer recorded; the kernel's stash still scales with `unfolds`) | one fused node since stage 18 — 24 nodes at any dimensions |
| backward memory | `∝ units × unfolds × seqLen` held to `Backward` (the fused kernel's stash) | `∝ units × seqLen` — no `unfolds` factor |
| large `ts` behavior | relaxes toward steady state (stable, implicit scheme) | relaxes toward steady state (exact for frozen activations) |
| variable `ts` | yes — per step ([recipe 5](#5-event-driven-sequences-with-variable-ts)) | yes — same contract |
| precision vs the ODE | first-order in `ts/unfolds`; raise `unfolds` to tighten | first-order in `ts` (Lemma 1 freezes activations); the two converge as `ts → 0` |
| constructor | `NewLTC(inDim, units, wiring, unfolds, rng)` | `NewCfC(inDim, units, wiring, rng)` — no `unfolds` |

Rule of thumb: **start with the CfC** when memory or long sequences
hurt (no `unfolds` factor at all); **choose the LTC** when you want the
reference implementation's exact Euler dynamics or are reproducing ncps
numbers. Everything else — the 13 trainable tensors and their init
ranges, fixed ±1 reversal potentials, wiring, the `ts` contract, the
`Cell` interface — is shared.

Because the parameter draw order is identical, the two cells are
drop-in swappable behind `nn.Cell` — the same loop trains either:

```go
package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/nn"
	"github.com/qorm/LNN/optimizer"
	"github.com/qorm/LNN/tensor"
)

const (
	inDim, units  = 1, 8
	seqLen, batch = 12, 16
	lr, ts        = 0.05, 1.0
	iters         = 250
)

func main() {
	// 1. same seed -> bit-identical initialization across the two cells
	// (identical parameter draw order).
	ltc := nn.NewLTC(inDim, units, nil, 4, rand.New(rand.NewSource(42)))
	cfc := nn.NewCfC(inDim, units, nil, rand.New(rand.NewSource(42)))
	pl, pc := ltc.Parameters(), cfc.Parameters()
	same := len(pl) == len(pc)
	for i := range pl {
		a, b := pl[i].Data.Data, pc[i].Data.Data
		for j := range a {
			if math.Float32bits(a[j]) != math.Float32bits(b[j]) {
				same = false
			}
		}
	}
	fmt.Printf("%d trainable tensors each; same-seed init bit-identical: %v\n", len(pl), same)

	// 2. both cells satisfy the same interfaces: one generic train
	// function drives either — swap the constructor, keep the loop.
	fmt.Printf("LTC accumulator task: final loss %.6f\n", train(ltc, rand.New(rand.NewSource(7))))
	fmt.Printf("CfC accumulator task: final loss %.6f\n", train(cfc, rand.New(rand.NewSource(7))))
}

// trainableCell is satisfied by *nn.LTC and *nn.CfC alike: the Cell
// interface (Step/StateSize) plus Module (Parameters).
type trainableCell interface {
	nn.Cell
	nn.Module
}

// train is cell-type agnostic.
func train(cell trainableCell, rng *rand.Rand) float32 {
	readout := nn.NewLinear(units, 1, rand.New(rand.NewSource(99)))
	params := nn.ParametersOf(cell, readout)
	opt := optimizer.NewSGD(lr)
	var last float32
	for it := 0; it < iters; it++ {
		xs := make([]*autograd.Variable, seqLen)
		targets := make([]*autograd.Variable, seqLen)
		state := make([]float32, batch)
		for t := 0; t < seqLen; t++ {
			xb := make([]float32, batch)
			yb := make([]float32, batch)
			for b := 0; b < batch; b++ {
				u := float32(1)
				if rng.Intn(2) == 0 {
					u = -1
				}
				xb[b] = u
				s := state[b] + 0.25*u
				if s > 1 {
					s = 1
				} else if s < -1 {
					s = -1
				}
				state[b] = s
				yb[b] = s
			}
			xs[t] = autograd.Var(tensor.FromData(xb, batch, inDim))
			targets[t] = autograd.Var(tensor.FromData(yb, batch, 1))
		}
		ys, _ := nn.Unroll(cell, xs, nil, ts)
		var acc *autograd.Variable
		for t, y := range ys {
			diff := autograd.Sub(readout.Forward(y), targets[t])
			sq := autograd.Hadamard(diff, diff)
			if t == 0 {
				acc = sq
			} else {
				acc = autograd.Add(acc, sq)
			}
		}
		loss := autograd.Scale(autograd.MeanAll(acc), 1/float32(seqLen))
		for _, p := range params {
			p.ZeroGrad()
		}
		loss.Backward()
		opt.Step(params)
		last = loss.Value()
	}
	return last
}
```

Measured output (seeds 42 / 7 / 99 — deterministic):

```
13 trainable tensors each; same-seed init bit-identical: true
LTC accumulator task: final loss 0.033956
CfC accumulator task: final loss 0.044693
```

**Key lines:**

- `nn.Cell` (Step/StateSize) is satisfied by both cells; composing it
  with `nn.Module` (Parameters) gives one interface a generic training
  loop can take — swap `NewLTC(..., unfolds, rng)` for
  `NewCfC(..., rng)` and nothing else changes.
- Same seed ⇒ bit-identical initialization across the two cells, so an
  A/B comparison starts from the same point; the small final-loss
  difference above is the two integrators, not the init.
- Memory: a fully wired cell's load-time peak is `92·U²` bytes with
  `U = units = inDim` ([persistence.md](persistence.md)); the *training*
  graph multiplies per-step cost by `seqLen` and — for the LTC only —
  by `unfolds` ([pitfalls.md](pitfalls.md) §9).

**See also:** [ltc.md](ltc.md) and [cfc.md](cfc.md) — equation-level
correspondence and the measured LTC→CfC convergence (`~O(ts²)` per
step); [faq.md](faq.md) "should I use the LTC or the CfC?".

---

## 9. Composing multi-module models

**Scenario:** a real model is several modules in series — cell → hidden
layer → readout. `ParametersOf` flattens them all into the one slice
your loop and optimizer work on; forward chaining is plain function
composition.

```go
package main

import (
	"fmt"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/nn"
	"github.com/qorm/LNN/optimizer"
	"github.com/qorm/LNN/tensor"
)

func main() {
	rng := rand.New(rand.NewSource(42))

	// Compose: CfC cell -> hidden Linear -> readout Linear.
	cell := nn.NewCfC(2, 8, nil, rng) // Step output: [batch, 8]
	hidden := nn.NewLinear(8, 16, rng)
	readout := nn.NewLinear(16, 1, rng)

	// ParametersOf flattens all three modules into one slice — that
	// slice is the optimizer's world. Order: cell's 13 tensors, then
	// hidden W/B, then readout W/B.
	params := nn.ParametersOf(cell, hidden, readout)
	fmt.Printf("modules: CfC(2->8) + Linear(8->16) + Linear(16->1)\n")
	fmt.Printf("parameters: %d tensors (%d cell + 2 + 2)\n", len(params), len(cell.Parameters()))

	// Forward chaining: every Forward is a plain op graph; compose freely.
	x := autograd.Var(tensor.Uniform(rng, -1, 1, 4, 2)) // batch 4
	y, h := cell.Step(x, nil, 1.0)                      // y: [4, 8], h: [4, 8]
	z := hidden.Forward(y)                              // z: [4, 16]
	pred := readout.Forward(z)                          // pred: [4, 1]
	fmt.Printf("shapes: x %v -> cell %v (state %v) -> hidden %v -> readout %v\n",
		x.Data.Shape, y.Data.Shape, h.Data.Shape, z.Data.Shape, pred.Data.Shape)

	// Train the whole chain on a toy target.
	opt := optimizer.NewAdamDefault(0.01)
	target := autograd.Const(tensor.FromData([]float32{0.5, -0.5, 0.25, -0.25}, 4, 1))
	for it := 0; it < 300; it++ {
		y, _ := cell.Step(x, nil, 1.0)
		pred := readout.Forward(hidden.Forward(y))
		diff := autograd.Sub(pred, target)
		loss := autograd.MeanAll(autograd.Hadamard(diff, diff))
		for _, p := range params {
			p.ZeroGrad()
		}
		loss.Backward()
		opt.Step(params)
		if it == 0 || it == 299 {
			fmt.Printf("iter %3d  loss=%.6f\n", it, loss.Value())
		}
	}
}
```

Measured output (seed 42):

```
modules: CfC(2->8) + Linear(8->16) + Linear(16->1)
parameters: 17 tensors (13 cell + 2 + 2)
shapes: x [4 2] -> cell [4 8] (state [4 8]) -> hidden [4 16] -> readout [4 1]
iter   0  loss=0.167217
iter 299  loss=0.000001
```

**Key lines:**

- `nn.ParametersOf(cell, hidden, readout)` — variadic over
  `nn.Module` (`interface{ Parameters() []*autograd.Variable }`); the
  returned 17-tensor slice is what you `ZeroGrad`, hand to `Step`, and
  (with `serialize.WriteParameters`) checkpoint. **Keep its order
  stable** across save/load ([recipe 4](#4-checkpoint-and-resume-bit-exact)).
- Forward is composition: `readout.Forward(hidden.Forward(y))`. Every
  `Forward`/`Step` call adds nodes to the current graph, so one
  `Backward` differentiates through readout, hidden layer and all cell
  dynamics at once.
- Shapes flow `[batch, inDim] → [batch, units] → [batch, 16] → [batch, 1]`;
  `Linear.Forward` takes `[batch, in]` and returns `[batch, out]`.

**See also:** [training.md](training.md#aggregating-parameters-from-nn-modules)
for `ParametersOf` and the nil-`Grad` rule for unused modules;
[architecture.md](architecture.md) for what a graph node is.

---

## 10. Loading untrusted model files safely

**Scenario:** a model file arrives from outside your process — a
download, an upload, another team's checkpoint. Treat it as hostile
input: every load-path failure is an `error` (never a panic), and the
message tells you *which kind* of failure it is. Classify and react.

```go
package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"strings"

	"github.com/qorm/LNN/nn"
)

// classify maps a load-path error onto the operator-facing bucket. Every
// failure here is an error, never a panic — that is the persistence
// contract (doc/persistence.md).
func classify(err error) string {
	switch {
	case err == nil:
		return "ok"
	case errors.Is(err, io.ErrUnexpectedEOF):
		return "truncated stream (interrupted write / bad copy) — safe to retry"
	case strings.Contains(err.Error(), "model kind"):
		return "wrong loader for this file (kind mismatch) — use the matching LoadXxx"
	case strings.Contains(err.Error(), "unsupported format version"):
		return "format version skew — newer writer or corruption; update the library"
	case strings.Contains(err.Error(), "bad magic"):
		return "not an LNN stream at all — wrong file"
	case strings.Contains(err.Error(), "load limit"):
		return "model exceeds the load-path caps — hostile or oversized header"
	default:
		return "other validation failure — inspect the message"
	}
}

func attempt(name string, f func() error) {
	err := f()
	fmt.Printf("%-28s -> %-28s (%v)\n", name, classify(err), err)
}

func main() {
	// A legitimate CfC stream, to cross-load and to corrupt.
	var buf bytes.Buffer
	cell := nn.NewCfC(2, 4, nil, rand.New(rand.NewSource(1)))
	if err := nn.SaveCfC(&buf, cell); err != nil {
		panic(err)
	}
	raw := buf.Bytes()
	fmt.Printf("legit CfC stream: %d bytes\n\n", len(raw))

	attempt("garbage bytes", func() error {
		_, err := nn.LoadCfC(bytes.NewReader([]byte("this is not a model file..")))
		return err
	})
	attempt("empty file", func() error {
		_, err := nn.LoadCfC(bytes.NewReader(nil))
		return err
	})
	attempt("truncated at half", func() error {
		_, err := nn.LoadCfC(bytes.NewReader(raw[:len(raw)/2]))
		return err
	})
	attempt("LTC loader on CfC file", func() error {
		_, err := nn.LoadLTC(bytes.NewReader(raw))
		return err
	})
	attempt("Linear loader on CfC file", func() error {
		_, err := nn.LoadLinear(bytes.NewReader(raw))
		return err
	})

	badVer := append([]byte(nil), raw...)
	badVer[13] = 99 // version byte of the embedded tensor stream
	attempt("corrupt version byte", func() error {
		_, err := nn.LoadCfC(bytes.NewReader(badVer))
		return err
	})

	// Craft an LTC header claiming units = 4096: kind + 3 int32s + blob.
	// The header cap check fires BEFORE the blob is parsed.
	huge := make([]byte, 1+12+9)
	huge[0] = 0                                   // kind LTC
	binary.LittleEndian.PutUint32(huge[1:], 4)    // inDim
	binary.LittleEndian.PutUint32(huge[5:], 4096) // units
	binary.LittleEndian.PutUint32(huge[9:], 4)    // unfolds
	copy(huge[13:], []byte("LNNS"))
	huge[17] = 1 // version
	attempt("units=4096 header", func() error {
		_, err := nn.LoadLTC(bytes.NewReader(huge))
		return err
	})
	hugeU := make([]byte, 1+12+9)
	hugeU[0] = 0
	binary.LittleEndian.PutUint32(hugeU[1:], 4)
	binary.LittleEndian.PutUint32(hugeU[5:], 8)
	binary.LittleEndian.PutUint32(hugeU[9:], 4096) // unfolds
	copy(hugeU[13:], []byte("LNNS"))
	hugeU[17] = 1
	attempt("unfolds=4096 header", func() error {
		_, err := nn.LoadLTC(bytes.NewReader(hugeU))
		return err
	})

	// The happy path still works.
	attempt("legit CfC stream", func() error {
		_, err := nn.LoadCfC(bytes.NewReader(raw))
		return err
	})
}
```

Measured output:

```
legit CfC stream: 827 bytes

garbage bytes                -> wrong loader for this file (kind mismatch) — use the matching LoadXxx (nn: stream holds model kind 116 (unknown), not CfC (kind 1))
empty file                   -> truncated stream (interrupted write / bad copy) — safe to retry (nn: reading model kind: unexpected EOF)
truncated at half            -> truncated stream (interrupted write / bad copy) — safe to retry (serialize: tensor 7: truncated stream: claims 64 data bytes but only 11 remain: unexpected EOF)
LTC loader on CfC file       -> wrong loader for this file (kind mismatch) — use the matching LoadXxx (nn: stream holds model kind 1 (CfC), not LTC (kind 0))
Linear loader on CfC file    -> wrong loader for this file (kind mismatch) — use the matching LoadXxx (nn: stream holds model kind 1 (CfC), not Linear (kind 2))
corrupt version byte         -> format version skew — newer writer or corruption; update the library (serialize: unsupported format version 99 (this build reads version 1): the stream was written by a newer version of the library; update this build to read it)
units=4096 header            -> model exceeds the load-path caps — hostile or oversized header (nn: LTC header has units=4096, exceeding the load limit 2048)
unfolds=4096 header          -> model exceeds the load-path caps — hostile or oversized header (nn: LTC header has unfolds=4096, exceeding the load limit 1024)
legit CfC stream             -> ok                           (<nil>)
```

**Key lines:**

- Truncation wraps `io.ErrUnexpectedEOF` — match it with `errors.Is`,
  not string matching. Everything else is classified by its message
  prefix (`nn:` model level, `serialize:` stream level, `optimizer:`
  state level).
- Cross-loading is a *named* error, not a misparse: the loader reads
  the one-byte kind tag first and tells you what the file actually
  holds ("stream holds model kind 1 (CfC)").
- Header caps are checked **before** the tensor blob is parsed: a
  22-byte stream claiming `units = 4096` costs a handful of
  allocations, not ~1.4 GiB. Load-path limits: `units`/`inDim ≤ 2048`,
  `unfolds ≤ 1024` (constructors are not capped — your own allocation
  decision).
- A failing load has **zero side effects**: on a destination model,
  every shape is validated before any value is copied, so the model
  you already have stays exactly as it was. Version skew refuses
  rather than guesses (`version 1` is the only layout this build
  reads; higher versions say "update this build", lower say
  "corrupt or forged").

**See also:** [persistence.md](persistence.md#the-untrusted-stream-safety-contract)
for the full safety contract (limits table, both reader classes,
fuzzing evidence) and [pitfalls.md](pitfalls.md) §10 for the
user-facing summary; [faq.md](faq.md) "how do I read load errors?".

---

## 11. Annealing the learning rate mid-training

**Scenario:** start fast, finish fine — change the learning rate during
training. Every hyperparameter is a plain exported struct field; write
it, and the next `Step` uses the new value. No scheduler object exists.

```go
package main

import (
	"fmt"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/optimizer"
	"github.com/qorm/LNN/tensor"
)

func main() {
	rng := rand.New(rand.NewSource(42))

	const n = 32
	xs, ys := make([]float32, n), make([]float32, n)
	for i := range xs {
		xs[i] = rng.Float32()*2 - 1
		ys[i] = 2*xs[i] + 1
	}
	x := autograd.Const(tensor.FromData(xs, n, 1))
	y := autograd.Const(tensor.FromData(ys, n, 1))

	w := autograd.Var(tensor.Randn(rng, 1, 1))
	b := autograd.Var(tensor.New(1))
	params := []*autograd.Variable{w, b}

	adam := optimizer.NewAdamDefault(0.1) // hyperparameters are exported fields
	const epochs = 400
	for epoch := 0; epoch < epochs; epoch++ {
		pred := autograd.Add(autograd.MatMul(x, w), b)
		diff := autograd.Sub(pred, y)
		loss := autograd.MeanAll(autograd.Hadamard(diff, diff))
		for _, p := range params {
			p.ZeroGrad()
		}
		loss.Backward()
		adam.Step(params)

		// Anneal: writing the field takes effect from the next Step.
		if epoch == 200 {
			adam.LR = 0.01
		}
		if epoch == 300 {
			adam.LR = 0.001
		}

		switch epoch {
		case 0, 99, 199, 201, 299, 301, 399:
			fmt.Printf("epoch %3d  LR=%.3f  loss=%.6f  w=%.6f  b=%.6f\n",
				epoch, adam.LR, loss.Value(), w.Data.Data[0], b.Data.Data[0])
		}
	}
}
```

Measured output (seed 42):

```
epoch   0  LR=0.100  loss=1.398637  w=0.576583  b=0.100000
epoch  99  LR=0.100  loss=0.000009  w=1.995972  b=0.999955
epoch 199  LR=0.100  loss=0.000000  w=1.999978  b=0.999999
epoch 201  LR=0.010  loss=0.000000  w=1.999991  b=0.999996
epoch 299  LR=0.010  loss=0.000000  w=2.000002  b=1.000001
epoch 301  LR=0.001  loss=0.000000  w=2.000002  b=1.000001
epoch 399  LR=0.001  loss=0.000000  w=2.000002  b=1.000001
```

**Key lines:**

- `adam.LR = 0.01` is the entire schedule — the fields `LR`,
  `Beta1`/`Beta2`/`Eps` (Adam), `LR`/`Mu` (Momentum), `LR` (SGD) are
  exported and read fresh on every `Step`.
- The same works for any hand-rolled rule: schedule your `lr` constant
  however you like — the loop is yours.
- Trust model: constructors validate (`NewAdam` panics on `LR ≤ 0`),
  but `Step` trusts the fields as written — writing `NaN` into `LR`
  gets you precisely the arithmetic you asked for.

**See also:** [training.md](training.md#hyperparameters-are-exported-fields)
(the anneal pattern and its measured results, which this recipe
reproduces); recipe 4 — hyperparameters are deliberately *not* in the
`"LNO1"` state stream, so set them on the destination optimizer when
resuming.

---

## 12. Deterministic reproduction

**Scenario:** make a run reproducible — for debugging, for papers, for
regression tests. The rule is strict: same seeds ⇒ bit-identical
trajectories, compared at the `Float32bits` level.

```go
package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/nn"
	"github.com/qorm/LNN/optimizer"
	"github.com/qorm/LNN/tensor"
)

func run(seed int64) (uint32, uint32) {
	rng := rand.New(rand.NewSource(seed))
	cell := nn.NewCfC(1, 6, nil, rng)
	readout := nn.NewLinear(6, 1, rng)
	params := nn.ParametersOf(cell, readout)
	opt := optimizer.NewAdamDefault(0.01)
	data := rand.New(rand.NewSource(seed + 1))
	var last uint32
	for it := 0; it < 120; it++ {
		xs := make([]*autograd.Variable, 10)
		targets := make([]*autograd.Variable, 10)
		state := make([]float32, 8)
		for t := 0; t < 10; t++ {
			xb := make([]float32, 8)
			yb := make([]float32, 8)
			for b := 0; b < 8; b++ {
				u := float32(1)
				if data.Intn(2) == 0 {
					u = -1
				}
				xb[b] = u
				s := state[b] + 0.25*u
				if s > 1 {
					s = 1
				} else if s < -1 {
					s = -1
				}
				state[b] = s
				yb[b] = s
			}
			xs[t] = autograd.Var(tensor.FromData(xb, 8, 1))
			targets[t] = autograd.Var(tensor.FromData(yb, 8, 1))
		}
		ys, _ := nn.Unroll(cell, xs, nil, 1.0)
		var acc *autograd.Variable
		for t, y := range ys {
			diff := autograd.Sub(readout.Forward(y), targets[t])
			sq := autograd.Hadamard(diff, diff)
			if t == 0 {
				acc = sq
			} else {
				acc = autograd.Add(acc, sq)
			}
		}
		loss := autograd.Scale(autograd.MeanAll(acc), 1/10.0)
		for _, p := range params {
			p.ZeroGrad()
		}
		loss.Backward()
		opt.Step(params)
		last = math.Float32bits(loss.Value())
	}
	return last, math.Float32bits(params[0].Data.Data[0])
}

func main() {
	l1, p1 := run(42)
	l2, p2 := run(42) // same seeds again
	l3, _ := run(43)  // different seed

	fmt.Printf("run A (seed 42): final loss bits %08x, param[0] bits %08x\n", l1, p1)
	fmt.Printf("run B (seed 42): final loss bits %08x, param[0] bits %08x\n", l2, p2)
	fmt.Printf("same seed, two runs: bit-identical: %v\n", l1 == l2 && p1 == p2)
	fmt.Printf("run C (seed 43): final loss bits %08x — differs: %v\n", l3, l3 != l1)
}
```

Measured output:

```
run A (seed 42): final loss bits 3c5dd2c0, param[0] bits 3f1dd476
run B (seed 42): final loss bits 3c5dd2c0, param[0] bits 3f1dd476
same seed, two runs: bit-identical: true
run C (seed 43): final loss bits 3c7f5455 — differs: true
```

**Key lines:**

- Seed discipline means *every* `rand.Rand` in the run is seeded —
  model initialization and data generation separately (as above), so
  changing one stream doesn't perturb the other.
- Compare bit patterns (`math.Float32bits`, `NaN`/`−0` included), not
  printed decimals: two runs can print the same `%.6f` and still
  differ in the last bit.
- Same seed ⇒ identical RNG streams ⇒ identical graphs ⇒ identical
  `float32` arithmetic, on a given platform and toolchain
  ([pitfalls.md](pitfalls.md) §7).

**See also:** [persistence.md](persistence.md) "Golden vectors" — the
cross-platform nuance: the format layout is byte-frozen everywhere,
but float payloads can differ by ≤ 1 ULP per fused multiply-add across
architectures (arm64 vs amd64), so the golden tests assert a 16 ULP
window off the generating architecture; [faq.md](faq.md) "do last-digit
differences across machines matter?".

---

## 13. Long-sequence training: chunked BPTT with UnrollRemat

**Scenario:** the sequence grows long enough that "the whole graph stays
resident until `Backward`" becomes the memory wall — at T = 512 a full
unroll pins ~11.5 MB of live graph where `UnrollRemat` retains ~0.65 MB
(~18×; `BenchmarkUnrollPeakMemory512` /
`BenchmarkUnrollRematPeakMemory512`, chunk 16, units 8, batch 16).
`nn.UnrollRemat` differentiates `lossFn(ys)` through time with
**bit-identical gradients** to `Unroll` + `loss.Backward()`, at
O(chunkSize) peak graph memory instead of O(len(xs)). The program below
proves the bit-identity for both cells, then trains with it.

```go
package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/nn"
	"github.com/qorm/LNN/optimizer"
	"github.com/qorm/LNN/tensor"
)

const (
	inDim, units  = 1, 8
	seqLen, batch = 48, 16
	chunk         = 8
	lr, ts        = 0.01, 1.0
	iters         = 250
)

// makeBatch draws one fresh bounded-accumulator batch:
// s_t = clip(s_{t-1} + 0.25*u_t, -1, 1), u_t in {-1, +1}.
func makeBatch(rng *rand.Rand) (xs, targets []*autograd.Variable) {
	xs = make([]*autograd.Variable, seqLen)
	targets = make([]*autograd.Variable, seqLen)
	state := make([]float32, batch)
	for t := 0; t < seqLen; t++ {
		xb := make([]float32, batch)
		yb := make([]float32, batch)
		for b := 0; b < batch; b++ {
			u := float32(1)
			if rng.Intn(2) == 0 {
				u = -1
			}
			xb[b] = u
			s := state[b] + 0.25*u
			if s > 1 {
				s = 1
			} else if s < -1 {
				s = -1
			}
			state[b] = s
			yb[b] = s
		}
		xs[t] = autograd.Var(tensor.FromData(xb, batch, inDim))
		targets[t] = autograd.Var(tensor.FromData(yb, batch, 1))
	}
	return xs, targets
}

// mseLoss is the per-step MSE readout loss. It is spelled so the loss
// graph's DFS visits the step outputs in ascending order (t = 0, 1, 2,
// ...) — the remat fast path; a descending spelling is the documented
// worst case (see the trade-off table in the recipe).
func mseLoss(readout *nn.Linear, targets []*autograd.Variable) func(ys []*autograd.Variable) *autograd.Variable {
	return func(ys []*autograd.Variable) *autograd.Variable {
		var acc *autograd.Variable
		for t, y := range ys {
			diff := autograd.Sub(readout.Forward(y), targets[t])
			sq := autograd.Hadamard(diff, diff)
			if t == 0 {
				acc = sq
			} else {
				acc = autograd.Add(acc, sq)
			}
		}
		return autograd.Scale(autograd.MeanAll(acc), 1/float32(len(ys)))
	}
}

// bitIdentical runs one whole-graph backward (the reference) and one
// UnrollRemat call over the same cell, data and loss, and reports
// whether loss value and every parameter gradient agree bit for bit.
func bitIdentical(cell interface {
	nn.Cell
	nn.Module
}, rng, data *rand.Rand) bool {
	readout := nn.NewLinear(units, 1, rng)
	params := nn.ParametersOf(cell, readout)
	xs, targets := makeBatch(data)
	lossFn := mseLoss(readout, targets)

	for _, p := range params {
		p.ZeroGrad()
	}
	ys, _ := nn.Unroll(cell, xs, nil, ts)
	ref := lossFn(ys)
	ref.Backward()
	refGrads := make([][]float32, len(params))
	for i, p := range params {
		refGrads[i] = append([]float32(nil), p.Grad.Data...)
	}

	for _, p := range params {
		p.ZeroGrad()
	}
	_, _, rmLoss := nn.UnrollRemat(cell, params, xs, nil, ts, chunk, lossFn)
	if math.Float32bits(ref.Value()) != math.Float32bits(rmLoss.Value()) {
		return false
	}
	for i, p := range params {
		for j := range p.Grad.Data {
			if math.Float32bits(p.Grad.Data[j]) != math.Float32bits(refGrads[i][j]) {
				return false
			}
		}
	}
	return true
}

func main() {
	// 1. Bit-identity against the whole-graph backward, for both cells
	//    (the LTC exercises the extra sigma sweep of its spine class).
	ltc := nn.NewLTC(inDim, units, nil, 4, rand.New(rand.NewSource(42)))
	cfc := nn.NewCfC(inDim, units, nil, rand.New(rand.NewSource(42)))
	fmt.Printf("T=%d chunk=%d, remat vs whole-graph backward, bit-identical: LTC %v, CfC %v\n",
		seqLen, chunk,
		bitIdentical(ltc, rand.New(rand.NewSource(1)), rand.New(rand.NewSource(7))),
		bitIdentical(cfc, rand.New(rand.NewSource(1)), rand.New(rand.NewSource(7))))

	// 2. A real training loop with UnrollRemat: same four-phase
	//    discipline, the loss just comes back already backpropagated.
	rng := rand.New(rand.NewSource(42))
	cell := nn.NewCfC(inDim, units, nil, rng)
	readout := nn.NewLinear(units, 1, rng)
	params := nn.ParametersOf(cell, readout) // every trainable leaf — audited
	opt := optimizer.NewAdamDefault(lr)
	data := rand.New(rand.NewSource(7))
	var first, last float64
	for it := 0; it < iters; it++ {
		xs, targets := makeBatch(data)
		for _, p := range params {
			p.ZeroGrad()
		}
		_, _, loss := nn.UnrollRemat(cell, params, xs, nil, ts, chunk, mseLoss(readout, targets))
		opt.Step(params)
		if it == 0 {
			first = float64(loss.Value())
		}
		last = float64(loss.Value())
		if it%50 == 0 || it == iters-1 {
			fmt.Printf("iter %3d  loss=%.6f\n", it, loss.Value())
		}
	}
	fmt.Printf("first=%.6f last=%.6f\n", first, last)
}
```

Measured output (seeds 42 / 1 / 7 — deterministic):

```
T=48 chunk=8, remat vs whole-graph backward, bit-identical: LTC true, CfC true
iter   0  loss=0.750716
iter  50  loss=0.075355
iter 100  loss=0.016325
iter 150  loss=0.012297
iter 200  loss=0.015780
iter 249  loss=0.010747
first=0.750716 last=0.010747
```

**Key lines:**

- `params := nn.ParametersOf(cell, readout)` must list **every**
  trainable leaf the cell's `Step` consumes: the structural probe
  audits completeness and panics naming the missing index — an
  unlisted leaf would silently accumulate one extra gradient copy per
  sweep (a 2–3× wrong value). Loss-only leaves (the readout's) may be
  listed but need not be; `xs`/`h0` gradients always participate.
- `lossFn` receives the **detached** per-step outputs (bit-identical
  values, no graph behind them) and is called exactly once; its return
  value comes back already backpropagated — the gradients sit in the
  leaves as after `loss.Backward()`, so the `ZeroGrad` → `Step`
  discipline is unchanged. The returned `ys`/`hN` are detached too:
  safe to read and to feed into further computation, impossible to
  differentiate through (that already happened).
- "Bit-identical" is asserted with `math.Float32bits`, not printed
  decimals — two runs can print the same `%.6f` and differ in the last
  bit (recipe 12). T = 48 with chunk = 8 cuts six recompute units.
- The LTC check exercises the σ sweep its spine-class `cm` requires;
  the CfC takes the rest sweep alone (two forwards, one backward — the
  ideal remat price). Both match the whole-graph backward bit for bit.

**Choosing `chunkSize`:** peak graph memory scales ~linearly with
`chunkSize` (chunk 8 ≈ half the chunk-16 figure above), while the O(T)
detached outputs/states are tiny (one `[batch, units]` tensor per
step), so smaller chunks cost only per-unit fixed overhead — 4–16 is
the sane band on this engine. One structural caveat: a seeded step that
is *not* a record high of the loss's visit order glues itself to its
successor, merging recompute units beyond `chunkSize`. Spell the loss
so the outputs are visited in ascending step order (the natural
accumulation order above) and units stay exactly `chunkSize`; a
descending visit order with a small chunk is the extreme worst case —
everything merges into one O(T) unit and `UnrollRemat` then costs
strictly more peak memory AND more compute than one whole-graph
backward (the price of bitwise fidelity, not a bug). A regularizer
closing over parameters is legal (the loss is called exactly once) but
must be spelled data-first — `Add(data, penalty)`, never
`Add(penalty, data)` — or made a constant; the probe panics on a
violation.

| | `Unroll` + `loss.Backward()` | `UnrollRemat` |
|---|---|---|
| peak graph memory | O(T × per-step graph) — T = 512: ~11.5 MB live | O(chunkSize × per-step graph) + O(T) small tensors — T = 512, chunk 16: ~0.65 MB retained |
| compute per iteration | 1 forward + 1 backward | CfC: 2 forwards + 1 backward; LTC: 3 forwards + 2 backwards (σ sweep); a non-ascending loss adds one affine pass for either cell |
| gradients | the reference | bit-identical to the reference (both cells, above) |
| loss shape | anything | ascending visit order is the fast path; adversarial orders force unit merges (worst case can exceed full unroll) |
| `params` argument | — | must list every `Step`-consumed trainable leaf; audited, panics |
| cell requirement | any `nn.Cell` | per-step graph structure must be a pure function of `(x, h)` — both provided cells qualify; a value-dependent branch drifts ~1–2 ULP undetectably ([pitfalls.md](pitfalls.md)) |
| use when | the default; short-to-medium sequences | long sequences, or memory-tight deployments |

**See also:** [architecture.md](architecture.md) for the two-pass
mechanism and the three fold classes it replays;
[pitfalls.md](pitfalls.md) for the archived residual corners (worst
case, value-dependent cells, the double-NaN payload corner); the
`nn.UnrollRemat` godoc for the full contract; [api.md](api.md) for the
one-line entry.
