# Training models with lnn

> English | [中文](zh/training.md)

**Summary:** training with lnn is a four-phase loop (zero grads → forward
→ `Backward` → parameter update) over `Parameters()`, and you own
stability (learning rate and gradient clipping). The update phase is
either five hand-rolled lines of Go — the basis for understanding the
engine — or one `optimizer.Step` call (the `optimizer` package ships
SGD/Momentum/Adam), the recommended form for production training. This
guide shows both end to end.

**Audience:** engineers about to train their first model with the library.

## The loop

Every training iteration is:

```
1. ZeroGrad every parameter            // grads accumulate; always reset
2. forward: build a fresh graph        // ops are eager, the graph is recorded
3. loss.Backward()                     // one Backward per graph
4. update parameters in place          // plain Go over p.Data / p.Grad
```

Parameters are leaf `*autograd.Variable`s; updating them means writing
their `Data` buffers directly. A complete, runnable program (fits
`y = 2x + 1` with a hand-rolled linear model):

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

	const n = 32
	xs, ys := make([]float32, n), make([]float32, n)
	for i := range xs {
		xs[i] = rng.Float32()*2 - 1
		ys[i] = 2*xs[i] + 1
	}
	x := autograd.Const(tensor.FromData(xs, n, 1)) // Const documents "not trainable"
	y := autograd.Const(tensor.FromData(ys, n, 1))

	w := autograd.Var(tensor.Randn(rng, 1, 1))
	b := autograd.Var(tensor.New(1))

	const epochs, lr = 200, 0.1
	for epoch := 0; epoch < epochs; epoch++ {
		pred := autograd.Add(autograd.MatMul(x, w), b)
		diff := autograd.Sub(pred, y)
		loss := autograd.MeanAll(autograd.Hadamard(diff, diff))

		w.ZeroGrad()
		b.ZeroGrad()
		loss.Backward()

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

Actual output (Go 1.26, seed 42 — deterministic):

```
epoch   0  loss=1.398637  w=0.5503  b=0.1674
epoch  40  loss=0.006162  w=1.8658  b=0.9795
epoch  80  loss=0.000047  w=1.9883  b=0.9982
epoch 120  loss=0.000000  w=1.9990  b=0.9998
epoch 160  loss=0.000000  w=1.9999  b=1.0000
epoch 199  loss=0.000000  w=2.0000  b=1.0000
```

Three disciplines that the engine does not enforce for you:

- **Build a fresh graph every iteration** (the loop above re-runs the
  forward ops). Calling `Backward` twice on the *same* graph is defined
  but doubles leaf gradients — see [pitfalls.md](pitfalls.md).
- **`ZeroGrad` before, never after, `Backward`** — gradients accumulate
  into leaves across calls and across graphs by design.
- **Do not mutate parameter `Data` between forward and `Backward`:** a
  few backward steps read parent data at backward time, so in-place
  updates belong strictly after the backward pass.

## Aggregating parameters from nn modules

`nn.Module` is `interface{ Parameters() []*autograd.Variable }`, and
`nn.ParametersOf` flattens several modules into one slice — that slice *is*
your optimizer state:

```go
cell := nn.NewLTC(1, 8, nil, 4, rng)      // 13 trainable parameter tensors
readout := nn.NewLinear(8, 1, rng)        // + W and B
params := nn.ParametersOf(cell, readout)  // 15 variables

for it := 0; it < iters; it++ {
	// ... build xs, unroll, compute loss ...
	for _, p := range params {
		p.ZeroGrad()
	}
	loss.Backward()
	for _, p := range params {
		if p.Grad == nil { // a parameter unused in this graph has nil grad
			continue
		}
		for i := range p.Data.Data {
			p.Data.Data[i] -= lr * p.Grad.Data[i]
		}
	}
}
```

`examples/ltc-sequence` is this pattern complete, on a task that requires
cross-step memory (a bounded accumulator): `go run ./examples/ltc-sequence`
trains for 250 iterations and prints the loss falling from `0.690761` to
`0.041996`. The examples live in the repository — clone it
(`git clone https://github.com/qorm/LNN.git`) and run from the repository
root.

## Using the optimizer package

The `optimizer` package packages phase 4 of the loop — exactly the
hand-rolled update above, same `float32` arithmetic, same in-place writes
to `p.Data` — as one auditable method call. Three explicit structs, no
configuration objects, no reflection:

| Optimizer | Constructor | Update rule |
|---|---|---|
| `SGD` | `optimizer.NewSGD(lr)` | `p -= LR·g` — the hand-rolled loop itself |
| `Momentum` | `optimizer.NewMomentum(lr, mu)` | `v = Mu·v + g`, then `p -= LR·v` — velocity stores *unscaled* gradients, exactly the "Optional: momentum" snippet below |
| `Adam` | `optimizer.NewAdam(lr, beta1, beta2, eps)` or `optimizer.NewAdamDefault(lr)` (Kingma & Ba's 0.9 / 0.999 / 1e-8) | bias-corrected first/second moment estimates |

All implement `optimizer.Optimizer` (`Step(params []*autograd.Variable)`),
constructors validate their arguments and panic with the offending value,
and `SGD` reproduces the Quick start loop at the top of this guide
line-for-line — measured: identical output at every printed epoch,
recovering `w = 2.0000, b = 1.0000` at epoch 199 with seed 42.

The loop with `Step`:

```go
params := nn.ParametersOf(cell, readout)
opt := optimizer.NewAdamDefault(0.01)

for it := 0; it < iters; it++ {
	for _, p := range params {
		p.ZeroGrad()
	}
	loss := ...       // build a fresh graph
	loss.Backward()   // one Backward per graph
	opt.Step(params)  // update in place
}
```

### The Step contract

- **`Step` never calls `ZeroGrad`.** Leaf gradients accumulate across
  `Backward` calls by design, and when to reset them is *your* contract.
  Zero before every iteration for plain training; zero once every `N`
  iterations while stepping after each `Backward`, and you get gradient
  accumulation over `N` micro-batches for free — an optimizer that zeroed
  on your behalf would silently break that pattern.
- **Parameters with nil `Grad` are skipped.** A parameter unused in the
  last graph (e.g. an unused module handed over by `nn.ParametersOf`)
  keeps its `Data` and — for stateful optimizers — its state. Adam's
  per-parameter step counter does not advance either, so the parameter's
  first real update carries exactly the bias correction of a fresh
  optimizer.
- `Step` assumes `p.Grad` has the same shape as `p.Data`, which
  autograd's `addGrad` guarantees.

### Hyperparameters are exported fields

Every hyperparameter is a plain exported struct field — read and write it
directly; adjusting the learning rate mid-training is a supported
pattern:

```go
adam := optimizer.NewAdamDefault(0.1)
// ...
if epoch == 200 {
	adam.LR = 0.01 // anneal: in effect from the next Step
}
```

Measured on the quick-start fit (seed 42): Adam annealed `0.1 → 0.01 →
0.001` at epochs 200/300 reaches loss `0.000009` at epoch 99 and
`w = 2.0000, b = 1.0000` by epoch 199.

Constructors validate; `Step` trusts field values as written — the same
trust model as a hand-rolled loop trusting its `lr` constant. So
`optimizer.NewSGD(+Inf)` is *accepted* (it satisfies `lr > 0`; every step
then produces `±Inf`, or `NaN` where a gradient element is exactly zero),
and writing a nonsense value into `adam.LR` after construction gets you
precisely the arithmetic you asked for.

### State is keyed by pointer identity

`Momentum` and `Adam` keep per-parameter state (velocity; Adam's moment
buffers and step count) in maps keyed by `*autograd.Variable` pointer.
Consequences:

- The same variable stepped repeatedly accumulates its state; distinct
  variables never share state, even if identically shaped.
- The state maps *pin* every variable they have ever seen (map keys are
  strong references): discard the optimizer when you discard a model.
- Re-pointing a variable's `Data` at a new **same-sized** tensor keeps
  its state — the optimizer still sees the same parameter. **Resizing**
  a parameter between steps panics rather than silently corrupting the
  update.
- **Aliased variables couple.** Two `Variable`s built over the *same*
  `Tensor` (sharing one `Data` slice) are distinct map keys but one
  buffer: stepping both applies each update to the shared storage —
  measured at SGD `LR = 0.1` with unit gradients, the value moves by
  `0.2`, not `0.1`. Treat aliased variables as one parameter and step it
  once.

### Numerics

`float32` everywhere, including optimizer state. Adam's update is
self-normalizing (`m'/sqrt(v')` stays bounded near `±1` regardless of
gradient scale), so no wide-magnitude sum ever forms and the `float64`
trick used for the global gradient norm in the clipping section below
does not apply here. Adam's square root goes through `math.Sqrt` — the
standard library has no `float32` sqrt, and one correctly-rounded
conversion per element is not an accumulation, so it cannot drift.

## Why the hand-rolled loop remains

The library's contract is a small, readable, auditable numeric core; an
update rule is five lines of Go, and writing it yourself keeps lr
schedules, clipping and regularization in your code, visible and
diffable, rather than hidden behind a framework abstraction. Nothing in
the graph engine assumes how leaves are updated. The `optimizer` package
(above) packages precisely this loop for the three common rules and is
the recommended form for production training; the hand-rolled version
stays the basis for understanding what `Step` does — and for every
update rule the package does not cover (weight-decay variants, exotic
schedules). Both forms share one discipline: clipping remains
caller-owned (next section).

## Gradient clipping: do it

Plain `float32` SGD has zero built-in stabilization, and the LTC in
particular can produce large gradient spikes (its ODE denominator is
guarded only by `eps = 1e-8`, and the division gradient scales as `1/b²` —
a `1e-8` divisor amplifies gradients to ~`1e16`). A single spike can push
parameters into overflow territory from which `float32` does not recover.
Clip the **global gradient norm** on anything beyond a toy problem —
exactly as `examples/ltc-sequence` does:

```go
const lr, maxNorm = 0.05, 1.0

// After loss.Backward(): scale the step so the global gradient norm
// never exceeds maxNorm.
var norm2 float64
for _, p := range params {
	if p.Grad == nil {
		continue
	}
	for _, g := range p.Grad.Data {
		norm2 += float64(g) * float64(g) // accumulate in float64
	}
}
scale := lr
if norm := math.Sqrt(norm2); norm > maxNorm {
	scale = lr * maxNorm / norm
}
for _, p := range params {
	if p.Grad == nil {
		continue
	}
	for i := range p.Data.Data {
		p.Data.Data[i] -= float32(scale) * p.Grad.Data[i]
	}
}
```

Measured in `examples/ltc-sequence` (seed 42, units=8, unfolds=4, 12-step
sequence): gradient norm `2.50` → update scale `0.019975` instead of
`0.05` on the first iteration, with a maximum observed norm of `6.04`
over the full 250-iteration run. The clip engages on most early
iterations and is the difference between converging and diverging on
this task.

With `optimizer.Step`, clip the same way but rescale the gradients
themselves — `Step` applies its own learning rate, so accumulate the
norm in `float64` and, when it exceeds `maxNorm`, multiply every element
of every `p.Grad.Data` by `maxNorm/norm` **before** the `Step` call:

```go
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
```

The complete train → save → load → resume example in
[persistence.md](persistence.md) shows exactly this form in a full
program.

### Optional: momentum

Momentum is the same five lines with a velocity buffer; nothing else
changes:

```go
vel := tensor.New(1) // one velocity buffer per parameter, same shape
const beta = 0.9

// inside the loop, replacing the plain SGD update:
for i := range w.Data.Data {
	vel.Data[i] = beta*vel.Data[i] + w.Grad.Data[i]
	w.Data.Data[i] -= lr * vel.Data[i]
}
```

Verified: minimizing `(w - 5)²` from a random start with `lr=0.05`,
`beta=0.9` reaches `w = 5.0007` in 150 iterations. Note that momentum
stores *unscaled* gradients in `vel`; if you combine it with norm
clipping, apply the same `scale` to the gradient before adding it to the
velocity, or clip the velocity itself — pick one and be consistent.

`optimizer.NewMomentum(0.05, 0.9)` is this snippet packaged: it stores
unscaled gradients in its velocity buffer with the same arithmetic, and
reaches the same `w = 5.0007` (the library's own
`TestMomentumMatchesDocExample` regression-tests the two against each
other).

## Why did my training diverge?

A checklist ordered by how often each cause appears:

| symptom | likely cause | fix |
|---|---|---|
| loss → `NaN` within a few steps | lr too large; parameters overshoot into `Exp`/`Log` overflow | reduce lr 3–10x; add norm clipping |
| sudden `NaN` after many good steps | a `Log(x)` saw `x ≤ 0`, or a `Div` saw a zero divisor (`+Inf` forward, `Inf` gradients backward) | clamp inputs to `Log`; keep divisors bounded away from 0 |
| LTC loss spikes/diverges | gradient spike from the `1/(den+eps)` division (`den` can approach `eps = 1e-8`) | global norm clipping (above); smaller lr |
| everything `NaN` after a step | `float32` overflow propagated: once one element is `NaN`, MatMul-free paths carry it through the whole graph | clip; check `ts` is sane (anchor it to the task's sampling interval — one time unit per step means `ts = 1.0`; `ts ≥ 1e-3` keeps full physical fidelity — see [ltc.md](ltc.md)) |
| gradients exactly 2x too big | `Backward` called twice on the same graph, or `ZeroGrad` missing | one `Backward` per fresh graph; `ZeroGrad` all params first |
| gradients finite but wrong | parameter `Data` mutated between forward and backward | update strictly after `Backward` |
| slow creep, no convergence | lr too small, or loss averaging hides progress | sanity-check with the quick-start program in the root README |
