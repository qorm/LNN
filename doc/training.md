# Training models with lnn

> English | [中文](zh/training.md)

**Summary:** lnn ships no optimizer on purpose — you write a four-line
loop (zero grads → forward → `Backward` → explicit parameter update) over
`Parameters()`, and you own stability (learning rate and gradient
clipping). This guide shows the supported pattern end to end.

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

	"lnn/autograd"
	"lnn/tensor"
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
  few backward closures read parent data at backward time, so in-place
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
`0.041996`.

## Why there is no optimizer

By design. The library's contract is a small, readable, auditable numeric
core; an update rule is five lines of Go, and writing it yourself keeps
lr schedules, clipping and regularization in your code, visible and
diffable, rather than hidden behind a framework abstraction. Nothing in
the graph engine assumes how leaves are updated — SGD, momentum and
clipping all fall out of `p.Data`/`p.Grad` access.

Roadmap (not implemented, no timeline): built-in optimizers and the CfC
cell. When they land, the hand-rolled pattern above remains valid.

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

Measured in a real LTC step (seed 42, units=8, unfolds=4, 6-step sequence):
gradient norm `2.50` → update scale `0.019975` instead of `0.05` on the
first iteration, with a maximum observed norm of about `6.0` over a
250-iteration run. The clip engages on most early iterations and is the
difference between converging and diverging on this task.

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

## Why did my training diverge?

A checklist ordered by how often each cause appears:

| symptom | likely cause | fix |
|---|---|---|
| loss → `NaN` within a few steps | lr too large; parameters overshoot into `Exp`/`Log` overflow | reduce lr 3–10x; add norm clipping |
| sudden `NaN` after many good steps | a `Log(x)` saw `x ≤ 0`, or a `Div` saw a zero divisor (`+Inf` forward, `Inf` gradients backward) | clamp inputs to `Log`; keep divisors bounded away from 0 |
| LTC loss spikes/diverges | gradient spike from the `1/(den+eps)` division (`den` can approach `eps = 1e-8`) | global norm clipping (above); smaller lr |
| everything `NaN` after a step | `float32` overflow propagated: once one element is `NaN`, MatMul-free paths carry it through the whole graph | clip; check `ts` is sane (`ts ≥ 1e-3` keeps full physical fidelity — see [ltc.md](ltc.md)) |
| gradients exactly 2x too big | `Backward` called twice on the same graph, or `ZeroGrad` missing | one `Backward` per fresh graph; `ZeroGrad` all params first |
| gradients finite but wrong | parameter `Data` mutated between forward and backward | update strictly after `Backward` |
| slow creep, no convergence | lr too small, or loss averaging hides progress | sanity-check with the quick-start program in the root README |
