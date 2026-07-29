# Pitfalls, boundaries, and residual risks

> English | [中文](zh/pitfalls.md)

**Summary:** everything the library does *not* protect you from, distilled
from the red-team audit into user-facing caveats: the single-threaded
contract, `float32` overflow, exact repeated-`Backward` semantics, and the
technical-debt roadmap.

**Audience:** read this before shipping anything on lnn.

## 1. lnn is single-threaded by design

`Backward` mutates the `Grad` buffers of leaf variables with plain
`+=` and no synchronization (`autograd/variable.go`); tensors expose their
`Data` directly; `math/rand.Rand` is not goroutine-safe. Running `Backward`
concurrently on graphs that share parameters is a data race that loses or
corrupts gradients — verified empirically under `go test -race`, and the
reason the audit downgraded concurrency from "bug" to "contract".

**Why not just add locks:** the library trades that generality away for a
kernel with zero synchronization overhead and no hidden coupling; every
buffer is a plain slice you can reason about.

**The supported parallel pattern:** give each goroutine its *own* cell,
tensors, variables and RNG. Never share a `Variable`, `Tensor`, graph — or
`rand.Rand` — across goroutines. Verified race-free under `-race`:

```go
var wg sync.WaitGroup
for g := 0; g < 4; g++ {
	wg.Add(1)
	go func(g int) {
		defer wg.Done()
		rng := rand.New(rand.NewSource(int64(g))) // own RNG
		cell := nn.NewLTC(1, 4, nil, 4, rng)      // own parameters
		x := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 1))
		out, _ := cell.Step(x, nil, 0.1)          // own graph
		// ... own loss, own Backward, own update ...
		_ = out
	}(g)
}
wg.Wait()
```

Aggregating results (e.g. averaging losses) must happen through your own
channels/mutexes, on plain values, after each goroutine finishes its
backward pass.

## 2. float32 overflow is not guarded globally

Storage is `float32` everywhere; there is no overflow detection and no
`float64` mode. Concrete ways to produce `Inf`/`NaN`:

| operation | overflow behavior |
|---|---|
| `Exp(x)` / `Pow(e, x)` | `+Inf` for `x ≳ 88` |
| `Log(x)` | `NaN` for `x < 0`, `-Inf` for `x = 0` — domain is *not* checked |
| `Div(a, b)` | `±Inf` for `b = 0` (plain float32 division semantics) |
| accumulations (`SumAll`, `MeanAll`) | rounding drift grows with length; accumulation is plain left-to-right `float32` addition |
| `Softplus(x)` | exactly `x` for `x > 20` (stable branch), otherwise `log1p(exp)` |
| `SoftmaxRows`/`LogSoftmaxRows` | internally stabilized (max-subtracted); safe for large logits |

Stabilized internals exist only where listed (plus `Sigmoid`, and LTC's
`ts` scaling — see [ltc.md](ltc.md)). Everything else is your problem:
keep logits bounded, clamp inputs to `Log`, and clip gradients
([training.md](training.md)). Once a single `NaN` exists, it propagates
through elementwise paths for the rest of the iteration.

**One asymmetric NaN behavior to know:** `MatMul` skips zero multipliers
in its inner loop (`tensor/ops.go:20`), so `0 * NaN` contributes `0`
instead of poisoning the product:

```go
tensor.MatMul(tensor.FromData([]float32{0}, 1, 1),
	tensor.FromData([]float32{float32(math.NaN())}, 1, 1)).Data // [0], not [NaN]
```

Do not rely on MatMul to surface NaNs in sparse positions.

## 3. Repeated Backward on the same graph: exactly linear

Build a fresh graph each iteration and call `Backward` once. If you call
`Backward` twice on the *same* graph, leaf gradients grow by exactly one
more full backward pass — two calls give precisely twice the gradient,
three calls three times (intermediate gradients are cleared after each
traversal, which is what keeps it linear instead of super-linear; the
pre-fix bug accumulated 3x):

```go
a := autograd.New([]float32{1, 2, 3}, 3)
y := autograd.SumAll(autograd.Hadamard(a, a)) // dy/da = 2a
y.Backward()
fmt.Println(a.Grad.Data) // [2 4 6]
y.Backward()             // same graph, second run
fmt.Println(a.Grad.Data) // [4 8 12] — exactly doubled
```

Related: gradients accumulate across *different* graphs sharing a leaf
until you `ZeroGrad` — that is the feature the training loop relies on.

## 4. Do not mutate data between forward and Backward

A few backward closures read parent `Data` at backward time (e.g. `Log`
computes `1/x` from the parent's *current* values), and `MatMul`'s
backward uses the saved operand tensors. Mutating a leaf between the
forward pass and `Backward` silently changes the gradient:

```go
x := autograd.New([]float32{2}, 1)
l := autograd.SumAll(autograd.Log(x)) // d/dx = 1/x, evaluated in Backward
x.Data.Data[0] = 8                    // mutation before Backward
l.Backward()
fmt.Println(x.Grad.Data) // [0.125] = 1/8, not 1/2
```

In-place parameter updates therefore belong strictly *after* `Backward`.

## 5. GatherRows copies its indices (fixed hazard)

`autograd.GatherRows(a, idx)` copies `idx` on entry
(`autograd/ops.go:271`), so the caller may freely reuse or mutate the
slice between forward and backward — the gradient is computed from the
indices used in the forward pass:

```go
m := autograd.New([]float32{1, 2, 3, 4}, 2, 2)
idx := []int{0, 1}
out := autograd.GatherRows(m, idx)
idx[0], idx[1] = 1, 0 // mutate the caller's slice before Backward
autograd.SumAll(out).Backward()
fmt.Println(m.Grad.Data) // [1 0 0 1] — still correct
```

This used to be a silent-corruption bug (the closure captured the slice
by reference and the red team measured scrambled gradients); it is now
regression-tested.

## 6. Tiny-ts is a finiteness-only domain

`LTC.Step` accepts any positive finite `ts`, but below `ts ≈ 1e-38` the
`unfolds/ts` scale overflows `float32` and is clamped: outputs stay finite
(verified at `1e-40` and `1e-300`), but physical fidelity is gone. The
normal training regime (`ts ≳ 1e-3`) is bit-identical to the naive ODE
algebra. Full table in [ltc.md](ltc.md#the-time-span-ts).

## 7. Random initialization fine print

- **`Randn` is tail-truncated at ≈ 7.43σ.** Box-Muller's `u1` uniform is
  clamped away from zero at `1e-12` (`tensor/random.go:35-36`), so
  `|sample| ≤ sqrt(−2·ln(1e-12)) ≈ 7.43` always holds — the omitted tail
  mass is ~1e-13. Immaterial for weight initialization; do **not** use
  `Randn` as a general normal sampler (e.g. Monte Carlo that relies on
  tail events). Removing the clamp would change fixed-seed streams, so it
  is kept and documented rather than fixed.
- **`Uniform(lo, hi)` with `lo > hi` mirrors instead of panicking:**
  values fall in `[hi, lo]`. Deliberate legacy behavior kept for backward
  compatibility; callers relying on bounds should pass `lo ≤ hi`.
- Same seed ⇒ bitwise identical streams and models (`TestLTCDeterministicSameSeed`).

## 8. Shape conventions are asymmetric

`SumRows → [1, n]` vs `SumCols → [m]`, and 1D⊕1D results are promoted to
`[1, n]`. These are historical conventions that `nn` and `autograd`
internals depend on; changing them is an API-breaking change tracked
separately (roadmap). Full tables and workarounds in
[shapes-and-broadcasting.md](shapes-and-broadcasting.md).

## 9. The graph is the memory model

Every intermediate tensor stays alive until `Backward` completes. One LTC
step unrolls `unfolds` ODE iterations of O(units²) synapse ops; a sequence
of `T` steps multiplies that by `T`. Memory grows with ops per iteration,
not just parameters — keep `units`, `unfolds` and sequence length modest.

## Roadmap and technical debt

Tracked openly; none of it blocks production use within the documented
scope (single-threaded, modest scale, sane `ts`/`lr`, caller clips
gradients):

| item | status |
|---|---|
| `autograd.Div` closed form | **done:** single graph node with quotient-rule backward (`da = g/b`, `db = −g·a/b²`, `autograd/ops.go:171-194`). Note the inherent `1/b²` gradient amplification for small divisors remains — with LTC's `eps = 1e-8` floor that is up to ~`1e16` — so gradient clipping stays recommended |
| Unify reduction shape conventions (`SumRows`/`SumCols`, 1D promotion) | separate evaluation; API-breaking |
| `tensor.Stack` | experimental: yields 3D tensors that no other op consumes (`tensor/tensor.go:165`); kept for compatibility |
| CfC (Closed-form Continuous-time) cell | not implemented |
| Built-in optimizers | not implemented; hand-rolled SGD is the supported pattern |
| Serialization (Save/Load) | not implemented; parameters are plain `[]float32` buffers — snapshot `p.Data.Data` yourself |
| Benchmarks/CI tooling beyond `make test` | in progress |
