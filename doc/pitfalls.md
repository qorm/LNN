# Pitfalls, boundaries, and residual risks

> English | [中文](zh/pitfalls.md)

**Summary:** everything the library does *not* protect you from, distilled
from the red-team audit into user-facing caveats: the single-threaded
contract, `float32` overflow, exact repeated-`Backward` semantics, the
untrusted-stream contract of the persistence layer, and the
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

Stabilized internals exist only where listed (plus `Sigmoid`, LTC's `ts`
scaling — see [ltc.md](ltc.md) — and the CfC's exprel decay factor and
`ts` scaling, see [cfc.md](cfc.md)). Everything else is your problem:
keep logits bounded, clamp inputs to `Log`, and clip gradients
([training.md](training.md)). Once a single `NaN` exists, it propagates
through elementwise paths for the rest of the iteration.

**Worse, a `NaN` that reaches an optimizer's state is permanent.** The
momentum/second-moment estimates of `Momentum` and `Adam` are running
accumulators: once a single `NaN` gradient is folded into them, that
parameter's state is `NaN` forever, and every later step multiplies the
poison forward — a long red-team run (one injected `NaN`, 60 iterations)
stayed all-`NaN` to the end and never recovered. Crucially, **later
healthy gradients do not wash the state clean**, and `ZeroGrad` does not
help either: the poison lives in the optimizer's per-parameter buffers,
not in `Grad`. A contaminated optimizer must be discarded and rebuilt
(fresh state), or the affected parameters reset. This is the strongest
argument for keeping logits bounded and clipping gradients
([training.md](training.md)) — once `NaN` enters a stateful optimizer,
the only fix is starting that state over.

**One asymmetric NaN behavior to know:** `MatMul` skips zero multipliers
in its inner loop, and the skip tests only the **left** operand
(`tensor/ops.go:20`). The behavior is therefore directional: `0 * NaN`
contributes `0` (the zero multiplier is skipped before the product is
formed), but `NaN * 0` still yields `NaN` (the `NaN` left multiplier is
not zero, so the product `NaN * 0 = NaN` is accumulated):

```go
tensor.MatMul(tensor.FromData([]float32{0}, 1, 1),
	tensor.FromData([]float32{float32(math.NaN())}, 1, 1)).Data // [0], not [NaN]
tensor.MatMul(tensor.FromData([]float32{float32(math.NaN())}, 1, 1),
	tensor.FromData([]float32{0}, 1, 1)).Data // [NaN] — the other order poisons
```

Do not rely on MatMul to surface NaNs in sparse positions, and do not
assume the zero-skipping is symmetric — it is not.

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

A few backward steps read parent `Data` at backward time (e.g. `Log`
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
(`autograd/ops.go:855`), so the caller may freely reuse or mutate the
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
step unrolls `unfolds` ODE iterations, each O(units) vector blocks plus
two MatMul contractions since the phase-6 synapse vectorization (down
from O(units²) per-synapse nodes — see [ltc.md](ltc.md)), and the
phase-7 backward overhaul halved the per-node allocation count again,
with the phase-8 Sigmoid–Hadamard fusion taking it further still
(`UnrollBackward` 31,983 allocs/op, −73% cumulative from the original
loop — see [architecture.md](architecture.md)); a sequence of
`T` steps multiplies that by `T`. Memory grows with ops per iteration,
not just parameters — keep `units`, `unfolds` and sequence length
modest, or use the `CfC` cell ([cfc.md](cfc.md)), whose closed-form
step has no `unfolds` factor.

## 10. Persistence treats model files as untrusted input

`nn.SaveLTC`/`LoadLTC`, `SaveCfC`/`LoadCfC`, `SaveLinear`/`LoadLinear`
and the `serialize` package underneath are the library's documented
exception to panic-on-misuse: a checkpoint is exactly the kind of input
the program does not control, so **every failure on the load path is an
error, never a panic**, and a hostile stream can allocate only in
proportion to the bytes it actually delivers. The contract in brief:

- **Fixed limits, validated before any allocation:** one tensor ≤
  `2^30` float32s (4 GiB payload), one stream ≤ `2^20` tensors, rank ≤
  `8` axes, and `LoadLTC` refuses `unfolds > 1024` before the blob is
  even parsed. Element counts use overflow-safe multiplication, so a
  stream claiming a `1<<62`-wide dimension is an error, not a
  petabyte-sized `make()`.
- **Known-length readers** (`bytes.Reader` etc.) get every payload
  claim checked against the remaining bytes first; **unknown-length
  readers** (`io.Pipe`, `net.Conn`, `gzip.Reader`) get progressive
  allocation — a claim of `2^30` elements that stops after 18 bytes
  peaks at ~33 KiB and fails with `io.ErrUnexpectedEOF`.
- **Model-level checks:** exact kind byte (cross-loading is a named
  error), masks exactly `{0, 1}`, reversal potentials exactly `±1`
  (`NaN`/`±Inf`/`0`/fractions refused), exact tensor count, and all
  shapes validated before any value is copied — a failing load leaves
  the destination untouched.
- **`serialize.LoadParameters` leaves stale `Grad` in place:** it
  overwrites `Data` in place (variable identities, and thus graph
  edges, survive) and deliberately does not touch `Grad`. Call
  `ZeroGrad` before reusing loaded variables in a new graph — exactly
  as before any training step.

Format spec, API guide and the complete contract — including the
versioning rule (version 1 only; unknown versions error out rather than
mis-parse) — in [persistence.md](persistence.md).

## Roadmap and technical debt

Tracked openly; none of it blocks production use within the documented
scope (single-threaded, modest scale, sane `ts`/`lr`, caller clips
gradients):

| item | status |
|---|---|
| `autograd.Div` closed form | **done:** single graph node with quotient-rule backward (`da = g/b`, `db = −g·a/b²`, `autograd/ops.go:793-810`). Note the inherent `1/b²` gradient amplification for small divisors remains — with LTC's `eps = 1e-8` floor that is up to ~`1e16` — so gradient clipping stays recommended |
| LTC synapse vectorization | **done (phase 6):** masks folded out of the hot path; per-presynaptic-neuron vector blocks + two construction-time indicator-matrix MatMul contractions ([ltc.md](ltc.md)). `LTCStep` 7,360 → 3,440 allocs/op (−53.3%), `UnrollBackward` 120,163 → 68,688 (−42.8%). The whole `Step` is ULP-equivalent to the pre-rewrite loop (forward ≤ 1.79e-7, gradients ≤ 1.19e-7, independent red-team oracle); bitwise identity holds only for the isolated `synapses()` drive |
| Autograd backward overhaul (closures, fusion, `addGrad` cloning) | **done (phase 7):** per-node backward closures replaced by op-kind tag dispatch; `addGrad` first-contribution ownership transfer (Clone share ~20% → ~1%); unary backward chains fused (Sigmoid/Tanh 4→1 nodes), MatMul transpose buffers removed, product-and-reduce fused — all gated bitwise against the pre-rewrite oracle on 52k differential graphs, with an arm64 FMA conversion barrier keeping fused loops two-rounding-exact ([architecture.md](architecture.md)). `UnrollBackward` 68,688 → 33,963 allocs/op (−50.55%); four other benchmarks −23%…−58%, zero regressions. Remaining per-node overhead is tracked in the `tensor.New` item below |
| `tensor.New` fixed per-node overhead | future: profiling puts 64.9% of the remaining allocations in each node's forward output plus its `Shape`/`Data` double allocation; further compression needs fixed-capacity parent slots (blocked by existing structural-assertion tests) and a fixed-rank `Tensor` shape (public-API break) |
| Sigmoid–Hadamard fused backward (LTC hot-path pattern) | **done (phase 8):** `autograd.SigmoidHadamard(z, w)` fuses the hot-path `Hadamard(Sigmoid(z), w)` into one node (adopted at `nn/ltc.go:347`). The forward is bitwise by construction (it runs the same two tensor ops), and the regular 2D backward reaches bitwise equivalence with the legacy two-node chain by rounding the `g⊙w` product at the very same site; irregular shapes or manually seeded gradients fall back to the legacy composition verbatim (quirks and panic contract preserved). `LTCStep` 2,442 → **2,306** allocs/op (−5.6%), `UnrollBackward` 33,963 → **31,983** (−5.8%) — a single-digit gain, and the measured structural ceiling: each site saves exactly one graph node plus one backward intermediate, the rest being `tensor.New`'s per-node overhead (previous item). See [architecture.md](architecture.md) |
| CfC `erev` dead gradients | **done (phase 8):** the CfC's reversal potentials are now baked into construction-time indicator matrices — the LTC's trick, adopted. The `erev`/`sErev` fields are plain `*tensor.Tensor` with no gradient to compute, so the dead gradient is *structurally impossible* rather than merely zero; `drive()` collapses to two MatMul contractions. Bitwise-equivalent to the former `Var`-leaf drive (red-team differential test), and the load path rebuilds the indicators from the streamed polarities ([persistence.md](persistence.md)) |
| Indicator-matrix O(units³) materialization | **load side closed (phase 8), root fix open:** the dense `[pre·units, units]` reduction indicators make construction- and load-time memory O(units³) while the header that controls it is only 9–13 bytes. `LoadLTC`/`LoadCfC` now cap `units`/`inDim` at 256 (≤ 256 MiB of indicator matrices per loaded cell, ~320 MiB peak), turning a 13-byte units=4096 attack stream that used to attempt ~550 GB until the OS killed the process into a valued error. The root fix — sparse contractions that never materialize the `[units², units]` indicators, on the constructor side too — remains open ([persistence.md](persistence.md)) |
| Unify reduction shape conventions (`SumRows`/`SumCols`, 1D promotion) | separate evaluation; API-breaking |
| `tensor.Stack` | experimental: yields 3D tensors that no other op consumes (`tensor/tensor.go:170`); kept for compatibility |
| CfC (Closed-form Continuous-time) cell | **done (phase 6):** `nn.CfC` (`nn/cfc.go`) — same ODE and synapse parameterization as the LTC, driven by the Lemma 1 closed form; paper↔code correspondence and verification trail in [cfc.md](cfc.md). New API: may still evolve |
| Built-in optimizers | **done (phase 6):** the `optimizer` package (SGD/Momentum/Adam, 100% coverage); the hand-rolled loop remains the supported pattern for understanding and for rules the package does not cover — [training.md](training.md) covers both |
| Serialization (Save/Load) | **done (phase 7):** the `serialize` package (versioned `"LNNS"` tensor streams) plus six `nn` Save/Load functions for LTC/CfC/Linear; hostile-stream safe — errors never panics, fixed limits validated before allocation, progressive allocation on unknown-length readers (a 4 GiB claim that stops after 18 bytes peaks at ~33 KiB). Red-team mutation fuzzing: 7,500 mutants with 0 panics, plus 1,200 mutants after the resource-exhaustion hardening, again 0 panics. Format spec, API guide and the full safety contract in [persistence.md](persistence.md) |
| Benchmarks/CI tooling | **done:** `make bench` (13 benchmarks) + GitHub Actions CI (gofmt gate, vet, build, `test -race`, example smoke test) |
