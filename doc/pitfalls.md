# Pitfalls, boundaries, and residual risks

> English | [中文](zh/pitfalls.md)

**Summary:** everything the library does *not* protect you from, distilled
from the red-team audit into user-facing caveats: the single-threaded
contract, `float32` overflow, exact repeated-`Backward` semantics, the
untrusted-stream contract of the persistence layer, and the
technical-debt roadmap.

**Audience:** read this before shipping anything on LNN.

## 1. LNN is single-threaded by design

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
(`autograd/ops.go:891`), so the caller may freely reuse or mutate the
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
internals depend on; evaluated in the v0.4.0 API-stability window and
**frozen** — unifying them would void the differential-fuzz evidence base
for symmetry alone, with zero performance gain. The full decision trail,
tables and workarounds are in
[shapes-and-broadcasting.md](shapes-and-broadcasting.md).

## 9. The graph is the memory model

Every intermediate tensor stays alive until `Backward` completes. One LTC
step unrolls `unfolds` ODE iterations, each O(units) activation blocks
since the phase-6 synapse vectorization (down from O(units²) per-synapse
nodes), with the presynaptic reduction folded — `+0`-seeded, ended by
normalizing MatMuls — since the phase-9 sparse contraction eliminated
the dense `[units², units]` indicator matrices (see
[ltc.md](ltc.md)), and the phase-7 backward overhaul halved the
per-node allocation count, with the phase-8 Sigmoid–Hadamard fusion
taking it further still, and the phase-10 embedded shape backing
(inline `[4]int` shape storage, one fewer allocation per tensor)
trimming it once more, and the stage-16 fused unfold kernel (one
`FusedOp` node per LTC step, bitwise vs the graph path) taking it to
`UnrollBackward` **3,750 allocs/op** — −97% cumulative from the
original loop, with `LTCStep` at **77** (the phase-9 step had raised
allocs ~30% but cut wall-clock ~13%, phase 10 cut allocs a further
−23% with wall-clock flat, stage 16 cut wall-clock ~2.1–2.3×
further, and stage 19 cut allocations another 67%/44% by moving the
sensory path into the kernel and reusing scratch — see
[architecture.md](architecture.md)); a
sequence of `T` steps multiplies that by `T`. Memory grows with ops
per iteration, not just parameters — keep `units`, `unfolds` and
sequence length modest, or use the `CfC` cell ([cfc.md](cfc.md)),
whose closed-form step has no `unfolds` factor, or cap peak graph
memory at O(chunk) with `nn.UnrollRemat`
([architecture.md](architecture.md);
[cookbook.md](cookbook.md#13-long-sequence-training-chunked-bptt-with-unrollremat)
recipe 13).

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
- **`optimizer.SaveState`/`LoadState` follow the same discipline:**
  the `"LNO1"` state stream is validate-all-then-apply (a failing load
  leaves the optimizer bit-for-bit as it was), errors never panic, and
  hostile size claims stay within a tested byte budget. With state
  saved, resumed training is bit-identical to uninterrupted training —
  see the optimizer state section of [persistence.md](persistence.md).

Format spec, API guide and the complete contract — including the
versioning rule (v1 and v2 read, v2 written; unknown versions error out
rather than mis-parse) — in [persistence.md](persistence.md).

## Residual register (stages 16 and 18)

Archived corners of the stage-16 work (the fused LTC kernel and
`UnrollRemat`) and the stage-18 work (the fused CfC kernel), disclosed
rather than fixed — none of them blocks the
documented use; the format follows the audit's archival table
(issue / source / severity / disposition):

| # | issue | source | severity | disposition |
|---|---|---|---|---|
| 1 | Fused LTC path: a NaN produced by an operation whose operands are **both** NaNs does not preserve its payload/sign bit across the fused and the former graph path (NaN positions and every finite value are bitwise). The fused CfC step (stage 18) carries the identical acceptance level. NaNs **created** on the overflow paths (`Inf*0`) likewise carry an architecture- and compile-context-dependent sign — the differential tests assert the NaN position set and finite values, never NaN bits (the stage-18 CI amd64 run pinned this corner empirically) | stage-16 red team | Low (accepted) | Same acceptance class as the single-source sign-bit corner F9-1 — training-irrelevant, no observable trajectory divergence; the payload depends on compiler instruction selection, not on any replicated rounding structure. Contract stated in the NaN-semantics headers of `nn/ltc_fused.go` and `nn/cfc_fused.go` |
| 2 | `UnrollRemat` on a cell whose per-step **graph structure depends on tensor values** drifts ~1–2 ULP, invisible to the one-step structural probe | stage-16 red team | Low (accepted) | godoc contract: `Step`'s graph structure must be a pure function of `(x, h)`; both provided cells satisfy it. Value-branching custom cells are out of contract by construction |
| 3 | `UnrollRemat` worst case: an adversarial loss visit order (a descending visit with a small chunkSize is the extreme) merges recompute units up to O(T) — peak memory AND compute then exceed one whole-graph backward | stage-16 red team | Info | Documented in the godoc and the cookbook's chunkSize guidance; spell losses so outputs are visited in ascending step order (data-first) and the O(chunk) fast path holds |
| 4 | The params-completeness audit is only as honest as the cell's own `Parameters()`: a `Module` that omits a `Step`-consumed leaf from its list defeats the audit, and that leaf is then multi-counted silently (2–3× wrong gradient) | stage-16 red team | Low (accepted) | Both provided cells report completely; for custom cells, a complete `Parameters()` is the cell's side of the contract — the audit covers what the cell discloses |
| 5 | `nn` coverage is 99.9%, not 100%: the sole uncovered statement is a constructively unreachable nil-root firewall guard in `UnrollRemat`'s unit sweep | stage-16 API sweep | Info | Kept as a firewall against future unit-cut changes, with the unreachability argument annotated in `nn/remat.go`; disclosed in the README status table (the optimizer row's 99.6% discipline) |
| 6 | Fused kernels are stricter about **state shape** than the graph path was: a broadcast-compatible wrong `h` (e.g. `[batch, 1]`, `[1, units]`) used to be silently broadcast into wrongly shaped gradients; both fused cells now panic up front | stage-18 red team | Info (defect fix) | The `Cell` contract always required `h` of shape `[batch, StateSize()]` — the graph path's leniency was a defect, the panic is the contract. Same call on both kernels (LTC earlier, CfC in stage 18); pinned by differential tests |
| 7 | A **panicked backward** behaves differently across paths: the graph path leaves partial internal accumulators delivered (a retried `Backward` panics again on the polluted graph), while the fused kernels panic before any delivery and stay clean on retry | stage-18 red team | Info | Not replicated — the clean behavior is strictly preferable and the pollution was never reproducible anyway. Guidance is path-independent: after a `recover`, discard the graph and build a fresh one |

## Roadmap and technical debt

Tracked openly; none of it blocks production use within the documented
scope (single-threaded, modest scale, sane `ts`/`lr`, caller clips
gradients):

| item | status |
|---|---|
| `autograd.Div` closed form | **done:** single graph node with quotient-rule backward (`da = g/b`, `db = −g·a/b²`, `autograd/ops.go:829-846`). Note the inherent `1/b²` gradient amplification for small divisors remains — with LTC's `eps = 1e-8` floor that is up to ~`1e16` — so gradient clipping stays recommended |
| LTC synapse vectorization | **done (phase 6; contraction re-done in phase 9):** masks folded out of the hot path; per-presynaptic-neuron vector blocks ([ltc.md](ltc.md)). Phase 6 contracted the presynaptic axis with indicator-matrix MatMuls (`LTCStep` 7,360 → 3,440 allocs/op, `UnrollBackward` 120,163 → 68,688); phase 9 replaced the indicators with a `+0`-seeded sparse fold ended by a normalizing identity MatMul — bitwise-identical to the indicator implementation in forward and backward (1,164-config differential vs the published oracle + independent red-team rerun), while the whole `Step` remains ULP-equivalent to the *original* per-synapse loop (forward ≤ 1.79e-7, gradients ≤ 1.19e-7). Phase 9 raised allocs ~43%/~30% (fold-stage cloning) but cut ns/op ~21%/~13% — wall-clock is a net benefit; the phase-10 embedded shape backing (next rows) then cut allocs a further −18%/−23% with wall-clock flat. Values before stage 16: `LTCStep` 2,707 / `UnrollBackward` 31,994 allocs/op — superseded by the stage-16 fusion row below |
| Autograd backward overhaul (closures, fusion, `addGrad` cloning) | **done (phase 7):** per-node backward closures replaced by op-kind tag dispatch; `addGrad` first-contribution ownership transfer (Clone share ~20% → ~1%); unary backward chains fused (Sigmoid/Tanh 4→1 nodes), MatMul transpose buffers removed, product-and-reduce fused — all gated bitwise against the pre-rewrite oracle on 52k differential graphs, with an arm64 FMA conversion barrier keeping fused loops two-rounding-exact ([architecture.md](architecture.md)). `UnrollBackward` 68,688 → 33,963 allocs/op (−50.55%); four other benchmarks −23%…−58%, zero regressions. Remaining per-node overhead is tracked in the `tensor.New` item below |
| `tensor.New` fixed per-node overhead (`Shape`/`Data` double allocation) | **done (v0.4.0, option ② embedded backing):** profiling put 64.9% of the remaining allocations in each node's forward output plus its `Shape`/`Data` double allocation (re-measured 60.4% just before the fix). v0.4.0 eliminates the `Shape` half with an embedded `[4]int` shape buffer (embedded backing, inline `shapeBuf` in the struct): zero heap allocation for rank ≤ 4, heap fallback beyond it (so `serialize`'s rank-8 streams stay compatible). Five benchmarks −18~−26% allocs/op (each tensor operator lost exactly one shape allocation), wall-clock unchanged within ±a few % noise and bytes +3% — the win is allocation count and GC hygiene, not wall-clock. Option ② (embedded backing, ~10 internal touch points, zero API breakage) was chosen over option ① (a value-type shape field, the same saving but 233 `.Shape` accesses and 7 direct writes broken); the new exported `Tensor.Reshape` replaces direct `Shape` writes (negative dimensions panic). The residual (fixed-capacity parent slots) was then **done too (v0.4.3):** inline 2-slot + variadic overflow in the graph node, five benchmarks −10~−22% allocs, bitwise-equivalent against a 15,360-graph differential fuzz — closing the last performance backlog item before stage 16 |
| Sigmoid–Hadamard fused backward (LTC hot-path pattern) | **done (phase 8):** `autograd.SigmoidHadamard(z, w)` fuses the hot-path `Hadamard(Sigmoid(z), w)` into one node (adopted at `nn/ltc.go:423`). The forward is bitwise by construction (it runs the same two tensor ops), and the regular 2D backward reaches bitwise equivalence with the legacy two-node chain by rounding the `g⊙w` product at the very same site; irregular shapes or manually seeded gradients fall back to the legacy composition verbatim (quirks and panic contract preserved). `LTCStep` 2,442 → **2,306** allocs/op (−5.6%), `UnrollBackward` 33,963 → **31,983** (−5.8%) — a single-digit gain, and the measured structural ceiling: each site saves exactly one graph node plus one backward intermediate, the rest being `tensor.New`'s per-node overhead (previous item). See [architecture.md](architecture.md) |
| CfC `erev` dead gradients | **done (phase 8; indicators removed in phase 9):** the CfC's reversal potentials no longer enter the graph — the ±1 signs ride row-view constants sharing the `erev`/`sErev` storage. The fields are plain `*tensor.Tensor` with no gradient to compute, so the dead gradient is *structurally impossible* rather than merely zero; `drive()` ends in the same sparse fold as the LTC (`nn/cfc.go`'s contract is line-for-line isomorphic). Bitwise-equivalent to the former `Var`-leaf drive (red-team differential test), and `LoadCfC` overwrites the `erev`/`sErev` storage in place, which the row views pick up with no rebuild ([persistence.md](persistence.md)) |
| ~~Indicator-matrix O(units³) materialization~~ | **done (phase 9, root fix landed):** the sparse contraction ([ltc.md](ltc.md)) never materializes a `[units², units]` indicator, in the constructor OR the load path — construction of a fully-wired `units = 1024` cell costs ~32 MB (measured 36.4 MiB), not the old ~8 GiB cliff. The `maxUnits`/`maxInDim` load caps were re-derived on the new O(units²) memory model: `256 → 2048` (peak `92·U² B` ≈ 368 MiB at the cap — the same ~320 MiB budget class as the old regime, 8× the capacity), and a minimal attack stream now peaks at ~1.5× its delivered bytes, so the F1 contract (hostile streams allocate in proportion to delivered bytes) is honored at the root rather than bandaged ([persistence.md](persistence.md)) |
| Optimizer state persistence | **done (phase 9):** `optimizer.SaveState`/`LoadState` (`"LNO1"` state streams) — resumed training is bit-identical to uninterrupted training (50+50 vs 100 steps, per-parameter trajectories and losses, all three optimizers). Untrusted-stream discipline: validate-all-then-apply with zero side effects, errors never panics, hostile claims within a tested byte budget, `maxT = 2²⁴` load-only cap on Adam's update count. Format spec and contracts in [persistence.md](persistence.md) |
| Unify reduction shape conventions (`SumRows`/`SumCols`, 1D promotion) | **frozen (v0.4.0):** the evaluation measured the true cost of unification — it would invalidate all 96k+52k+522 differential-fuzz graphs, force a redo of the phase-7 magnitude-equivalence proof, and rewrite 11 lift sites / 23 guards / ≥17 tests — in exchange for symmetry and zero performance gain; frozen in the zero-user window. Decision trail in [shapes-and-broadcasting.md](shapes-and-broadcasting.md) |
| `tensor.Stack` | **removed (v0.4.0):** zero in-library callers and zero doc examples; it yielded 3D tensors that no other op consumes, so it was dropped to narrow the public surface (API hygiene) |
| CfC (Closed-form Continuous-time) cell | **done (phase 6):** `nn.CfC` (`nn/cfc.go`) — same ODE and synapse parameterization as the LTC, driven by the Lemma 1 closed form; paper↔code correspondence and verification trail in [cfc.md](cfc.md). New API: may still evolve |
| Built-in optimizers | **done (phase 6):** the `optimizer` package (SGD/Momentum/Adam, 100% coverage); the hand-rolled loop remains the supported pattern for understanding and for rules the package does not cover — [training.md](training.md) covers both |
| Serialization (Save/Load) | **done (phase 7):** the `serialize` package (versioned `"LNNS"` tensor streams) plus six `nn` Save/Load functions for LTC/CfC/Linear; hostile-stream safe — errors never panics, fixed limits validated before allocation, progressive allocation on unknown-length readers (a 4 GiB claim that stops after 18 bytes peaks at ~33 KiB). Red-team mutation fuzzing: 7,500 mutants with 0 panics, plus 1,200 mutants after the resource-exhaustion hardening, again 0 panics. Format spec, API guide and the full safety contract in [persistence.md](persistence.md) |
| Benchmarks/CI tooling | **done:** `make bench` (18 benchmarks) + GitHub Actions CI on **amd64 and arm64** (ubuntu-latest + ubuntu-24.04-arm; gofmt gate, vet, build, `test -race`, example smoke test, fuzz smoke over 11 native targets) |
| LTC ODE-unroll graph-operator fusion; rematerialized BPTT | **done (stage 16):** `autograd.FusedOp` (caller-owned backward closure) + the LTC's unfolds fused into one graph node (`nn/ltc_fused.go`) — forward and gradients bitwise vs the pre-fusion graph path, public API unchanged. Measured (`-benchtime=200x`): `LTCStep` ~203 µs / 2,122 allocs → **~87 µs / 236**, `UnrollBackward` ~2.8–3.3 ms / 28,273 → **~1.33 ms / 6,662** (~2.1–2.3×). `nn.UnrollRemat` adds chunked BPTT with bit-exact gradients at O(chunk) peak graph memory (T = 512: ~0.65 MB retained vs ~8.3 MB live). Archived corners in the residual register above; mechanism in [architecture.md](architecture.md), usage recipe in [cookbook.md](cookbook.md#13-long-sequence-training-chunked-bptt-with-unrollremat) |
| CfC closed-form step fusion; differential-fuzz hardening; dual-arch CI | **done (stage 18):** the CfC's whole per-step subgraph fused into one `FusedOp` node (`nn/cfc_fused.go`) — `66 + 14·(inDim+units)` nodes → **24 at any dims** (190→24 at `inDim=1, units=8`, 346→24 at `4, 16`), forward and gradients bitwise vs the graph path (chained/stacked included), public API unchanged, no new exported symbols; `h`-first parents keep the CfC spine-free, so `UnrollRemat` stays at its 2F+1B ideal. Measured (`-benchtime=200x`): `CfCStep` ~83.5 µs / 1,066 allocs → **~34 µs / 52** (~2.4×, −95% allocs), `UnrollRematCfC` ~1.86 ms / 22,106 → **~0.83 ms / 5,000** (~2.2×, −77%). Three bitwise differential fuzz targets (`FuzzLTCFusedDifferential`, `FuzzCfCFusedDifferential`, `FuzzUnrollRematDifferential`; 36 seed corpus files) joined the fuzz suite (8 → 11 targets), and CI now runs every gate on amd64 **and** arm64. Two behavioral tightenings (state-shape strictness, pre-delivery panic) in the residual register above |
| Sensory path into the fused LTC kernel; VJP/DFS/broadcast hygiene | **done (stage 19):** 19a internalized the sensory drive and the `numConst`/`denBase` assemblies (~`10·inDim+2` nodes per step) — parents 9 → 12, the `hvN` delivery slot re-derived, a Step now records **34 nodes at any dims** (15 op + 19 leaves; 84 before, 626 pre-fusion; remat fold classes unchanged, pinned). 19b allocates the VJP's 18 scratch planes once per backward (delivery buffers stay fresh — `addGrad` ownership is part of the bitwise contract); `Backward`'s DFS scratch is pooled (re-entrant fallback, panic-pristine, `TopoOrder` results always fresh; `UnrollBackward` B/op −5.1%); broadcast fresh-shape arms return value-type `[2]int` (3 → 2 allocs/op). Measured (`-benchtime=200x`): `LTCStep` **~78 µs / 77 allocs / 23 KB** (v0.5.2 ~87 µs / 236 / 47.7 KB), `UnrollBackward` **~1.29 ms / 3,750 / 558 KB** (was ~1.33 ms / 6,662 / 927 KB), `UnrollRemat` **~3.4 ms / 9,373**, `UnrollRematCfC` **~0.84 ms / 4,829**; CfC untouched. Zero API/behavior change (examples byte-identical, serialization bytes identical) |
| ~~Remat fold-classification cache~~ | **vetoed and documented (stage-19a C9):** no sound cache key exists — the classification depends on graph *structure* the caller rebuilds per call, and a stale hit would silently skip the multi-class panic the probe exists to report. Rejected like the frozen conventions above; the cheap half landed instead (the two loss-graph walks share one `TopoOrder`) |
| AdEMAMix + Schedule-Free AdamW optimizers; `"LNO1"` kinds 3/4 | **done (stage 20):** two new update rules, each ported against a third-party float64 reference of its paper (AdEMAMix, arXiv:2409.03137 — dual gradient EMAs, the slow one deliberately uncorrected, α/β3 warmup schedulers; ScheduleFreeAdamW, arXiv:2405.15682 — iterate averaging instead of a decay schedule, with the `Train`/`Eval` y↔x conversion contract and a per-parameter mode bit that makes `Step` panic on an eval-mode parameter before modifying anything). Max deviation vs the references 9.5e-6 / 3.3e-7, discrimination-gated; 50+50 vs 100 bit-exact resume for both, warmup-boundary-crossing included. `"LNO1"` gains kinds 3/4 (AdEMAMix reuses Adam's record layout; ScheduleFree adds `k`/`pow2`/`lrMax`/`wsum`/`mode` — the mode byte makes eval-mode checkpoints cross-instance resumable; `lrMax`/`wsum` non-finite rejected as corruption), version stays 1 — an extension tag, not a migration; kinds 0–2 byte-unchanged (hand-written legacy kind-2 streams still load). Guides: [training.md](training.md) (equations, choice guidance), [persistence.md](persistence.md) (wire layout), [cookbook.md](cookbook.md) recipe 14. Residual, informational: the optimizer-level mode flag gates only state-less parameters (a safe-side overkill, upstream-consistent); a parameter resized mid-`Step` panics non-atomically (pre-existing, all optimizers alike) |
