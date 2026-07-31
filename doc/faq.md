# LNN FAQ

> English | [中文](zh/faq.md)

Frequent questions, short answers — each with a pointer to the guide
that carries the full explanation and, where useful, a tested snippet
(the [cookbook](cookbook.md) has the longer task recipes).

- [My loss won't go down. What do I check?](#my-loss-wont-go-down-what-do-i-check)
- [I got a `NaN` loss. Now what?](#i-got-a-nan-loss-now-what)
- [How do I choose `ts`?](#how-do-i-choose-ts)
- [How do I choose `units` and `unfolds`?](#how-do-i-choose-units-and-unfolds)
- [A big `units` model doesn't fit in memory. What now?](#a-big-units-model-doesnt-fit-in-memory-what-now)
- [Why are my gradients still growing after `Backward`?](#why-are-my-gradients-still-growing-after-backward)
- [Why doesn't `Step` call `ZeroGrad`?](#why-doesnt-step-call-zerograd)
- [Should I use the LTC or the CfC?](#should-i-use-the-ltc-or-the-cfc)
- [Do last-digit differences across machines matter?](#do-last-digit-differences-across-machines-matter)
- [How do I read load errors like "stream holds model kind …"?](#how-do-i-read-load-errors-like-stream-holds-model-kind-)
- [How do I resume training with Adam?](#how-do-i-resume-training-with-adam)

---

## My loss won't go down. What do I check?

**Short answer:** work the symptom table in
[training.md](training.md#why-did-my-training-diverge), in order — it
is ordered by how often each cause appears. The usual suspects: lr too
large (loss → `NaN`) or too small (slow creep), missing `ZeroGrad` or a
double `Backward` (gradients exactly doubled), no gradient clipping on
an LTC (the `1/(den+eps)` division spikes gradients), and parameter
`Data` mutated between forward and `Backward`.

Then instrument once with [cookbook recipe 6](cookbook.md#6-inspecting-and-debugging-a-model):
print per-parameter gradient norms and scan for `NaN`/`Inf`. A norm of
exactly `2×` what you expect is the double-`Backward`/missing-`ZeroGrad`
fingerprint; a nil `Grad` on a wired-in module is a graph-construction
bug; a huge norm on one parameter says where clipping must engage.

## I got a `NaN` loss. Now what?

**Short answer:** find the op that produced it, then — if you used
Adam or Momentum — assume the optimizer state is poisoned and rebuild
it. [pitfalls.md](pitfalls.md) §2 has the full overflow table.

The two facts that make `NaN` serious here:

1. There is no global overflow guard. `Exp(x)` overflows for
   `x ≳ 88`, `Log(x)` yields `NaN`/`−Inf` for `x ≤ 0` unchecked,
   `Div(a, 0)` is `±Inf`, and once one element is `NaN` it rides the
   elementwise paths through the rest of the iteration.
2. **A `NaN` that reaches optimizer state is permanent.** Momentum's
   velocity and Adam's moments are running accumulators; one `NaN`
   gradient poisons them, and neither later healthy gradients nor
   `ZeroGrad` washes it out — the poison lives in the optimizer's
   buffers, not in `Grad`. Measured, injecting one `NaN` gradient at
   iteration 3 of an Adam run:

```go
w := autograd.Var(tensor.FromData([]float32{1}, 1))
opt := optimizer.NewAdamDefault(0.1)
for it := 0; it < 25; it++ {
	loss := autograd.Pow(autograd.Sub(w, autograd.Const(tensor.FromData([]float32{3}, 1))), 2)
	w.ZeroGrad()
	loss.Backward()
	if it == 3 {
		w.Grad.Data[0] = float32(math.NaN()) // one poisoned gradient
	}
	opt.Step([]*autograd.Variable{w})
	// ...
}
```

```
iter  0  w=1.1
iter  4  w=NaN
iter  8  w=NaN
iter 12  w=NaN
iter 16  w=NaN
iter 20  w=NaN
iter 24  w=NaN
```

The only fix is discarding the optimizer (fresh state) or resetting
the affected parameters. Prevention is the argument for clipping
([training.md](training.md#gradient-clipping-do-it)) and bounded
logits: once `NaN` is in a stateful optimizer, you start that state
over.

## How do I choose `ts`?

**Short answer:** anchor one time unit to something physical (the
sampling interval is the usual choice) and pass the time span each
step covers — `ts = 1.0` if each sequence step is one sampling
interval. Keep `ts ≳ 1e-3` for full physical fidelity; `ts` must be
positive and finite or `Step` panics. Full contract and finiteness
domains: [ltc.md](ltc.md#the-time-span-ts).

What `ts` does, measured on one CfC step from zero state with input
`x = 1` (the LTC behaves the same way — Euler over its `unfolds`):

```
ts= 0.01  out[0]=+0.007009  |state|=+0.007009
ts= 0.10  out[0]=+0.064129  |state|=+0.064129
ts= 1.00  out[0]=+0.304690  |state|=+0.304690
ts=10.00  out[0]=+0.351644  |state|=+0.351644
```

Small `ts` barely advances the membrane; large `ts` relaxes it toward
its steady state (the output saturates). If your events arrive at
irregular gaps, drive `ts` per step — [cookbook recipe 5](cookbook.md#5-event-driven-sequences-with-variable-ts).

## How do I choose `units` and `unfolds`?

**Short answer:** `units` is model capacity (start small — liquid
cells need far fewer units than classical RNNs), `unfolds` is
integrator precision for the LTC only (the ncps default is 6; more
tightens the Euler solve at linearly growing graph cost). The CfC has
no `unfolds` — its closed-form step is constant-cost in the time span.

The costs, measured:

| cost | formula / number |
|---|---|
| cell parameters + masks, fully wired | `O(units²)` — load/construction peak `92·U²` bytes with `U = units = inDim` ([persistence.md](persistence.md)); `U = 2048` ≈ 368 MiB |
| construction of a fully wired `units = 1024` cell | ~32 MB measured (36.4 MiB LTC / 32.4 MiB CfC) since the sparse contraction — not the old ~8 GiB indicator cliff |
| LTC training graph per sequence | `∝ units × unfolds × seqLen` activation blocks held until `Backward` ([pitfalls.md](pitfalls.md) §9) |
| CfC training graph per sequence | `∝ units × seqLen` — no `unfolds` factor |
| per-step wall clock | LTC grows with `unfolds`; CfC constant in the time span |

So: keep `units` modest, keep `unfolds` at 4–8 unless you have a
reason, and if `seqLen × unfolds` blows up the graph, switch to the
CfC or shorten the unroll.

## A big `units` model doesn't fit in memory. What now?

**Short answer**, in order of payoff:

1. **Switch to the CfC** — it removes the `unfolds` factor from the
   training graph entirely ([cfc.md](cfc.md)). Same ODE, same 13
   parameters, analytic integrator.
2. **Sparsify the wiring.** `nn.RandomSparse(inDim, units, sensoryP,
   recurrentP, rng)` drops synapses independently with probability
   `1−p`; unwired synapses are arithmetically neutral, so the sparse
   contraction pays only for wiring that exists ([ltc.md](ltc.md),
   wiring section). Masks are drawn once at construction and immutable.
3. **Shorten the unrolled window** — the graph holds every
   intermediate tensor of every unrolled step until `Backward`
   ([pitfalls.md](pitfalls.md) §9); truncated BPTT over shorter
   segments is the standard relief.
4. Only then, reduce `units`.

```go
// sparse cell: ~50% of synapses exist, drawn once at construction
wiring := nn.RandomSparse(4, 256, 0.5, 0.5, rng)
cell := nn.NewLTC(4, 256, wiring, 6, rng)
cfc := nn.NewCfC(4, 256, wiring, rng) // same wiring works for either cell
```

(`NewLTC`/`NewCfC` themselves are not memory-capped — the `2048`
`units`/`inDim` caps are load-path only, because a load's input is an
untrusted stream; see [persistence.md](persistence.md).)

## Why are my gradients still growing after `Backward`?

**Short answer:** because leaf gradients **accumulate** across
`Backward` calls — across different graphs too — until you `ZeroGrad`.
It is the core gradient semantics, not a leak: recipe 3 (gradient
accumulation) is built on exactly this.

```go
a := autograd.New([]float32{1, 2, 3}, 3)
y := autograd.SumAll(autograd.Hadamard(a, a)) // dy/da = 2a
y.Backward()
fmt.Println(a.Grad.Data) // [2 4 6]
y.Backward()             // same graph, second run
fmt.Println(a.Grad.Data) // [4 8 12] — exactly doubled
```

The standard loop zeroes first: `ZeroGrad` every parameter → forward →
one `Backward` → update. Calling `Backward` twice on the *same* graph
doubles the gradient exactly (three calls, three times —
[pitfalls.md](pitfalls.md) §3). If your gradients look like an integer
multiple of what they should be, count your `Backward` and `ZeroGrad`
calls.

## Why doesn't `Step` call `ZeroGrad`?

**Short answer:** because zeroing is *your* contract, and an optimizer
that zeroed on your behalf would silently break gradient accumulation.

The accumulation semantics are the library's gradient contract;
`optimizer.Step` is deliberately just the update phase
([training.md](training.md#the-step-contract)). Zero before every
iteration for plain training; zero once every `N` iterations while
running `Backward` after each micro-batch, and the same `Step` call gives you
gradient accumulation for free — measured in
[cookbook recipe 3](cookbook.md#3-gradient-accumulation) to reproduce
the full-batch gradient to `3.6e-7` (float32 addition order).

## Should I use the LTC or the CfC?

**Short answer:** same ODE, two integrators — pick by cost, not by
expressiveness. The CfC's closed-form step has constant graph cost
(no `unfolds`); the LTC's Euler loop costs `unfolds` substeps per step
but is the reference implementation's exact scheme. Start with the
CfC when memory or long sequences hurt; choose the LTC to reproduce
ncps dynamics. They share everything else — parameters, wiring, `ts`
contract — and are drop-in swappable behind `nn.Cell` (same seed even
gives bit-identical initialization).

Decision table, swap demo and measured comparison:
[cookbook recipe 8](cookbook.md#8-choosing-between-ltc-and-cfc).
Paper↔code for each cell: [ltc.md](ltc.md), [cfc.md](cfc.md).

## Do last-digit differences across machines matter?

**Short answer:** same architecture: no difference is acceptable —
same seeds are bit-identical there, and the library's own tests pin
bit patterns. Across architectures (arm64 vs amd64): up to 1 ULP per
fused multiply-add contraction, and chains accumulate (measured ≤ 6
ULP on CfC Box-Muller initialization). That is Go's doing — the
language permits fusing float operations into one rounding — not the
library's.

The grading the golden-vector tests use ([persistence.md](persistence.md),
"Golden vectors"):

| what | guarantee |
|---|---|
| wire format layout (magic, version, shapes, counts, endianness) | byte-frozen on **every** platform |
| float payloads, same platform/toolchain | bit-for-bit (`Float32bits`, `NaN`/`−0` included) |
| float payloads, other architectures | within a 16 ULP window (~2.7× the observed maximum; tight enough that 32 ULP fails as corruption) |

So: compare bit patterns with `math.Float32bits` when asserting on
one machine, allow a small ULP window across machines, and never
string-compare printed decimals — see
[cookbook recipe 12](cookbook.md#12-deterministic-reproduction) for
the seed discipline that makes same-machine runs reproducible.

## How do I read load errors like "stream holds model kind …"?

**Short answer:** every load-path failure is an `error` (never a
panic), and the message tells you the bucket. Read the prefix:
`nn:` = model level, `serialize:` = tensor-stream level, `optimizer:`
= state-stream level.

| message | meaning | action |
|---|---|---|
| `nn: stream holds model kind 1 (CfC), not LTC (kind 0)` | wrong loader for this file | use `nn.LoadCfC` — the message names what the file actually is |
| `serialize: bad magic … not an LNN tensor stream` | not an LNN file at all | check you pointed at the right file |
| `serialize: unsupported format version 99 (this build reads version 1): … newer version …` | written by a newer library | update the library |
| `…: no earlier layout exists, the stream is corrupt or forged` | version byte below 1 | corruption — reject |
| `…: truncated stream: claims N data bytes but only M remain: unexpected EOF` | file cut short (match with `errors.Is(err, io.ErrUnexpectedEOF)`) | re-transfer; a failing load left your model untouched |
| `nn: LTC header has units=4096, exceeding the load limit 2048` | header over the load-path caps (also `unfolds` 1024, `inDim` 2048) | hostile or oversized — the check fired before any blob allocation |

All messages above are real outputs from
[cookbook recipe 10](cookbook.md#10-loading-untrusted-model-files-safely),
which has the full classify-and-react pattern. The contract behind
them — fixed limits, validate-before-allocate, zero side effects on
failure — is in [persistence.md](persistence.md#the-untrusted-stream-safety-contract).

## How do I resume training with Adam?

**Short answer:** three streams, not one — the model, any extra
modules' parameters, and the optimizer state; then load all three
into fresh objects and keep the hyperparameters identical.
[cookbook recipe 4](cookbook.md#4-checkpoint-and-resume-bit-exact) is
the complete program; measured there: 50 steps → checkpoint → 50
resumed steps match an uninterrupted 100-step run **bit for bit**
(every per-step loss and the final parameters, `Float32bits`).

```go
// save
must(nn.SaveCfC(modelFile, cell))                        // the cell (SaveLTC for LTCs)
must(serialize.WriteParameters(paramFile, readout.Parameters()))
must(optimizer.SaveState(stateFile, opt, params))        // Adam's m, v, t, bias powers

// resume
loaded, err := nn.LoadCfC(modelFile)
readC := nn.NewLinear(units, 1, rng)                     // seed irrelevant
must(serialize.LoadParameters(paramFile, readC.Parameters()))
optC := optimizer.NewAdamDefault(lr)                     // SAME hyperparameters
must(optimizer.LoadState(stateFile, optC, nn.ParametersOf(loaded, readC)))
```

The details that bite: the model checkpoint alone silently resets
Adam's adaptation (bias correction restarts at `t = 0`); the
`params` order at Load must match Save (state is keyed by index);
and hyperparameters are deliberately not in the `"LNO1"` stream —
`LoadState` verifies the saved `Beta^t` powers against the
destination optimizer, so a betas mismatch fails loudly. Format and
contracts: [persistence.md](persistence.md) (optimizer state section).
