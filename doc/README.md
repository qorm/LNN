# LNN documentation

> English | [中文](zh/README.md)

Engineer-oriented guides for building on the `LNN` library. API details are
in the godoc of each package (`go doc github.com/qorm/LNN/tensor`,
`go doc github.com/qorm/LNN/autograd`, `go doc github.com/qorm/LNN/nn`,
`go doc github.com/qorm/LNN/optimizer`, `go doc github.com/qorm/LNN/serialize`); these guides cover the concepts,
conventions and sharp edges behind the signatures.

The documentation has two axes: **concept guides** (what the library is
and why it is the way it is) and **task docs** (how to do a specific
thing). If you arrive with "I want to do X", start from
[cookbook.md](cookbook.md) or [faq.md](faq.md); if you want to
understand the engine, take the reading order below.

## Guides

| Guide | In one sentence |
|---|---|
| [training.md](training.md) | The training loop this library is designed for — hand-rolled (the basis) and via the `optimizer` package (SGD/Momentum/Adam/AdEMAMix/Schedule-Free AdamW, the recommended production form): parameter aggregation, ZeroGrad/Backward discipline, hyperparameter hot-swapping, pointer-keyed state, and why gradient clipping matters. |
| [persistence.md](persistence.md) | The `"LNNS"` wire format byte by byte, the six Save/Load APIs, optimizer state persistence (`optimizer.SaveState`/`LoadState`, `"LNO1"` streams — resumed training bit-identical to uninterrupted), runnable train → save → load → resume examples, and the untrusted-stream safety contract (validate before allocating, fixed limits, progressive allocation). |
| [shapes-and-broadcasting.md](shapes-and-broadcasting.md) | The full broadcasting rule table, the (honestly asymmetric) reduction shape conventions, and a quick-reference for every output shape. |
| [ltc.md](ltc.md) | The Liquid Time-Constant cell, equation by equation against the code: ODE, semi-implicit Euler algebra, the sparse synaptic contraction, parameter table, the `ts` contract, and wiring. |
| [cfc.md](cfc.md) | The Closed-form Continuous-time cell (Nature Machine Intelligence 2022): the Lemma 1 closed-form solution, Algorithm 1's LTC-to-closed-form compilation, exprel stabilization, and its relation to the LTC — same ODE, analytic integrator instead of Euler. |
| [architecture.md](architecture.md) | The three-layer design (tensor → autograd → nn) plus the optimizer and serialize packages, why tensors have no strides, how the computation graph's op-kind-tagged backward and fused gradient loops work. |
| [pitfalls.md](pitfalls.md) | Known boundaries and residual risks from the red-team audit, as user-facing caveats: concurrency, float32 overflow, repeated Backward, tiny `ts`, untrusted model files, and the technical-debt roadmap. |
| [cookbook.md](cookbook.md) | Task recipes, each a complete measured program: the minimal loop, Adam + clipping, gradient accumulation, bit-exact checkpoint/resume, variable-`ts` event-driven sequences, model inspection, custom losses, LTC-vs-CfC, multi-module composition, loading untrusted files, LR annealing, deterministic reproduction, long-sequence training with `UnrollRemat` (chunked BPTT), Schedule-Free AdamW and the train/eval contract. |
| [faq.md](faq.md) | Frequent questions, short answers: loss not falling, `NaN` losses, choosing `ts`/`units`/`unfolds`, gradient accumulation semantics, cross-platform last digits, reading load errors, resuming with Adam. |
| [api.md](api.md) | API quick reference: every exported symbol per package in one line, with a pointer to its canonical documentation (godoc / concept guides). (New — created by a parallel documentation task.) |
| [pgo.md](pgo.md) | Profile-guided optimization, measured honestly: why a library ships no profile, the workflow for profiling your own binary, and this repo's benchmark numbers — including the inlining decisions the headline gains hinge on. |

A complete Simplified Chinese mirror of these guides lives in [`zh/`](zh/README.md), one file per row above.

## Pick your path

**① First time with LNN.** Root [`README.md`](../README.md) quick start
(copy-paste runnable) → [training.md](training.md) (the loop and its
disciplines) → [cookbook.md](cookbook.md) recipes 1–3 (minimal loop,
optimizer + clipping, gradient accumulation). You can train after this.

**② Deploying or checkpointing.** [persistence.md](persistence.md)
(the format and the six Save/Load APIs) →
[cookbook.md](cookbook.md#4-checkpoint-and-resume-bit-exact) (train →
save → load → resume, bit-exact, including Adam state) →
[faq.md](faq.md#how-do-i-read-load-errors-like-stream-holds-model-kind-)
(reading load errors) → [pitfalls.md](pitfalls.md) §10 before loading
files from strangers.

**③ Migrating from ncps / PyTorch.** [ltc.md](ltc.md) and
[cfc.md](cfc.md) (the "Relation to the ncps reference" tables map each
ncps concept to its LNN counterpart — `elapsed_time` → `ts`,
`implicit_param_constraints` → softplus, defaults adopted verbatim) →
[shapes-and-broadcasting.md](shapes-and-broadcasting.md) (broadcasting
is an *enumerated subset*, not NumPy's; reduction shapes are
asymmetric — the two conventions that bite porters first) →
[pitfalls.md](pitfalls.md) (single-threaded by contract, caller-owned
clipping, no framework layers — what is deliberately missing).

**④ Auditing internals / contributing.** [architecture.md](architecture.md)
(the three layers and the graph mechanics) → the per-package godoc
(`go doc github.com/qorm/LNN/tensor`, `…/autograd`, `…/nn`,
`…/optimizer`, `…/serialize`) → the repository's `PLAN.md` and
`PROGRESS.md` (the full development and red-team audit trail, phase by
phase) → [pitfalls.md](pitfalls.md)'s technical-debt roadmap for what
is known and accepted.

## Suggested reading order

1. **[training.md](training.md)** — gets a model learning (hand-rolled loop, then the `optimizer` package); everything else is reference.
2. **[persistence.md](persistence.md)** — checkpoint what you trained, before diving into cell theory.
3. **[ltc.md](ltc.md)** — if you are here for liquid neural networks.
4. **[cfc.md](cfc.md)** — the closed-form sibling cell; read right after ltc.md, whose ODE it reuses.
5. **[shapes-and-broadcasting.md](shapes-and-broadcasting.md)** — the conventions you will hit within your first hour.
6. **[architecture.md](architecture.md)** — mental model for debugging and extending.
7. **[pitfalls.md](pitfalls.md)** — read before shipping.

The task docs cut across this order: [cookbook.md](cookbook.md) and
[faq.md](faq.md) link into whichever concept guide explains each
recipe's or answer's *why*, so you can enter from them at any point
(and the "Pick your path" section above routes by reader profile).

The repository root `README.md` has the quick start; `examples/ltc-sequence`
and `examples/cfc-sequence` are complete, runnable training loops on a toy
sequence task (hand-rolled SGD and the optimizer-package form, respectively).
They are part of the repository: clone it (`git clone https://github.com/qorm/LNN.git`)
and run them from the repository root.
