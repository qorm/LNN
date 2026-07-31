# lnn documentation

> English | [中文](zh/README.md)

Engineer-oriented guides for building on the `lnn` library. API details are
in the godoc of each package (`go doc github.com/qorm/LNN/tensor`,
`go doc github.com/qorm/LNN/autograd`, `go doc github.com/qorm/LNN/nn`,
`go doc github.com/qorm/LNN/serialize`); these guides cover the concepts,
conventions and sharp edges behind the signatures.

## Guides

| Guide | In one sentence |
|---|---|
| [training.md](training.md) | The training loop this library is designed for — hand-rolled (the basis) and via the `optimizer` package (SGD/Momentum/Adam, the recommended production form): parameter aggregation, ZeroGrad/Backward discipline, hyperparameter hot-swapping, pointer-keyed state, and why gradient clipping matters. |
| [persistence.md](persistence.md) | The `"LNNS"` wire format byte by byte, the six Save/Load APIs, optimizer state persistence (`optimizer.SaveState`/`LoadState`, `"LNO1"` streams — resumed training bit-identical to uninterrupted), runnable train → save → load → resume examples, and the untrusted-stream safety contract (validate before allocating, fixed limits, progressive allocation). |
| [shapes-and-broadcasting.md](shapes-and-broadcasting.md) | The full broadcasting rule table, the (honestly asymmetric) reduction shape conventions, and a quick-reference for every output shape. |
| [ltc.md](ltc.md) | The Liquid Time-Constant cell, equation by equation against the code: ODE, semi-implicit Euler algebra, the sparse synaptic contraction, parameter table, the `ts` contract, and wiring. |
| [cfc.md](cfc.md) | The Closed-form Continuous-time cell (Nature Machine Intelligence 2022): the Lemma 1 closed-form solution, Algorithm 1's LTC-to-closed-form compilation, exprel stabilization, and its relation to the LTC — same ODE, analytic integrator instead of Euler. |
| [architecture.md](architecture.md) | The three-layer design (tensor → autograd → nn) plus the optimizer and serialize packages, why tensors have no strides, how the computation graph's op-kind-tagged backward and fused gradient loops work. |
| [pitfalls.md](pitfalls.md) | Known boundaries and residual risks from the red-team audit, as user-facing caveats: concurrency, float32 overflow, repeated Backward, tiny `ts`, untrusted model files, and the technical-debt roadmap. |

A complete Simplified Chinese mirror of these guides lives in [`zh/`](zh/README.md), one file per row above.

## Suggested reading order

1. **[training.md](training.md)** — gets a model learning (hand-rolled loop, then the `optimizer` package); everything else is reference.
2. **[persistence.md](persistence.md)** — checkpoint what you trained, before diving into cell theory.
3. **[ltc.md](ltc.md)** — if you are here for liquid neural networks.
4. **[cfc.md](cfc.md)** — the closed-form sibling cell; read right after ltc.md, whose ODE it reuses.
5. **[shapes-and-broadcasting.md](shapes-and-broadcasting.md)** — the conventions you will hit within your first hour.
6. **[architecture.md](architecture.md)** — mental model for debugging and extending.
7. **[pitfalls.md](pitfalls.md)** — read before shipping.

The repository root `README.md` has the quick start; `examples/ltc-sequence`
and `examples/cfc-sequence` are complete, runnable training loops on a toy
sequence task (hand-rolled SGD and the optimizer-package form, respectively).
