# lnn documentation

> English | [中文](zh/README.md)

Engineer-oriented guides for building on the `lnn` library. API details are
in the godoc of each package (`go doc lnn/tensor`, `go doc lnn/autograd`,
`go doc lnn/nn`); these guides cover the concepts, conventions and sharp
edges behind the signatures.

## Guides

| Guide | In one sentence |
|---|---|
| [training.md](training.md) | How to write the hand-rolled training loop this library is designed for: parameter aggregation, ZeroGrad/Backward discipline, plain SGD, and why gradient clipping matters. |
| [shapes-and-broadcasting.md](shapes-and-broadcasting.md) | The full broadcasting rule table, the (honestly asymmetric) reduction shape conventions, and a quick-reference for every output shape. |
| [ltc.md](ltc.md) | The Liquid Time-Constant cell, equation by equation against the code: ODE, semi-implicit Euler algebra, parameter table, the `ts` contract, and wiring. |
| [architecture.md](architecture.md) | The three-layer design (tensor → autograd → nn), why tensors have no strides, how the computation graph and its backward closures work. |
| [pitfalls.md](pitfalls.md) | Known boundaries and residual risks from the red-team audit, as user-facing caveats: concurrency, float32 overflow, repeated Backward, tiny `ts`, and the roadmap. |

A complete Simplified Chinese mirror of these guides lives in [`zh/`](zh/README.md), one file per row above.

## Suggested reading order

1. **[training.md](training.md)** — gets a model learning; everything else is reference.
2. **[shapes-and-broadcasting.md](shapes-and-broadcasting.md)** — the conventions you will hit within your first hour.
3. **[ltc.md](ltc.md)** — if you are here for liquid neural networks.
4. **[architecture.md](architecture.md)** — mental model for debugging and extending.
5. **[pitfalls.md](pitfalls.md)** — read before shipping.

The repository root `README.md` has the quick start; `examples/ltc-sequence`
is a complete, runnable training loop on a toy sequence task.
