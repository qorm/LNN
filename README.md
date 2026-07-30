# lnn

> [简体中文](README_zh.md)

[![CI](https://github.com/qorm/LNN/actions/workflows/ci.yml/badge.svg)](https://github.com/qorm/LNN/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/qorm/LNN.svg)](https://pkg.go.dev/github.com/qorm/LNN)

A pure-Go numerical computation library: dense `float32` tensors, reverse-mode
automatic differentiation, and Liquid Time-Constant (LTC) neural cells —
with **zero third-party dependencies** (the standard library is the only
import). The LTC implementation follows Hasani et al.,
[Liquid Time-constant Networks](https://ojs.aaai.org/index.php/AAAI/article/view/17017)
(AAAI 2021) and the reference implementation
[`mlech26l/ncps`](https://github.com/mlech26l/ncps).

lnn is small and explicit. It favors readable, auditable kernels over breadth:
no code generation, no GPU backend, no operator overloading tricks — just Go.

## Packages

| Package | Purpose |
|---|---|
| `github.com/qorm/LNN/tensor` | Dense row-major `float32` tensors with a 1D/2D-focused op set: matmul, elementwise math with limited broadcasting, activations, reductions, slicing, random initialization. |
| `github.com/qorm/LNN/autograd` | A dynamic computation-graph engine. Each op tags its output `Variable` with an op kind; `Backward` walks the graph in reverse topological order, dispatches each node's gradient propagation, and accumulates gradients into leaves. |
| `github.com/qorm/LNN/nn` | Neural-network building blocks: `Linear` layers, `Wiring` synapse topologies, the `LTC` liquid cell and its closed-form sibling `CfC`, and the `Cell`/`Unroll` abstractions for driving recurrent cells over sequences. |
| `github.com/qorm/LNN/optimizer` | Explicit parameter-update rules over `autograd`: SGD, heavy-ball Momentum, and Adam (Kingma & Ba, bias-corrected). One `Step(params)` call replaces the hand-rolled update loop. |
| `github.com/qorm/LNN/serialize` | Versioned binary persistence: a compact little-endian tensor stream (`"LNNS"`, version 1) whose load path treats input as untrusted — every failure an error (never a panic), size claims validated before allocation, progressive allocation on unknown-length readers. The storage layer behind `nn`'s six Save/Load functions. |

## Documentation

Guides for building on the library live in [`doc/`](doc/):

| Guide | Covers |
|---|---|
| [doc/training.md](doc/training.md) | Hand-rolled training loops and the `optimizer` package (SGD/Momentum/Adam), gradient clipping, a divergence checklist |
| [doc/persistence.md](doc/persistence.md) | The `"LNNS"` wire format spec, the six Save/Load functions, the untrusted-stream safety contract, a runnable train→save→load→resume example |
| [doc/shapes-and-broadcasting.md](doc/shapes-and-broadcasting.md) | The broadcasting rule table, reduction output shapes, asymmetric conventions |
| [doc/ltc.md](doc/ltc.md) | The LTC paper↔code correspondence, parameter table, `ts` contract, wiring |
| [doc/cfc.md](doc/cfc.md) | The CfC closed-form cell: Lemma 1 paper correspondence, exprel stabilization, relation to the LTC |
| [doc/architecture.md](doc/architecture.md) | Three-layer design, computation-graph mechanics, the `float32` constraint |
| [doc/pitfalls.md](doc/pitfalls.md) | Concurrency contract, overflow scenarios, residual risks, roadmap |

Start with [doc/README.md](doc/README.md) for a suggested reading order.
Per-package API reference is available via godoc: `go doc github.com/qorm/LNN/tensor`,
`go doc github.com/qorm/LNN/autograd`, `go doc github.com/qorm/LNN/nn`.

## Installation

The module path is `github.com/qorm/LNN`; fetch it with:

```
go get github.com/qorm/LNN@latest
```

To work from source instead, clone the repository and point your app's
`go.mod` at the checkout with a `replace` directive:

```
git clone https://github.com/qorm/LNN.git
```

```go
// your app's go.mod
module myapp

go 1.26

require github.com/qorm/LNN v0.0.0

replace github.com/qorm/LNN => ../LNN
```

Inside the repository itself, `make build` / `make test` work as-is.

## Quick start

Training loops are written by hand against `autograd` — the basis for
understanding the library — and the `optimizer` package packages that
same loop as SGD/Momentum/Adam for production use. The program below
fits `y = 2x + 1` with a hand-rolled linear model and plain SGD; it
uses only `tensor` and `autograd`, and runs with `go run`. Replacing
the hand-rolled update with `optimizer.NewSGD(lr)` + `Step(params)`
reproduces the output exactly — see [doc/training.md](doc/training.md).

Plain `float32` SGD has no built-in stabilization: use modest learning rates,
and consider clipping the global gradient norm on larger problems
(`examples/ltc-sequence` clips to max-norm 1.0).

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

	// Training data: y = 2x + 1, stored as [n,1] matrices.
	const n = 32
	xs, ys := make([]float32, n), make([]float32, n)
	for i := range xs {
		xs[i] = rng.Float32()*2 - 1 // U(-1, 1)
		ys[i] = 2*xs[i] + 1
	}
	x := autograd.Const(tensor.FromData(xs, n, 1))
	y := autograd.Const(tensor.FromData(ys, n, 1))

	// Model parameters: y = x*w + b.
	w := autograd.Var(tensor.Randn(rng, 1, 1))
	b := autograd.Var(tensor.New(1))

	const epochs, lr = 200, 0.1
	for epoch := 0; epoch < epochs; epoch++ {
		// Forward: build a fresh graph each iteration.
		pred := autograd.Add(autograd.MatMul(x, w), b) // [n,1]
		diff := autograd.Sub(pred, y)                  // [n,1]
		loss := autograd.MeanAll(autograd.Hadamard(diff, diff))

		// Backward: gradients accumulate into the leaves w and b.
		w.ZeroGrad()
		b.ZeroGrad()
		loss.Backward()

		// Hand-rolled SGD step.
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

Actual output (Go 1.26, seed 42):

```
epoch   0  loss=1.398637  w=0.5503  b=0.1674
epoch  40  loss=0.006162  w=1.8658  b=0.9795
epoch  80  loss=0.000047  w=1.9883  b=0.9982
epoch 120  loss=0.000000  w=1.9990  b=0.9998
epoch 160  loss=0.000000  w=1.9999  b=1.0000
epoch 199  loss=0.000000  w=2.0000  b=1.0000
```

The loop recovers `w ≈ 2` and `b ≈ 1` to `float32` precision.

### Using the `nn` package

```go
rng := rand.New(rand.NewSource(1))

// A fully connected layer: y = x @ W + b.
fc := nn.NewLinear(4, 8, rng)   // W: [4,8], B: [8] (Xavier-uniform init)
y := fc.Forward(x)              // x: [batch,4] -> y: [batch,8]
params := fc.Parameters()       // []*autograd.Variable{W, B}

// A Liquid Time-Constant cell. Reversal potentials are fixed +/-1 constants
// and are not part of Parameters().
cell := nn.NewLTC(4, 8, nil, 6, rng) // inDim=4, units=8, fully connected wiring, 6 ODE unfolds
out, h := cell.Step(x, nil, 0.1)     // x: [batch,4], nil = zero initial state, time span ts=0.1

// Unroll any nn.Cell over a sequence; the whole sequence stays in the graph,
// so a loss built on ys differentiates through time with one Backward.
ys, hN := nn.Unroll(cell, xs, nil, 0.1) // xs: []*autograd.Variable of [batch,4]

// CfC is the closed-form sibling cell: same ODE and same 13-parameter
// synapse parameterization, but no unfolds — the Lemma 1 closed-form
// update advances the full span ts in a single step (doc/cfc.md).
cfc := nn.NewCfC(4, 8, nil, rng)
out2, h2 := cfc.Step(x, nil, 0.1)
```

`examples/ltc-sequence` puts this together into a complete training loop
(hand-rolled SGD over `nn.ParametersOf(cell, readout)`) on a toy sequence
task — run it with `go run ./examples/ltc-sequence`.
`examples/cfc-sequence` runs the same task with the CfC cell and the
recommended production form — caller-owned gradient-norm clipping plus
`optimizer.NewSGD` + `Step` — with the loss falling from `0.620651` to
`0.029091`.

## Numerics and scale

- **`float32` everywhere.** Tensor data is a flat `[]float32` in row-major
  layout. There is no `float64` mode.
- **1D/2D focused.** `MatMul` is defined for matrices only; `Rows`/`Cols`
  panic on anything but 2D. Elementwise ops work on any shape.
- **Broadcasting is an explicit, enumerated subset** — not general
  NumPy-style broadcasting. Binary elementwise ops accept exactly:
  - identical shapes,
  - a scalar (size-1 tensor) against anything,
  - a row vector (`[n]` or `[1,n]`) against a `[m,n]` matrix,
  - a column vector (`[m,1]`) against a `[m,n]` matrix,
  - `[m,1]` with `[1,n]`, producing the outer product `[m,n]`.

  Anything else panics with a descriptive message.
- **Shape conventions are not fully symmetric** (e.g. `SumRows` returns
  `[1,n]` while `SumCols` returns `[m]`, and 1D⊕1D results are promoted to
  `[1,n]`). Read the doc comments in `tensor/ops.go` before relying on a
  reduction's output shape.
- **The graph is retained until `Backward`.** Every intermediate tensor is
  kept alive by the computation graph, so memory grows with the number of
  ops. An LTC step unrolls `unfolds` ODE iterations into the graph, each
  `O(units)` vector blocks plus two MatMul contractions — down from
  `O(units²)` per-synapse nodes since the synapse vectorization; the
  phase-7 backward overhaul halved the per-node allocations again, and the
  phase-8 Sigmoid–Hadamard fusion trimmed them further
  (measured: `LTCStep` 2,306 allocs/op, `UnrollBackward` 31,983 —
  cumulative −69%/−73% from the original loop). Keep `units`, `unfolds`
  and sequence length modest on this engine; the `CfC` cell
  ([doc/cfc.md](doc/cfc.md)) has no `unfolds` factor at all.

## Concurrency contract

**lnn is single-threaded by design.**

- `Backward` mutates the `Grad` buffers of leaf variables; running it
  concurrently on variables that share parameters is a data race and loses or
  corrupts gradients (verified empirically under `go test -race`).
- `Variable` and `Tensor` expose their storage (`Data`) directly and carry no
  synchronization. Do not read or write one tensor from multiple goroutines.
- `math/rand.Rand` used for initialization is not goroutine-safe either.

The supported pattern for parallel workloads: give each goroutine its own
tensors, variables, and RNG, and never share a graph or its parameters across
goroutines.

## Status and roadmap

Honest maturity assessment as of this commit (coverage measured with
`go test -cover`, in-package):

| Package | Status |
|---|---|
| `tensor` | Core is stable and well tested (~99.7% line coverage). The single residual uncovered statement is a double-constant fill-loop body in `broadcastBinary` that is argued unreachable (its column count is always `1` on that path, so the loop never executes, and the `[1,1]×[1,1]` case is intercepted by the same-shape fast path first); it is documented rather than padded with a contrived test. The transpose-aware MatMul kernels added in phase 7 are exercised through the `autograd` package's tests. |
| `autograd` | Stable and well tested (100% line coverage); gradients pass finite-difference and bitwise-differential checks across the covered paths, including the phase-7 legacy-composition fallback branches for irregular manually seeded gradients and the phase-8 Sigmoid–Hadamard fusion's regular and fallback paths. |
| `nn` | Functional and well tested (100% line coverage): the LTC and CfC forward/backward paths are regression-tested, including closed-form degenerate-case checks, tiny/NaN `ts` guards, wiring validation, and Save/Load round-trips. Reversal potentials are fixed ±1 constants in both cells, baked into construction-time indicator matrices — not trainable, and with no dead gradient. CfC is a phase-6 feature and its API may still evolve. |
| `optimizer` | Stable, 100% line coverage: the three update rules are verified against independent reference implementations (SGD bit-for-bit, Adam ~1.6e-6 vs a float64 reference), and the pointer-keyed state semantics are regression-tested. |
| `serialize` | Stable, 97.8% line coverage: round-trip bit-exactness (NaN and −0 included) is regression-tested and byte-pinned by committed golden vectors; the hostile-stream contract — fixed limits validated before allocation (including the load-path `units`/`inDim` cap of 256 that bounds the O(units³) indicator matrices), progressive allocation on unknown-length readers — is pinned by allocation-count and byte-budget tests; and red-team mutation fuzzing produced zero panics across 7,500 mutants, plus a further 1,200 after the resource-exhaustion hardening. The resource bounds are documented in [doc/persistence.md](doc/persistence.md). |

The CfC (Closed-form Continuous-time) cell and built-in optimizers
shipped in phase 6, and serialization plus the autograd backward
overhaul in phase 7: `nn.CfC` ([doc/cfc.md](doc/cfc.md)) is a feature
whose API may still evolve; the `optimizer` package
(SGD/Momentum/Adam) and the `serialize` package plus the six `nn`
Save/Load functions ([doc/persistence.md](doc/persistence.md)) are
stable. The hand-rolled loop remains valid and is still the basis for
understanding the engine. Sequence unrolling is covered by the generic
`nn.Unroll` helper; `examples/ltc-sequence` shows the end-to-end
training pattern with hand-rolled SGD, and `examples/cfc-sequence` the
same task with the CfC cell and the recommended optimizer form (loss
`0.621 → 0.029`). Phase 8 closed the Sigmoid–Hadamard fused backward
(#13) and the CfC's `erev` dead gradients (#10), and added the
serialization golden vectors plus the load-path `units`/`inDim` caps.
The remaining roadmap is the technical-debt table in
[doc/pitfalls.md](doc/pitfalls.md) — headlined by the indicator-matrix
O(units³) materialization (#14, whose load side those caps already
close) and `tensor.New`'s fixed per-node overhead (#12).

Track the remediation plan and progress in `PLAN.md` and `PROGRESS.md`.

## Acknowledgments

lnn is an independent, from-scratch Go implementation built on the Liquid
Neural Networks research. We gratefully acknowledge:

- **Ramin Hasani, Mathias Lechner, Alexander Amini, Daniela Rus, and
  Radu Grosu** for Liquid Time-constant Networks (AAAI 2021) and
  Closed-form Continuous-time neural networks
  (*Nature Machine Intelligence* 4, 992–1003, 2022) — the equations this
  library implements;
- **Mathias Lechner** for [`mlech26l/ncps`](https://github.com/mlech26l/ncps),
  the reference implementation our LTC and wiring semantics were verified
  against;
- the authors of [`raminmh/CfC`](https://github.com/raminmh/CfC), the
  official CfC code used to cross-check the closed-form cell;
- the teams at **MIT CSAIL**, **TU Wien**, **IST Austria**, and
  **Liquid AI** for advancing and open-sourcing liquid neural networks.

All errors and design trade-offs in this Go port are our own.

## Development

```
make          # gofmt -w, go vet, go test ./... -count=1 -race
make cover    # full test run plus coverage.txt and a per-function summary
make build    # go build ./...
```

## License

MIT — see [LICENSE](LICENSE).
