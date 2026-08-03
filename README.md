# LNN

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

LNN is small and explicit. It favors readable, auditable kernels over breadth:
no code generation, no GPU backend, no operator overloading tricks — just Go.

In the current literature, liquid cells are read as non-linear state-space
models (SSMs): Liquid-S4 ([arXiv:2209.12951](https://arxiv.org/abs/2209.12951),
ICLR 2023) put LTC dynamics inside the S4 state-space formulation, and LrcSSM
([arXiv:2505.21717](https://arxiv.org/abs/2505.21717), NeurIPS 2025) scaled
liquid dynamics into a non-linear SSM layer that processes long sequences in
linear time. From that angle, this library's cells are interpretable,
variable-`ts`, sparsely-wired non-linear SSM cells —
[doc/cfc.md](doc/cfc.md) works out the formula-level correspondence between
the CfC update step and Mamba's selective-SSM step.

## Packages

| Package | Purpose |
|---|---|
| `github.com/qorm/LNN/tensor` | Dense row-major `float32` tensors with a 1D/2D-focused op set: matmul, elementwise math with limited broadcasting, activations, reductions, slicing, random initialization. Since v0.4.0, shapes up to rank 4 are stored inline in the tensor (one fewer heap allocation per construction), and `Tensor.Reshape` re-points a shape without reallocating. |
| `github.com/qorm/LNN/autograd` | A dynamic computation-graph engine. Each op tags its output `Variable` with an op kind; `Backward` walks the graph in reverse topological order, dispatches each node's gradient propagation, and accumulates gradients into leaves. |
| `github.com/qorm/LNN/nn` | Neural-network building blocks: `Linear` layers, `Wiring` synapse topologies, the `LTC` liquid cell and its closed-form sibling `CfC`, and the `Cell`/`Unroll` abstractions for driving recurrent cells over sequences. |
| `github.com/qorm/LNN/optimizer` | Explicit parameter-update rules over `autograd`: SGD, heavy-ball Momentum, Adam (Kingma & Ba, bias-corrected), AdEMAMix (ICLR 2025, dual gradient EMAs), and Schedule-Free AdamW (NeurIPS 2024, iterate averaging instead of a decay schedule, with a Train/Eval conversion contract). One `Step(params)` call replaces the hand-rolled update loop. State persistence (`SaveState`/`LoadState`, `"LNO1"` streams) makes resumed training bit-identical to uninterrupted training. |
| `github.com/qorm/LNN/serialize` | Versioned binary persistence: a compact little-endian tensor stream (`"LNNS"`, version 1) whose load path treats input as untrusted — every failure an error (never a panic), size claims validated before allocation, progressive allocation on unknown-length readers. The storage layer behind `nn`'s six Save/Load functions. |

## Documentation

Guides for building on the library live in [`doc/`](doc/):

| Guide | Covers |
|---|---|
| [doc/training.md](doc/training.md) | Hand-rolled training loops and the `optimizer` package (SGD/Momentum/Adam/AdEMAMix/Schedule-Free AdamW), gradient clipping, a divergence checklist |
| [doc/persistence.md](doc/persistence.md) | The `"LNNS"` wire format spec, the six Save/Load functions, optimizer state persistence (`"LNO1"` streams, bit-exact resume), the untrusted-stream safety contract, runnable train→save→load→resume examples |
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

go 1.26.5

require github.com/qorm/LNN v0.0.0

replace github.com/qorm/LNN => ../LNN
```

Inside the repository itself, `make build` / `make test` work as-is.

## Quick start

Training loops are written by hand against `autograd` — the basis for
understanding the library — and the `optimizer` package packages that
same loop as SGD/Momentum/Adam/AdEMAMix/Schedule-Free AdamW for production use. The program below
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

_Output recorded on arm64 (Apple Silicon); on other architectures the trailing digits may differ slightly due to floating-point contraction (Go permits fused multiply-add rounding)._

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

// Long sequences: UnrollRemat differentiates lossFn(ys) through time with
// the same gradients as Unroll + loss.Backward(), bit for bit, at
// O(chunkSize) peak graph memory instead of O(len(xs)). params must list
// every trainable leaf the cell's Step consumes (completeness is audited);
// lossFn builds the scalar loss from the detached per-step outputs.
ys, hN, loss := nn.UnrollRemat(cell, cell.Parameters(), xs, nil, 0.1, 8, lossFn)

// CfC is the closed-form sibling cell: same ODE and same 13-parameter
// synapse parameterization, but no unfolds — the Lemma 1 closed-form
// update advances the full span ts in a single step (doc/cfc.md).
cfc := nn.NewCfC(4, 8, nil, rng)
out2, h2 := cfc.Step(x, nil, 0.1)

// Since stage 18 that closed-form step is one fused graph node too —
// transparent to the caller: same API, forward and gradients
// bit-identical, ~34 µs and 52 allocs per step on this machine.
```

Choose `ts` to match your task's sampling interval: if each sequence step
corresponds to one time unit of the underlying process, use `ts = 1.0`.
The full `ts` contract — positive and finite, with a finiteness-only
domain below `ts ≈ 1e-38` — is in [doc/ltc.md](doc/ltc.md).

The examples form a graded path — start with hello-train and hello-use
(a few dozen lines each), then read the full training loops, then go
further with checkpointing and diagnostics. They live in the repository
— clone it (`git clone https://github.com/qorm/LNN.git`) and run from
the repository root; `go get` installations can browse them on GitHub.
Task-oriented recipes (gradient accumulation, checkpointing, custom
losses, event-driven time steps, and more) are in
[doc/cookbook.md](doc/cookbook.md) (中文版：[doc/zh/cookbook.md](doc/zh/cookbook.md)).

**Start here**

- `examples/hello-train` — the smallest possible training program: fit
  `y = 2x + 1` with one CfC neuron and hand-written gradient descent
  (~60 numbered lines: model, data, forward → backward → update), then
  save the model.
- `examples/hello-use` — the smallest possible inference program: load
  that saved model and run one forward step, printing predictions next
  to the true line (~30 lines, no training code).

**Full training loops**

- `examples/ltc-sequence` — a complete training loop (hand-rolled SGD
  over `nn.ParametersOf(cell, readout)`, with global gradient-norm
  clipping) on a toy bounded-accumulator sequence task:
  `go run ./examples/ltc-sequence`.
- `examples/cfc-sequence` — the same task with the CfC cell and the
  recommended production form — caller-owned clipping plus
  `optimizer.NewSGD` + `Step` — with the loss falling from `0.620651`
  to `0.029091`.

**Going further**

- `examples/ltc-resume` — checkpoints a trained LTC (model, readout
  parameters and Adam state via `optimizer.SaveState`) and resumes it
  into fresh objects built with a different seed, asserting the resumed
  trajectory is bit-identical to uninterrupted training.
- `examples/gradient-inspect` — trains a small LTC while printing
  per-parameter gradient L2 norms, the global max-abs gradient,
  NaN/Inf counts and parameter-update magnitudes every K iterations —
  including a raw-vs-clipped norm comparison — as a template for
  diagnosing stalled or diverging runs.

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
  ops. An LTC step — sensory synapses and `unfolds` ODE iterations alike —
  runs as **one fused graph node** (34 nodes per step at any dimensions;
  forward and gradients bit-identical to the former graph path), built on
  the same sparse presynaptic contraction — a
  `+0`-seeded fold ended by normalizing MatMuls, with no dense
  `[units², units]` indicator matrices anywhere (construction of a
  fully-wired `units = 1024` cell costs ~32 MB, not the ~8 GiB the
  earlier dense-indicator design required). Successive backward,
  allocation and fusion overhauls have brought the measured counts down
  to `LTCStep` **77 allocs/op** (~78 µs, 23 KB) and `UnrollBackward`
  **3,750** (~1.29 ms, 558 KB) on this machine (`-benchtime=200x`) —
  about 99%/97% below the original per-synapse loop, and −67%/−44%
  allocations against v0.5.2 (stage 19 moved the sensory path into the
  fused kernel, reuses the VJP's scratch planes, pools `Backward`'s DFS
  scratch, and fixed-array broadcast shapes). Keep `units`, `unfolds` and
  sequence length modest on this engine; the `CfC` cell
  ([doc/cfc.md](doc/cfc.md)) has no `unfolds` factor at all and fuses
  its whole step into one node as well (52 allocs/op, ~34 µs), and for
  genuinely long sequences `nn.UnrollRemat` differentiates through time
  at O(chunk) peak graph memory instead of O(sequence) — T = 512
  measured ~0.65 MB retained vs ~8.3 MB live for a full unroll
  (memory semantics and the worst case:
  [doc/architecture.md](doc/architecture.md),
  [doc/pitfalls.md](doc/pitfalls.md); a full recipe:
  [doc/cookbook.md](doc/cookbook.md#13-long-sequence-training-chunked-bptt-with-unrollremat)).

## Concurrency contract

**LNN is single-threaded by design.**

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
| `tensor` | Core is stable and well tested (~99.7% line coverage). The single residual uncovered statement is a double-constant fill-loop body in `broadcastBinary` that is argued unreachable (its column count is always `1` on that path, so the loop never executes, and the `[1,1]×[1,1]` case is intercepted by the same-shape fast path first); it is documented rather than padded with a contrived test. The transpose-aware MatMul kernels are exercised through the `autograd` package's tests. The v0.4.0 embedded shape backing removed one heap allocation per tensor construction (benchmark allocations −18~−26%, wall-clock neutral within noise). |
| `autograd` | Stable and well tested (100% line coverage); gradients pass finite-difference and bitwise-differential checks across the covered paths, including the legacy-composition fallback branches for irregular manually seeded gradients and the Sigmoid–Hadamard fusion's regular and fallback paths. |
| `nn` | Functional and well tested (~99.9% line coverage — the sole uncovered statement is a constructively unreachable nil-root firewall guard in `UnrollRemat`'s unit sweep, annotated in `nn/remat.go`; same disclosure discipline as the optimizer row): the LTC and CfC forward/backward paths are regression-tested, including closed-form degenerate-case checks, tiny/NaN `ts` guards, wiring validation, and Save/Load round-trips (a legitimate `units = 2048` stream round-trips at the load limit). Reversal potentials are fixed ±1 constants in both cells, carried by row-view constants over the sparse contraction — not trainable, and with no dead gradient (structurally impossible). The sparse contraction is regression-tested bitwise against the former indicator-matrix implementation, with a large-cell memory gate. Both cells' hot paths run as fused `FusedOp` nodes (the LTC's sensory drive and ODE unfolds — 34 nodes per step at any dimensions since stage 19a, with the VJP's scratch planes allocated once per backward; the CfC's whole closed-form step since stage 18), each regression-tested bitwise against its pre-fusion graph path across cell shapes, loss shapes, chained/stacked topologies and adversarial non-finite inputs, and gated by three bitwise differential fuzz targets (11 native fuzz targets total, CI gated on amd64 and arm64); `UnrollRemat`'s chunked BPTT is regression-tested bit-for-bit against whole-graph backward over a configuration × loss-shape differential matrix. CfC is the newer cell, and its API may still evolve. |
| `optimizer` | Stable, ~99.8% line coverage (the uncovered remainder is the physically unreachable parameter-count guard): the five update rules are verified against independent reference implementations (SGD bit-for-bit, Adam ~1.6e-6, AdEMAMix ~9.5e-6 and ScheduleFreeAdamW ~3.3e-7 against third-party float64 references of their papers, discrimination-gated), the pointer-keyed state semantics are regression-tested, ScheduleFreeAdamW's train/eval mode contract panics before modifying anything when stepping an eval-mode parameter, and state persistence (`SaveState`/`LoadState`, `"LNO1"` streams, kinds 0–4) is pinned by bit-exact resume tests (50+50 vs 100 steps, all five optimizers, warmup-boundary-crossing included) and hostile-stream tests (validate-all-then-apply with zero side effects, byte-budget gates). |
| `serialize` | Stable, 97.8% line coverage: round-trip bit-exactness (NaN and −0 included) is regression-tested and byte-pinned by committed golden vectors; the hostile-stream contract — fixed limits validated before allocation (including the load-path `units`/`inDim` cap of 2048, sized for the sparse contraction's O(units²) load-time memory), progressive allocation on unknown-length readers — is pinned by allocation-count and byte-budget tests; and mutation fuzzing produced zero panics across thousands of mutants, plus a further round after the resource-exhaustion hardening. The resource bounds are documented in [doc/persistence.md](doc/persistence.md). |

The CfC (Closed-form Continuous-time) cell and the built-in optimizers
are the newest additions: `nn.CfC` ([doc/cfc.md](doc/cfc.md)) is a
feature whose API may still evolve; the `optimizer` package
(SGD/Momentum/Adam/AdEMAMix/Schedule-Free AdamW) and the `serialize` package plus the six `nn`
Save/Load functions ([doc/persistence.md](doc/persistence.md)) are
stable. The hand-rolled loop remains valid and is still the basis for
understanding the engine. Sequence unrolling is covered by the generic
`nn.Unroll` helper; `examples/ltc-sequence` shows the end-to-end
training pattern with hand-rolled SGD, and `examples/cfc-sequence` the
same task with the CfC cell and the recommended optimizer form (loss
`0.621 → 0.029`). Later hardening kept the engine lean without breaking
documented usage: the sparse synaptic contraction eliminated the
O(units³) indicator matrices (construction ~32 MB at `units = 1024`,
not ~8 GiB), optimizer state persistence (`SaveState`/`LoadState`,
`"LNO1"` streams) made resumed training bit-exact for all three
optimizers, and an API-hygiene pass embedded shapes inline in each
tensor (benchmark allocations −18~−26%, wall-clock neutral; the
exported `Tensor.Reshape` replacing direct `Shape` writes), deleted the
unused `tensor.Stack`, and froze the asymmetric reduction conventions
after evaluation (decision trail in
[doc/shapes-and-broadcasting.md](doc/shapes-and-broadcasting.md)). A
later pass closed the last performance backlog item (fixed-capacity
parent slots in the graph node: −10~−22% allocs, bitwise-equivalent),
and the latest work fused the LTC's ODE unfolds into a single graph
node — ~2.1–2.3× on the step and unroll benchmarks, bit-exact against
the former graph path — and added `nn.UnrollRemat`, chunked
backpropagation through time with bit-exact gradients at O(chunk) peak
memory (mechanism in [doc/architecture.md](doc/architecture.md), recipe
13 in [doc/cookbook.md](doc/cookbook.md)). The remaining roadmap is the
technical-debt table in
[doc/pitfalls.md](doc/pitfalls.md): every remaining entry is
accepted-risk or informational grade.

The full development history and audit trail is recorded in
`PROGRESS.md`.

## Acknowledgments

LNN is an independent, from-scratch Go implementation built on the Liquid
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
