# lnn

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
| `lnn/tensor` | Dense row-major `float32` tensors with a 1D/2D-focused op set: matmul, elementwise math with limited broadcasting, activations, reductions, slicing, random initialization. |
| `lnn/autograd` | A dynamic computation-graph engine. Each op records a backward closure on its output `Variable`; `Backward` walks the graph in reverse topological order and accumulates gradients into leaves. |
| `lnn/nn` | Neural-network building blocks: `Linear` layers, `Wiring` synapse topologies, the `LTC` liquid cell, and the `Cell`/`Unroll` abstractions for driving recurrent cells over sequences. |

## Installation

The module path is the bare name `lnn` and there is no vanity import URL, so
`go get` over the network does not resolve it. Until the module is published,
vendor it with a `replace` directive:

```
git clone <this repository> LNN
```

```go
// your app's go.mod
module myapp

go 1.26

require lnn v0.0.0

replace lnn => ../LNN
```

Inside the repository itself, `make build` / `make test` work as-is.

## Quick start

The library ships no optimizer — training loops are written by hand against
`autograd`, which is exactly what the library is designed for. The program
below fits `y = 2x + 1` with a hand-rolled linear model and plain SGD.
It uses only `tensor` and `autograd`, and runs with `go run`.

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
```

`examples/ltc-sequence` puts this together into a complete training loop
(hand-rolled SGD over `nn.ParametersOf(cell, readout)`) on a toy sequence
task — run it with `go run ./examples/ltc-sequence`.

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
  ops. An LTC step unrolls `unfolds` ODE iterations of `O(units²)` synapse
  ops into the graph — keep `units` and `unfolds` modest on this engine.

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

Honest maturity assessment as of this commit:

| Package | Status |
|---|---|
| `tensor` | Core is stable and well tested (~86% line coverage). Some defensive checks (overflow-safe sizing, empty-input edge cases) are being hardened. |
| `autograd` | Stable and well tested (~98% line coverage); gradients pass finite-difference checks on the covered paths. |
| `nn` | Functional and well tested (~98% line coverage): the LTC forward/backward paths are regression-tested, including a closed-form degenerate-case check, tiny/NaN `ts` guards, and wiring validation. Reversal potentials are fixed ±1 constants, not trainable. The API may still evolve. |

Roadmap (not yet implemented): the CfC (Closed-form Continuous-time) cell
and built-in optimizers. Sequence unrolling is covered by the generic
`nn.Unroll` helper, and `examples/ltc-sequence` shows the supported
training pattern with hand-rolled SGD.

Track the remediation plan and progress in `PLAN.md` and `PROGRESS.md`.

## Development

```
make          # gofmt -w, go vet, go test ./... -count=1 -race
make cover    # full test run plus coverage.txt and a per-function summary
make build    # go build ./...
```

## License

MIT — see [LICENSE](LICENSE).
