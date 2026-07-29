# Architecture

> English | [中文](zh/architecture.md)

**Summary:** lnn is three layers — a `float32` numeric kernel (`tensor`), a
dynamic reverse-mode AD engine (`autograd`), and a model layer (`nn`) — each
importing only the one below it, with no framework, no code generation and no
dependencies beyond the standard library.

**Audience:** engineers debugging, profiling, or extending the library.

## Layer overview

```
┌──────────────────────────────────────────────────────────────┐
│  your code                                                   │
│  data, model assembly, loss, hand-rolled optimizer loop      │
└───────────────┬──────────────────────────────────────────────┘
                │  Step/Forward, Unroll, ParametersOf,
                │  ZeroGrad → Backward → explicit param update
┌───────────────▼──────────────────────────────────────────────┐
│  nn — model layer                                            │
│  Linear · LTC · Wiring · Cell/Unroll · Module/ParametersOf   │
└───────────────┬──────────────────────────────────────────────┘
                │  composes differentiable ops into a graph of
                │  *autograd.Variable
┌───────────────▼──────────────────────────────────────────────┐
│  autograd — dynamic graph + reverse-mode AD                  │
│  Variable{Data, Grad, parents, backward closure}             │
│  forward: eager; each op records its backward closure        │
│  Backward: reverse-topological walk, accumulation into leaves│
└───────────────┬──────────────────────────────────────────────┘
                │  every forward and backward computation is a
                │  plain tensor op
┌───────────────▼──────────────────────────────────────────────┐
│  tensor — numeric kernel                                     │
│  dense row-major []float32 · MatMul · broadcast elementwise  │
│  activations · reductions · slicing · RNG                    │
└──────────────────────────────────────────────────────────────┘
```

Import direction is strictly downward (`nn → autograd → tensor`); there are
no cycles and no cross-layer shortcuts except that `nn` calls `tensor`
directly for constants (wiring masks, epsilon) that need no gradient.

| Layer | Responsibility | What it deliberately lacks |
|---|---|---|
| `tensor` | Dense buffers and numeric kernels; validation by panic | strides, views, in-place ops, broadcasting beyond an enumerated subset |
| `autograd` | Eager forward + recorded backward closures; reverse-mode `Backward` | tape/session objects, graph optimization, higher-order derivatives |
| `nn` | Layers, cells, wiring, sequence unrolling, parameter aggregation | optimizers, serialization, CfC cells (roadmap) |

## Data flow of one training iteration

```
ZeroGrad(all params)            // clear leaf Grad buffers
        │
forward: ops execute eagerly    // each op allocates a new tensor and
        │                       // attaches a backward closure to its output
        ▼
   loss (scalar leaf of the graph)
        │
loss.Backward()                 // topo-sort the graph, run closures in
        │                       // reverse; leaf grads accumulate
        ▼
your update rule                // p.Data.Data[i] -= lr * p.Grad.Data[i]
```

A complete loop is in the root `README.md` quick start and in
[training.md](training.md); `examples/ltc-sequence` is the end-to-end
reference.

## Why tensors have no strides

`Tensor` is exactly `{Shape []int, Data []float32}`. There is no stride
array, so there are no views: every operation — including `SliceRow`,
`SliceCol`, `Transpose` and `Clone` — allocates a fresh buffer and copies.

The trade-off is deliberate for a library of this size:

- **Aliasing is impossible by construction.** Two tensors never share
  storage unless the user explicitly shares their `Data` slices, so there
  is no read-after-write hazard class to document or defend against, and
  every backward closure can capture its forward tensors without lifetime
  bookkeeping.
- **Immutability of constructed objects is real.** `Wiring` masks are
  exposed only through copying accessors, and the red-team audit confirmed
  that tampering with a returned mask cannot affect the cell (a pointer
  accessor would not have held that property).
- **The cost is explicit and predictable:** O(elements) copies per op and
  memory that grows with the graph (below). The library targets small,
  auditable models on CPU where clarity beats throughput; a strided
  view layer would be a different library.

MatMul does skip zero multipliers in its inner loop, which is the one
place where observable behavior differs from naive math: `0 * NaN`
contributes `0` rather than poisoning the result (see
[pitfalls.md](pitfalls.md)).

## The computation graph

Each `autograd.Variable` is a node:

```go
type Variable struct {
    Data *tensor.Tensor   // forward value
    Grad *tensor.Tensor   // accumulated gradient (nil until first backward)

    parents  []*Variable  // inputs of the op that produced this node
    backward func()       // closure: pushes Grad into parents
}
```

- **Leaves** (`Var`, `New`, `Const`) have no parents and no closure. They
  are parameters and inputs; gradients end their walk here.
- **Ops** run the forward computation eagerly via `tensor`, then record a
  closure on the output node. The closure reads `out.Grad` at backward
  time and calls `parent.addGrad(...)` for each input, reducing shapes
  with `tensor.SumToShape` where broadcasting happened. A few closures
  (e.g. `Log`) read a *parent's* `Data` at backward time, so leaf data
  must not be mutated between forward and backward
  ([pitfalls.md](pitfalls.md) has the details).
- **Backward** builds the topological order with a post-order DFS, runs
  the closures in reverse, and then clears `Grad` on every non-leaf node
  except the receiver.

### Non-leaf gradients are set to nil after each traversal — and why

Intermediate gradients are transient by design. If stale intermediate
`Grad` buffers survived a traversal, a second `Backward` on the same graph
would propagate through already-seeded nodes and leaf gradients would grow
super-linearly (the red team measured 3x for two calls on the pre-fix
code). Clearing them makes reruns exactly linear: N calls on the same
graph give N times the single-pass leaf gradient. The supported pattern is
still one `Backward` per freshly built graph; the linear rerun behavior is
a defined safety net, not a feature to build on
([pitfalls.md](pitfalls.md)).

### The graph is the memory model

Every intermediate tensor stays alive — referenced by its node's closure —
until `Backward` completes. Memory therefore scales with the number of ops
per iteration, not just with parameter size. One LTC `Step` unrolls
`unfolds` ODE iterations of O(units²) synapse ops into the graph, so keep
`units`, `unfolds` and sequence length modest on this engine.

## float32 is a global constraint

All storage and every public API is `float32`; there is no `float64` mode
and no mixed-precision story. Kernels that need headroom promote to
`float64` internally and round back — `Sigmoid`, `Tanh`, `Exp`, `Log`,
`Pow`, `Softplus` evaluate `math.*` in `float64`; `LogSoftmaxRows`
accumulates its normalizing sum in `float64`; `Randn` runs Box-Muller in
`float64`; and `LTC.scaledCapacitance` computes and clamps the `cm/dt`
scale in `float64` before converting (see [ltc.md](ltc.md)). Consequences
to plan with:

- ~7 decimal digits; accumulation error grows with reduction length.
- `Exp` of values above ~88 overflows to `+Inf`; there is no global guard
  (softmax families are internally stabilized, everything else is plain
  arithmetic).
- No subnormal-friendly kernels; denormal magnitudes simply flush toward
  zero.

## Concurrency

Single-threaded by design — `Backward` mutates leaf `Grad` buffers without
synchronization, and tensors expose storage directly. The supported
parallel pattern is per-goroutine instances (own cell, tensors, RNG,
graph), never shared state. See [pitfalls.md](pitfalls.md) for the
contract and a verified pattern.
