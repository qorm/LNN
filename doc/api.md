# API quick reference

> English | [中文](zh/api.md)

**What this page is:** a single-page, package-organized navigation layer over
the library's public surface — every exported symbol in one line, with its
panic/error profile and a pointer to the canonical documentation. **It is not
a godoc copy:** the authoritative contract for each symbol (full parameter
semantics, panic conditions, error classes, bit-level notes) lives in the
godoc, and the concepts behind the signatures live in the guides linked at the
bottom.

## How to read the API

The godoc is the contract. Three ways in:

```sh
go doc github.com/qorm/LNN/nn.NewLTC      # one symbol
go doc github.com/qorm/LNN/tensor          # a whole package (panics, layout, broadcasting)
go doc -all github.com/qorm/LNN/autograd   # every symbol in a package
```

Online: [pkg.go.dev/github.com/qorm/LNN](https://pkg.go.dev/github.com/qorm/LNN)
([tensor](https://pkg.go.dev/github.com/qorm/LNN/tensor) ·
[autograd](https://pkg.go.dev/github.com/qorm/LNN/autograd) ·
[nn](https://pkg.go.dev/github.com/qorm/LNN/nn) ·
[optimizer](https://pkg.go.dev/github.com/qorm/LNN/optimizer) ·
[serialize](https://pkg.go.dev/github.com/qorm/LNN/serialize)).

**Marker column:** `panic` = misuse or bad input panics (the library's default
discipline — inputs come from your program, so a bad shape is a caller bug);
`error` = failures are returned as errors (the persistence paths, which
consume untrusted byte streams — never a panic there); `—` = no failure mode
beyond the obvious (nil dereference on a nil receiver, etc.).

Two contracts cut across the whole table:

- **Shapes** — `[m, n]` means a 2D row-major tensor; the broadcasting rules
  are an enumerated subset, and the reduction output shapes are asymmetric.
  See [shapes-and-broadcasting.md](shapes-and-broadcasting.md) before relying
  on any output shape.
- **Gradients accumulate** — leaf `Grad` buffers add up across `Backward`
  calls; `ZeroGrad` every parameter before each backward pass
  ([training.md](training.md)).

## tensor — [godoc](https://pkg.go.dev/github.com/qorm/LNN/tensor) · [shapes](shapes-and-broadcasting.md)

Dense row-major `float32` tensors. `Tensor` is just `Shape []int` + flat
`Data []float32`; every op allocates a fresh buffer (no views, no aliasing
unless you share `Data` deliberately).

### Construction

| symbol | signature | semantics | fails |
|---|---|---|---|
| [`Tensor`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor) | `struct{ Shape []int; Data []float32 }` | dense row-major tensor; element `(i,j)` of `[m,n]` is `Data[i*n+j]` | — |
| [`New`](https://pkg.go.dev/github.com/qorm/LNN/tensor#New) | `New(shape ...int) *Tensor` | zero-filled; `New()` is rank-0, one zero | panic: negative dim, int64 overflow |
| [`FromData`](https://pkg.go.dev/github.com/qorm/LNN/tensor#FromData) | `FromData(data []float32, shape ...int) *Tensor` | copies `data` (no aliasing) | panic: size≠len(data), negative dim, overflow |
| [`FromRows`](https://pkg.go.dev/github.com/qorm/LNN/tensor#FromRows) | `FromRows(rows ...[]float32) *Tensor` | `[len(rows), len(rows[0])]`, copies; no rows → `[0,0]` | panic: ragged rows |
| [`Reshape`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.Reshape) | `(*Tensor).Reshape(dims ...int)` | re-points `Shape` without touching `Data`; rank > 4 falls back to a heap shape | panic: negative dim; element count is NOT checked |
| [`Clone`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.Clone) | `(*Tensor).Clone() *Tensor` | deep copy, shares no storage | — |
| [`ZerosLike`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.ZerosLike) / [`OnesLike`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.OnesLike) | `(*Tensor) … () *Tensor` | zero-/one-filled, same shape | — |
| [`SameShape`](https://pkg.go.dev/github.com/qorm/LNN/tensor#SameShape) | `SameShape(a, b *Tensor) bool` | identical dimension lists | — |

### Linear algebra (2D only)

| symbol | signature | semantics | fails |
|---|---|---|---|
| [`MatMul`](https://pkg.go.dev/github.com/qorm/LNN/tensor#MatMul) | `MatMul(a, b) *Tensor` | `[m,k] × [k,n] → [m,n]`; matrices only | panic: non-2D, inner dims differ |
| [`MatMulTransA`](https://pkg.go.dev/github.com/qorm/LNN/tensor#MatMulTransA) | `MatMulTransA(a, b) *Tensor` | `aᵀ·b`, `[m,k] & [m,n] → [k,n]`, no transpose buffer | panic: non-2D, row counts differ |
| [`MatMulTransB`](https://pkg.go.dev/github.com/qorm/LNN/tensor#MatMulTransB) | `MatMulTransB(a, b) *Tensor` | `a·bᵀ`, `[m,k] & [n,k] → [m,n]`, no transpose buffer | panic: non-2D, col counts differ |
| [`Transpose`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Transpose) | `Transpose(a) *Tensor` | `[m,n] → [n,m]` | panic: non-2D |

### Elementwise (any rank; binary ops = enumerated broadcasting)

| symbol | signature | semantics | fails |
|---|---|---|---|
| [`Add`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Add) / [`Sub`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Sub) / [`Hadamard`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Hadamard) | `(a, b *Tensor) *Tensor` | `+`, `−`, elementwise `*` under the enumerated broadcast rules; 1D⊕1D → `[1,n]`, `[1]⊕[1]` → `[1,1]`, `[m,1]⊗[n]` → outer product `[m,n]` | panic: not broadcastable |
| [`Scale`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Scale) | `Scale(a, s float32) *Tensor` | every element × s | — |
| [`Neg`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Neg) | `Neg(a) *Tensor` | `Scale(a, -1)` | — |
| [`Apply`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Apply) | `Apply(a, f func(float32) float32) *Tensor` | map `f` in flat row-major order | — |
| [`Tanh`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tanh) / [`Sigmoid`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Sigmoid) / [`Exp`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Exp) / [`Log`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Log) / [`Pow`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Pow) / [`Softplus`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Softplus) | unary: `(a) *Tensor`; `Pow(a, p)` | standard activations/math, stable sigmoid & softplus; no domain checks — `Log`/`Pow` yield NaN/Inf as float32 dictates | — |
| [`Clip`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Clip) | `Clip(a, lo, hi float32) *Tensor` | clamp to `[lo, hi]`; expects `lo ≤ hi` | — |

### Reduction

| symbol | signature | semantics | fails |
|---|---|---|---|
| [`SumAll`](https://pkg.go.dev/github.com/qorm/LNN/tensor#SumAll) | `SumAll(a) *Tensor` | total → shape `[1]`; empty → 0 | — |
| [`MeanAll`](https://pkg.go.dev/github.com/qorm/LNN/tensor#MeanAll) | `MeanAll(a) *Tensor` | mean → shape `[1]` | panic: empty tensor |
| [`SumRows`](https://pkg.go.dev/github.com/qorm/LNN/tensor#SumRows) | `SumRows(a) *Tensor` | axis-0 sums of `[m,n]` → **`[1,n]`** (stays 2D) | panic: non-2D |
| [`SumCols`](https://pkg.go.dev/github.com/qorm/LNN/tensor#SumCols) | `SumCols(a) *Tensor` | axis-1 sums of `[m,n]` → **`[m]`** (1D) — asymmetric with SumRows, frozen convention | panic: non-2D |
| [`SumToShape`](https://pkg.go.dev/github.com/qorm/LNN/tensor#SumToShape) | `SumToShape(grad, shape) *Tensor` | backward reducer: identity→clone, scalar→`[1]` total, `[n]`/`[1,n]`→column sums, `[m,1]`→row sums (full table in the guide) | panic: any other target |
| [`SoftmaxRows`](https://pkg.go.dev/github.com/qorm/LNN/tensor#SoftmaxRows) / [`LogSoftmaxRows`](https://pkg.go.dev/github.com/qorm/LNN/tensor#LogSoftmaxRows) | `(a) *Tensor` | per-row (log-)softmax of `[m,n]`, max-subtracted stable; zero columns → empty `[m,0]` | panic: non-2D |

### Slicing & access

| symbol | signature | semantics | fails |
|---|---|---|---|
| [`SliceCol`](https://pkg.go.dev/github.com/qorm/LNN/tensor#SliceCol) | `SliceCol(a, from, to int) *Tensor` | columns `[from,to)` → `[m, to-from]` copy | panic: non-2D, invalid/empty range |
| [`SliceRow`](https://pkg.go.dev/github.com/qorm/LNN/tensor#SliceRow) | `SliceRow(a, i int) *Tensor` | row `i` → `[1,n]` copy | panic: non-2D, `i` out of range |
| [`ConcatCol`](https://pkg.go.dev/github.com/qorm/LNN/tensor#ConcatCol) | `ConcatCol(ts ...*Tensor) *Tensor` | `[m,n1],[m,n2],… → [m, Σn]` | panic: none given, non-2D, row mismatch |
| [`At`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.At) / [`Set`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.Set) | `At(idx ...int) float32` / `Set(v float32, idx ...int)` | index one element, any rank | panic: wrong index count, out of bounds |
| [`Rows`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.Rows) / [`Cols`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.Cols) | `(*Tensor) … () int` | first/second dimension | panic: non-2D |
| [`Dims`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.Dims) / [`Size`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.Size) | `(*Tensor) … () int` | rank / element count | panic (Size): int64 overflow |
| [`Scalar`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.Scalar) / [`IsScalar`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.IsScalar) / [`IsRowVec`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.IsRowVec) | see godoc | one-element test/extract; row-vector test (`[n]` or `[1,n]`) | panic (Scalar): not size-1 |
| [`String`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.String) | `(*Tensor).String() string` | debug render; > 64 elements summarized | — |

### Random

| symbol | signature | semantics | fails |
|---|---|---|---|
| [`Uniform`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Uniform) | `Uniform(rng, lo, hi float32, shape ...int) *Tensor` | U(lo, hi); `lo > hi` mirrors the interval (legacy) | panic: nil rng, negative dim |
| [`Randn`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Randn) | `Randn(rng, shape ...int) *Tensor` | N(0,1) Box-Muller; tails truncated at ≈ 7.43σ (documented) | panic: nil rng, negative dim |

## autograd — [godoc](https://pkg.go.dev/github.com/qorm/LNN/autograd) · [training](training.md) · [architecture](architecture.md)

Dynamic computation graph with reverse-mode autodiff. Every op runs eagerly
and tags its output node; `Backward` walks the graph in reverse topological
order. The whole graph stays resident until `Backward` runs.

### Graph & leaves

| symbol | signature | semantics | fails |
|---|---|---|---|
| [`Variable`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Variable) | `struct{ Data, Grad *tensor.Tensor }` | graph node: value + accumulated gradient | — |
| [`Var`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Var) | `Var(t) *Variable` | leaf; **aliases** `t` (no copy) — in-place updates work | — |
| [`New`](https://pkg.go.dev/github.com/qorm/LNN/autograd#New) | `New(data []float32, shape ...int) *Variable` | leaf; copies data (`tensor.FromData`) | panic: shape/length mismatch |
| [`Const`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Const) | `Const(t) *Variable` | `Var` alias documenting constant intent; gradients still flow — ignore them | — |
| [`Detach`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Detach) | `Detach(v) *Variable` | `v`'s value as a fresh parentless leaf — **aliases** `v.Data` (zero copy); cuts gradient flow into `v`'s ancestors, not storage (an in-place update still moves a detached parameter) | — |

### Ops (each returns a fresh node; forward panics = the wrapped `tensor` op's panics)

| symbol | signature | semantics | fails |
|---|---|---|---|
| [`MatMul`](https://pkg.go.dev/github.com/qorm/LNN/autograd#MatMul) | `(a, b) *Variable` | `[m,k]×[k,n]→[m,n]` | panic: non-2D, inner dims |
| [`Add`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Add) / [`Sub`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Sub) / [`Hadamard`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Hadamard) | `(a, b) *Variable` | broadcast ops; backward reduces to operand shapes via `SumToShape` | panic: not broadcastable |
| [`Scale`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Scale) / [`Neg`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Neg) | `Scale(a, s)` / `Neg(a)` | constant scale / negate | — |
| [`Tanh`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Tanh) / [`Sigmoid`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Sigmoid) / [`Exp`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Exp) / [`Log`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Log) / [`Pow`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Pow) / [`Softplus`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Softplus) / [`Abs`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Abs) / [`Relu`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Relu) | unary `(a) *Variable`; `Pow(a, p)` | fused backwards; `Pow(_, 0)` gradient is exactly 0; `Abs`/`Relu` take gradient 0 at 0 | — |
| [`Div`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Div) | `(a, b) *Variable` | quotient rule; `b==0` → ±Inf forward (float32 division) | — |
| [`SigmoidHadamard`](https://pkg.go.dev/github.com/qorm/LNN/autograd#SigmoidHadamard) | `(z, w) *Variable` | fused `Sigmoid(z)⊙w` (LTC/CfC hot path), one node + reused sigmoid buffer | panic: not broadcastable |
| [`ConcatCol`](https://pkg.go.dev/github.com/qorm/LNN/autograd#ConcatCol) | `(vs ...*Variable) *Variable` | `[m, Σn]`; backward slices gradients back out | panic: none given, non-2D, row mismatch |
| [`SliceCol`](https://pkg.go.dev/github.com/qorm/LNN/autograd#SliceCol) / [`Col`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Col) | `SliceCol(a, from, to)` / `Col(a, i)` | `[m, to-from]` / `[m, 1]`; backward zero-pads | panic: non-2D, invalid range |
| [`SliceRow`](https://pkg.go.dev/github.com/qorm/LNN/autograd#SliceRow) | `(a, i) *Variable` | `[1, n]`; backward zero-pads | panic: non-2D, `i` out of range |
| [`SumAll`](https://pkg.go.dev/github.com/qorm/LNN/autograd#SumAll) / [`MeanAll`](https://pkg.go.dev/github.com/qorm/LNN/autograd#MeanAll) | `(a) *Variable` | scalar `[1]`; backward broadcasts `g` / `g/size` | panic (MeanAll): empty |
| [`GatherRows`](https://pkg.go.dev/github.com/qorm/LNN/autograd#GatherRows) | `(a, idx []int) *Variable` | `out[i] = a[i, idx[i]]` → 1D `[rows]`; idx copied | panic: non-2D, len(idx)≠rows, idx out of bounds |
| [`LogSoftmaxRows`](https://pkg.go.dev/github.com/qorm/LNN/autograd#LogSoftmaxRows) | `(a) *Variable` | stable per-row log-softmax, fused backward | panic: non-2D |
| [`FusedOp`](https://pkg.go.dev/github.com/qorm/LNN/autograd#FusedOp) | `FusedOp(data, parents, backward) *Variable` | custom op node: the caller computes the forward, the closure runs the node's **entire** backward — every `addGrad` to the parents plus the replaced subgraph's accumulation order is the closure's contract (the fused-LTC integration point, `nn/ltc_fused.go`) | panic: nil backward |

### Backward & inspection

| symbol | signature | semantics | fails |
|---|---|---|---|
| [`Backward`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Variable.Backward) | `(*Variable).Backward()` | reverse-mode from a scalar receiver; leaf gradients **accumulate** across calls and graphs; non-leaf grads cleared | panic: non-scalar receiver without a seeded `Grad` |
| [`ZeroGrad`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Variable.ZeroGrad) | `(*Variable).ZeroGrad()` | `Grad = nil` — call before every backward pass | — |
| [`Value`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Variable.Value) | `(*Variable).Value() float32` | read a size-1 node (a loss) | panic: not size-1 |
| [`TopoOrder`](https://pkg.go.dev/github.com/qorm/LNN/autograd#TopoOrder) | `TopoOrder(v) []*Variable` | DFS post-order of the graph rooted at `v` (parents before children, in construction order) — exactly `Backward`'s build order; read-only introspection | — |
| [`Parents`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Variable.Parents) | `(*Variable).Parents() []*Variable` | the node's parent list in construction order (empty for leaves); a **fresh copy per call** — mutating it cannot rewire the node | — |

## nn — [godoc](https://pkg.go.dev/github.com/qorm/LNN/nn) · [ltc](ltc.md) · [cfc](cfc.md) · [persistence](persistence.md)

### Modules

| symbol | signature | semantics | fails |
|---|---|---|---|
| [`Module`](https://pkg.go.dev/github.com/qorm/LNN/nn#Module) | `interface{ Parameters() []*autograd.Variable }` | anything owning trainable parameters | — |
| [`ParametersOf`](https://pkg.go.dev/github.com/qorm/LNN/nn#ParametersOf) | `ParametersOf(modules ...Module) []*autograd.Variable` | flatten; **order is positional contract** for the persistence APIs | — |
| [`Linear`](https://pkg.go.dev/github.com/qorm/LNN/nn#Linear) | `struct{ W, B *autograd.Variable }` | `y = x @ W + b`; `W [in,out]`, `B [out]` | — |
| [`NewLinear`](https://pkg.go.dev/github.com/qorm/LNN/nn#NewLinear) | `NewLinear(in, out int, rng) *Linear` | Xavier-uniform `W`, zero `B` | panic: nil rng, negative dims |
| [`Forward`](https://pkg.go.dev/github.com/qorm/LNN/nn#Linear.Forward) | `(*Linear).Forward(x) *autograd.Variable` | `[batch,in] → [batch,out]` | panic: non-2D `x`, width mismatch |
| [`Parameters`](https://pkg.go.dev/github.com/qorm/LNN/nn#Linear.Parameters) | `(*Linear).Parameters() []*autograd.Variable` | fixed order: `W`, `B` | — |

### Cells & sequences

| symbol | signature | semantics | fails |
|---|---|---|---|
| [`Cell`](https://pkg.go.dev/github.com/qorm/LNN/nn#Cell) | `interface{ Step(x, h, ts) (out, hNew); StateSize() int }` | one RNN step; `x [batch,inDim]`, `h [batch,units]` or nil | — |
| [`Unroll`](https://pkg.go.dev/github.com/qorm/LNN/nn#Unroll) | `Unroll(cell, xs, h0, ts) (ys, hN)` | thread a cell over a sequence in one graph — one `Backward` through time; empty `xs` → empty `ys`, `hN = h0` | panic: per `Step` |
| [`UnrollRemat`](https://pkg.go.dev/github.com/qorm/LNN/nn#UnrollRemat) | `UnrollRemat(cell, params, xs, h0, ts, chunkSize, lossFn) (ys, hN, loss)` | chunked BPTT with rematerialization (gradient checkpointing): the gradients of `Unroll` + `lossFn` + `loss.Backward()` **bit for bit**, with peak graph memory O(chunkSize) instead of O(len(xs)) — an adversarial loss visit order can exceed even full unroll (worst case, documented); `params` must list **every** `Step`-consumed trainable leaf (completeness audited), and the cell's per-step graph shape must be value-independent; returned `ys`/`hN` are detached (safe to read, no graph behind them) | panic: chunkSize < 1, params audit, loss-side consumer order, multi-class shared leaf, per `Step` |
| [`LTC`](https://pkg.go.dev/github.com/qorm/LNN/nn#LTC) | struct | Liquid Time-Constant cell (Hasani 2021): semi-implicit Euler over `unfolds` substeps, softplus positivity constraints, fixed ±1 reversal potentials (not in `Parameters()`) | — |
| [`NewLTC`](https://pkg.go.dev/github.com/qorm/LNN/nn#NewLTC) | `NewLTC(inDim, units, wiring, unfolds, rng) *LTC` | nil wiring = fully connected; init ranges follow the reference; fixed seed → bit-identical cell | panic: dims < 1, unfolds < 1, mask shape mismatch, nil rng |
| [`CfC`](https://pkg.go.dev/github.com/qorm/LNN/nn#CfC) | struct | Closed-form Continuous-time cell (Hasani 2022): the LTC's ODE advanced by its Lemma-1 closed form, no unfolding | — |
| [`NewCfC`](https://pkg.go.dev/github.com/qorm/LNN/nn#NewCfC) | `NewCfC(inDim, units, wiring, rng) *CfC` | same parameterization as the LTC, minus `unfolds` | panic: dims < 1, mask shape mismatch, nil rng |
| [`Step`](https://pkg.go.dev/github.com/qorm/LNN/nn#LTC.Step) | `(*LTC/*CfC).Step(x, h, ts) (out, hNew)` | `out [batch,units]` (affine map), `hNew [batch,units]` raw state; **ts must be positive and finite** | panic: NaN/±Inf/≤0 ts, bad x/h shape |
| [`StateSize`](https://pkg.go.dev/github.com/qorm/LNN/nn#LTC.StateSize) | `(*LTC/*CfC).StateSize() int` | the `units` count | — |
| [`Parameters`](https://pkg.go.dev/github.com/qorm/LNN/nn#LTC.Parameters) | `(*LTC/*CfC).Parameters() []*autograd.Variable` | 13 variables, **frozen order** (stream order in Save*, positional key for optimizer state) | — |

### Wiring

| symbol | signature | semantics | fails |
|---|---|---|---|
| [`Wiring`](https://pkg.go.dev/github.com/qorm/LNN/nn#Wiring) | struct | binary synapse masks; immutable after construction, exposed only as copies | — |
| [`FullyConnected`](https://pkg.go.dev/github.com/qorm/LNN/nn#FullyConnected) | `FullyConnected(inDim, units) *Wiring` | every synapse present | panic: dims < 1 |
| [`RandomSparse`](https://pkg.go.dev/github.com/qorm/LNN/nn#RandomSparse) | `RandomSparse(inDim, units, sensoryP, recurrentP, rng) *Wiring` | each synapse independent with probability p | panic: dims < 1, p ∉ [0,1] (NaN rejected), nil rng |
| [`Sensory`](https://pkg.go.dev/github.com/qorm/LNN/nn#Wiring.Sensory) / [`Recurrent`](https://pkg.go.dev/github.com/qorm/LNN/nn#Wiring.Recurrent) | `(*Wiring) … () *tensor.Tensor` | mask copies: `[inDim, units]` / `[units, units]` | — |
| [`SensoryRow`](https://pkg.go.dev/github.com/qorm/LNN/nn#Wiring.SensoryRow) / [`RecurrentRow`](https://pkg.go.dev/github.com/qorm/LNN/nn#Wiring.RecurrentRow) | `(*Wiring) … (i int) *tensor.Tensor` | row `i` as `[1, units]` copy | panic: `i` out of range |

### Persistence (errors, never panics — see [persistence.md](persistence.md))

| symbol | signature | semantics | fails |
|---|---|---|---|
| [`SaveLTC`](https://pkg.go.dev/github.com/qorm/LNN/nn#SaveLTC) / [`LoadLTC`](https://pkg.go.dev/github.com/qorm/LNN/nn#LoadLTC) | `(w, *LTC) error` / `(r) (*LTC, error)` | kind 0 + header `inDim, units, unfolds` + 17 tensors; load bit-exact, seed-independent | error: I/O; load: wrong kind, truncation (`io.ErrUnexpectedEOF`), unfolds > 1024, units/inDim > 2048, bad masks/reversals, version skew |
| [`SaveCfC`](https://pkg.go.dev/github.com/qorm/LNN/nn#SaveCfC) / [`LoadCfC`](https://pkg.go.dev/github.com/qorm/LNN/nn#LoadCfC) | `(w, *CfC) error` / `(r) (*CfC, error)` | kind 1 + header `inDim, units` + 17 tensors | error: as LoadLTC minus unfolds |
| [`SaveLinear`](https://pkg.go.dev/github.com/qorm/LNN/nn#SaveLinear) / [`LoadLinear`](https://pkg.go.dev/github.com/qorm/LNN/nn#LoadLinear) | `(w, *Linear) error` / `(r) (*Linear, error)` | kind 2 + `W`, `B`; dims live in `W`'s shape | error: I/O; load: wrong kind, count ≠ 2, non-2D `W`, bias mismatch |

## optimizer — [godoc](https://pkg.go.dev/github.com/qorm/LNN/optimizer) · [training](training.md) · [persistence](persistence.md)

Explicit update rules packaging exactly the hand-rolled loop: `ZeroGrad` →
forward → `Backward` → `Step(params)`. `Step` never zeroes gradients and
skips parameters with nil `Grad`. Hyperparameters are exported, hot-swappable
fields.

### Optimizers

| symbol | signature | semantics | fails |
|---|---|---|---|
| [`Optimizer`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#Optimizer) | `interface{ Step(params []*autograd.Variable) }` | in-place update contract | — |
| [`SGD`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#SGD) / [`NewSGD`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#NewSGD) | `struct{ LR float32 }` / `NewSGD(lr) *SGD` | `p -= LR·g`, stateless | panic: lr ≤ 0 or NaN (+Inf accepted, produces Inf updates) |
| [`Momentum`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#Momentum) / [`NewMomentum`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#NewMomentum) | `struct{ LR, Mu float32 }` / `NewMomentum(lr, mu)` | heavy-ball: `v = Mu·v + g; p -= LR·v`; velocity keyed by parameter pointer | panic: lr ≤ 0, mu ∉ [0,1); Step: parameter resized |
| [`Adam`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#Adam) / [`NewAdam`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#NewAdam) | `struct{ LR, Beta1, Beta2, Eps float32 }` / `NewAdam(lr, b1, b2, eps)` | bias-corrected moments, float32 throughout; state keyed by parameter pointer | panic: lr ≤ 0, betas ∉ [0,1), eps ≤ 0; Step: parameter resized |
| [`NewAdamDefault`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#NewAdamDefault) | `NewAdamDefault(lr) *Adam` | Kingma & Ba defaults: 0.9 / 0.999 / 1e-8 | panic: lr ≤ 0 |
| [`AdEMAMix`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#AdEMAMix) / [`NewAdEMAMix`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#NewAdEMAMix) | `struct{ LR, Beta1, Beta2, Beta3, Alpha, Eps float32; Warmup int }` / `NewAdEMAMix(lr, b1, b2, b3, alpha, warmup, eps)` | Adam + a second, deliberately **uncorrected** slow gradient EMA mixed with α (arXiv:2409.03137, ICLR 2025); α-linear/β3-half-life warmup schedulers; no decoupled weight decay; state keyed by parameter pointer | panic: lr ≤ 0, betas ∉ [0,1), alpha < 0, warmup < 0, eps ≤ 0; Step: parameter resized |
| [`NewAdEMAMixDefault`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#NewAdEMAMixDefault) | `NewAdEMAMixDefault(lr, warmup) *AdEMAMix` | paper defaults: 0.9 / 0.999 / 0.9999 / α=5 / 1e-8 | panic: lr ≤ 0 |
| [`ScheduleFreeAdamW`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#ScheduleFreeAdamW) / [`NewScheduleFreeAdamW`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#NewScheduleFreeAdamW) | `struct{ LR, Beta1, Beta2, Eps, WeightDecay float32; WarmupSteps int }` / `NewScheduleFreeAdamW(lr, b1, b2, eps)` | schedule-free AdamW (arXiv:2405.15682, NeurIPS 2024): gradients evaluated at `y`, base AdamW on `z`, deployable weights are the averaged `x`; **params hold `y` during training** — `Eval`/`Train` convert; `WeightDecay` decoupled at `y`; official-v1.3+ bias correction | panic: lr ≤ 0, b1 ∉ (0,1), b2 ∉ [0,1), eps ≤ 0; Step: eval-mode parameter (index named) or resized; Train/Eval: nil data or resized |
| [`NewScheduleFreeAdamWDefault`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#NewScheduleFreeAdamWDefault) | `NewScheduleFreeAdamWDefault(lr) *ScheduleFreeAdamW` | defaults β1 0.9, β2 0.999, eps 1e-8 (official LR guidance: 1×–10× scheduled AdamW) | panic: lr ≤ 0 |
| [`Train`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#ScheduleFreeAdamW.Train) / [`Eval`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#ScheduleFreeAdamW.Eval) | `(*ScheduleFreeAdamW).Train(params)` / `Eval(params)` | in-place y↔x conversion of every state-carrying parameter (idempotent; stateless parameters untouched): `Eval` for evaluation/export, `Train` before the next `Step` | panic: nil data, parameter resized |

### State persistence (`"LNO1"` streams — errors, never panics)

| symbol | signature | semantics | fails |
|---|---|---|---|
| [`SaveState`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#SaveState) | `SaveState(w, o Optimizer, params) error` | per-parameter state **keyed by index** — same order required at Load; hyperparameters NOT saved; SGD → 19-byte identity stream; deterministic bytes; kinds 0–4 = SGD/Momentum/Adam/AdEMAMix/ScheduleFreeAdamW (kind 4 also carries each parameter's train/eval mode bit) | error: unsupported optimizer, nil param, I/O |
| [`LoadState`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#LoadState) | `LoadState(r, o Optimizer, params) error` | validate-all-then-apply: a failing load leaves `o` untouched; present record replaces, absent deletes, stale keys survive; resumed run bit-identical (an eval-mode ScheduleFree stream restores eval mode — `Step` then guards with a panic until `Train`) | error: bad magic/version/kind, count mismatch, shape mismatch, pow inconsistency (Adam/AdEMAMix/ScheduleFree), `t`/`k` > 2²⁴, non-finite `lrMax`/`wsum`, bad mode byte, truncation (`io.ErrUnexpectedEOF`) |

## serialize — [godoc](https://pkg.go.dev/github.com/qorm/LNN/serialize) · [persistence](persistence.md)

The `"LNNS"` wire format: magic, version, count, then per tensor rank +
shape + little-endian float32 payload. The exception domain: loads treat
input as an untrusted byte stream — every failure is an error, never a
panic, and a hostile stream allocates only in proportion to delivered bytes.

| symbol | signature | semantics | fails |
|---|---|---|---|
| [`Version`](https://pkg.go.dev/github.com/qorm/LNN/serialize#Version) | `const Version uint8 = 1` | frozen format version; unknown versions rejected, never guessed | — |
| [`WriteTensors`](https://pkg.go.dev/github.com/qorm/LNN/serialize#WriteTensors) | `WriteTensors(w, ts []*tensor.Tensor) error` | encode a tensor slice | error: nil tensor, rank > 8, count > 2²⁰, negative dim, shape/Data disagreement, overflow, I/O |
| [`ReadTensors`](https://pkg.go.dev/github.com/qorm/LNN/serialize#ReadTensors) | `ReadTensors(r) ([]*tensor.Tensor, error)` | decode; fixed limits before allocation; known-length readers checked up front, unknown-length grow progressively | error: bad magic, version skew (directional), limits, truncation (`io.ErrUnexpectedEOF`), trailing bytes |
| [`WriteParameters`](https://pkg.go.dev/github.com/qorm/LNN/serialize#WriteParameters) | `WriteParameters(w, params []*autograd.Variable) error` | `WriteTensors` over `p.Data`, in order (order = load key) | error: nil param/Data, WriteTensors errors |
| [`LoadParameters`](https://pkg.go.dev/github.com/qorm/LNN/serialize#LoadParameters) | `LoadParameters(r, params []*autograd.Variable) error` | copy values back **in place** (pointer identity preserved); all shapes validated first; **stale `Grad` deliberately kept** — `ZeroGrad` before reuse | error: nil param, count/shape mismatch, ReadTensors errors |

## Concept cross-index

| if you are asking… | read |
|---|---|
| "what shape comes out of this op?" | [shapes-and-broadcasting.md](shapes-and-broadcasting.md) — full broadcast table, asymmetric reductions |
| "how do I train this?" | [training.md](training.md) — the four-phase loop, clipping, hot-swapping, pointer-keyed state |
| "how do I checkpoint / is this file format safe?" | [persistence.md](persistence.md) — `"LNNS"`/`"LNO1"` byte by byte, bit-exact resume, untrusted-stream contract |
| "what does the LTC actually compute?" | [ltc.md](ltc.md) — ODE ↔ code, sparse contraction, parameter table |
| "how is the CfC different?" | [cfc.md](cfc.md) — Lemma-1 closed form, exprel stabilization |
| "how do the layers fit together?" | [architecture.md](architecture.md) — tensor → autograd → nn, op-tagged backward |
| "what can go wrong in production?" | [pitfalls.md](pitfalls.md) — concurrency, float32 overflow, repeated Backward, tiny `ts` |

Guide index: [doc/README.md](README.md). The repository root `README.md` has
the quick start; `examples/` holds complete runnable training loops.
