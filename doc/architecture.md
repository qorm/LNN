# Architecture

> English | [中文](zh/architecture.md)

**Summary:** LNN is three layers — a `float32` numeric kernel (`tensor`), a
dynamic reverse-mode AD engine (`autograd`), and a model layer (`nn`) — each
importing only the one below it, with no framework, no code generation and no
dependencies beyond the standard library.

**Audience:** engineers debugging, profiling, or extending the library.

## Layer overview

```
┌──────────────────────────────────────────────────────────────┐
│  your code                                                   │
│  data, model, loss, update loop (hand-rolled or optimizer)   │
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
│  Variable{Data, Grad, parents, op kind + payload}            │
│  forward: eager; each op tags its output with an op kind     │
│  Backward: reverse-topological walk, tag-dispatched backward │
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
directly for constants (wiring masks, epsilon, the identity matrix and
`+0` scalar seeding the sparse contraction, the reversal row views) that
need no gradient. The `optimizer` package (SGD,
Momentum, Adam) sits beside `nn` on top of `autograd` — it imports only
`autograd` and writes the same plain-Go in-place updates a hand-rolled
loop would ([training.md](training.md)). The `serialize` package
(persistence) sits beside them on top of `tensor` and `autograd` — it is
the storage layer behind `nn`'s six Save/Load functions, exposed
separately so the wire format and its untrusted-stream safety contract
can be audited on their own ([persistence.md](persistence.md)).

| Layer | Responsibility | What it deliberately lacks |
|---|---|---|
| `tensor` | Dense buffers and numeric kernels; validation by panic | strides, views, in-place ops, broadcasting beyond an enumerated subset |
| `autograd` | Eager forward + op-kind-tagged backward dispatch; reverse-mode `Backward` | tape/session objects, per-node backward closures (replaced by a tag), graph optimization, higher-order derivatives |
| `nn` | Layers, cells (LTC and CfC), wiring, sequence unrolling, parameter aggregation, six Save/Load functions | its own wire format — persistence is the separate `serialize` package; built-in optimizers live in the separate `optimizer` package |
| `optimizer` | SGD/Momentum/Adam/AdEMAMix/ScheduleFreeAdamW as explicit structs over `autograd` leaves | state beyond per-parameter velocity/moments/the `z` sequence; lr schedules (caller-owned fields; ScheduleFreeAdamW replaces them by design) |
| `serialize` | Versioned `"LNNS"` tensor streams and model streams; error-only load path, safe against hostile streams | version negotiation (reads v1 and v2, writes only v2; unknown versions are rejected); compression, encryption |

## Data flow of one training iteration

```
ZeroGrad(all params)            // clear leaf Grad buffers
        │
forward: ops execute eagerly    // each op allocates a new tensor and
        │                       // tags its output with an op kind
        ▼
   loss (scalar leaf of the graph)
        │
loss.Backward()                 // topo-sort the graph, dispatch each
        │                       // node's backward in reverse; leaf grads
        │                       // accumulate
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
  every backward step can read its node's forward tensors without
  lifetime bookkeeping.
- **Immutability of constructed objects is real.** `Wiring` masks are
  exposed only through copying accessors, and the red-team audit confirmed
  that tampering with a returned mask cannot affect the cell (a pointer
  accessor would not have held that property).
- **The cost is explicit and predictable:** O(elements) copies per op and
  memory that grows with the graph (below). The library targets small,
  auditable models on CPU where clarity beats throughput; a strided
  view layer would be a different library.

MatMul does skip zero multipliers in its inner loop, which is the one
place where observable behavior differs from naive math — and the skip
tests only the **left** operand, so it is directional: `0 * NaN`
contributes `0` (the zero multiplier is skipped), while `NaN * 0` still
yields `NaN` (see [pitfalls.md](pitfalls.md)).

## The computation graph

Each `autograd.Variable` is a node:

```go
type Variable struct {
    Data *tensor.Tensor   // forward value
    Grad *tensor.Tensor   // accumulated gradient (nil until first backward)

    p        [2]*Variable // inline parent slots (every unary/binary op node)
    parents  []*Variable  // overflow parents (ConcatCol with more inputs)
    kind     opKind       // tag: which gradient formula runBackward dispatches
    scalar   float32      // payload: Scale factor / Pow exponent
    from, to int          // payload: SliceCol range / SliceRow index
    aux      *tensor.Tensor // payload: Div's inverse denominator / SigmoidHadamard's sigmoid buffer
    idx      []int        // payload: GatherRows indices (copied at construction)
    fused    func(*Variable) // payload: FusedOp's hand-written backward step
}
```

- **Leaves** (`Var`, `New`, `Const`) have no parents and the zero kind
  (`opLeaf`) — no backward step. They are parameters and inputs;
  gradients end their walk here.
- **Ops** run the forward computation eagerly via `tensor`, then tag the
  output node with an op kind plus its payload — **no per-node closure**.
  Backward steps used to be closures, and a closure capture allocated a
  heap object per graph node: one of the largest allocation sources in
  deep unrolled graphs. The tag dispatch of the phase-7 overhaul replaces
  that with a `uint8` kind and a few payload fields on the struct;
  `runBackward` is a 24-case switch (one case per `opKind`, including the
  no-op `opLeaf` and the `opFused` escape hatch below) whose case bodies
  are exactly the gradient formulas the
  closures used to carry. A few cases (e.g.
  `opLog`) read a *parent's* `Data` at backward time, so leaf data must
  not be mutated between forward and backward ([pitfalls.md](pitfalls.md)
  has the details).
- **Backward** builds the topological order with a post-order DFS, runs
  `runBackward` on each node in reverse, and then clears `Grad` on every
  non-leaf node except the receiver.

### Gradient buffers are owned, not cloned

`addGrad` panics on a shape mismatch (as before) — and on the *first*
contribution it takes ownership of the incoming buffer without cloning
it (所有权移交, ownership transfer). Every backward case passes either a
freshly allocated tensor or a buffer it dedicates to that call, so no
other reference can observe later accumulation into it. The `Add` case
shows the design at its sharpest: the a-branch hands `v.Grad` itself to
`a` via the taking reducer `sumToShapeTake` (safe because reverse
topological order guarantees every consumer of `v` has finished before
this step runs), while the b-branch deliberately goes through the
*cloning* `SumToShape` — when both operands share `v`'s shape it must
hand `b` a distinct buffer, or `a.Grad` and `b.Grad` would alias and
later accumulation into either would corrupt the other (think `Add(x, y)`
with both leaves reused downstream). The root seed gets the same
treatment centrally: `Backward` propagates from a private clone so an
auto-seeded ones buffer can be handed to a leaf without aliasing the
receiver's `Grad`, which is what keeps repeated `Backward` calls exactly
linear and manually seeded gradients pristine. The per-case ownership
audit lives in the comments of `autograd/ops.go`, and the alias design is
pinned by dedicated probes (`autograd/alias_probe_test.go`, plus the
shape-contract regressions in `autograd/f1_regress_test.go`) — the
Clone share of backward allocations fell from ~20% to ~1%.

`sumToShapeTake` itself is autograd-internal since v0.4.0: it used to be
the exported `tensor.SumToShapeTake`, but its only callers were the five
backward sites in `autograd/ops.go`, and an ownership footgun on the
public tensor surface bought nothing external — so it moved in and was
unexported (semantics, ownership contract and panic text preserved
verbatim). The cloning, alias-free `SumToShape` remains the sole public
reducer.

### Fused backward passes — and the FMA barrier

The same overhaul fused multi-node gradient chains into single loops:
the unary activations (Sigmoid, Tanh 4 graph nodes → 1; Log, Pow 3 → 1;
…), MatMul's backward via the new transpose-aware `MatMulTransA`/`MatMulTransB`
kernels (the identical products and accumulation order as MatMul over an
explicit `Transpose`, minus the two transpose buffers), and the
product-or-reduced-product fusion of `hadamardReduce` for Hadamard and
Div operands.

The engineering decision worth knowing about is the **FMA barrier**. A
naive fused multiply-accumulate loop — `r[j] += g[j] * x[j]` — is
compiled by the arm64 SSA backend into `FMADD` instructions: one
rounding, where the historical two-step path rounded twice (product
stored to a tensor element, then accumulated). The drift is ~1 ULP per
element — invisible per-op, catastrophic as a bitwise-equivalence claim
over 100k-node graphs. The barrier is `mul32`:

```go
func mul32(a, b float32) float32 {
    if math.Float32bits(a)&0x7F800000 == 0x7F800000 ||
        math.Float32bits(b)&0x7F800000 == 0x7F800000 {
        return a * b
    }
    return float32(float64(a) * float64(b))
}
```

For finite operands the exact product fits a 48-bit mantissa, which
`float64` represents precisely, so rounding it back to `float32` is
bit-identical to the hardware multiply — and the conversion leaves an
explicit rounding node in the SSA graph that FMA formation rules cannot
cross. Non-finite operands (NaN or ±Inf) take the **native** `a * b`
path instead: the float64 round-trip would recanonicalize NaN payloads
and diverge from the hardware float32 propagation the legacy chain ran.
The native product is a lone multiply with no adjacent add/sub in its
expression, and the branch leaves an If/Phi structure the FMA rules do
not match, so fusion cannot reach it either — verified with `go tool
compile -S`: zero `FMADD` family instructions across the package, while
a bare-loop control probe still emits them (the barrier is load-bearing,
not cargo-culted). A related subtlety: the negation fuses multiply by
`negOne`, a package-level *variable* holding −1, because the compiler
folds a multiply by the constant −1 into a sign flip (`FNEGS`), which
flips a NaN's sign bit while the legacy hardware multiply did not.

Fidelity wins over fixing, everywhere: the historical 1D→`[1, n]`
leaf-gradient shape quirk is replicated faithfully
(`elemwiseGradShape`), and irregular manually seeded gradient shapes
fall back to the literal legacy composition (`gradMatchesElemwise`
guards) — panic contract included. The whole overhaul was gated on
differential fuzzing against the pre-rewrite implementation extracted
from git history: 52,000 graphs, four classes of difference (finite
values, shapes, panic presence, NaN bits) all zero, with negative
controls confirming the gate actually detects seeded mutations. Measured
on this machine (`-benchtime=100x`): `UnrollBackward` 68,688 → **33,963
allocs/op (−50.55%)**, bytes −50.1%, time −24% — and the four other
benchmarks all down with zero regressions: `ChainForwardBackward`
−57.7%, `DivDenLoop` −56.7%, `LTCStep` −29.0%, `GatherRowsBackward`
−23.5%.

### Sigmoid–Hadamard fusion: the phase-8 capstone

The phase-8 `autograd.SigmoidHadamard(z, w)` (`autograd/ops.go`) fuses
the LTC hot-path pattern `Hadamard(Sigmoid(z), w)` — two graph nodes —
into one, and is adopted at the single shared sensory/recurrent entry of
`synapsesRows` (`nn/ltc.go:423`). It is the capstone of the backward
overhaul because it is the one fusion that needed a new operator rather
than a reorganization of existing ones, and its equivalence story has
three distinct parts:

- **The forward is bitwise by construction.** It runs the very same two
  tensor operations the composition ran — `tensor.Sigmoid`, then
  `tensor.Hadamard` — so shapes, broadcasting and values are identical
  by definition, not by measurement. The sigmoid buffer is kept on the
  node's `aux` slot so the backward reuses it instead of recomputing.
- **The regular backward is bitwise by rounding-site alignment.** On the
  2D path (the LTC hot path) the backward propagates
  `dz = g⊙w⊙s⊙(1−s)` in one fused loop and `dw = g⊙s` through the same
  `hadamardReduce` the Hadamard backward used. This was *not* designed to
  be bitwise — the design book expected a tolerance gate — but it turned
  out to be: rounding the `g⊙w` product through `mul32` at exactly the
  spot the legacy Hadamard backward handed the sigmoid node reproduces
  the legacy intermediate's rounding, and the outer `mul32(gw, s⊙(1−s))`
  grouping reproduces opSigmoid's fused loop. The result is bit-identical
  to the legacy two-node chain on the regular path (pinned by tests),
  which is also why the examples' training trajectories did not drift by
  a single bit.
- **The fallback is a verbatim replay.** Non-2D operands or an irregular
  manually seeded gradient reproduce the legacy `opHadamard`+`opSigmoid`
  pair verbatim — both `hadamardReduce` branches, then opSigmoid's own
  dispatch — so the values, shapes, the 1D→`[1,n]` lift quirk and the
  panic contract are all preserved exactly.

Measured in the same A/B window (`-benchtime=100x`): `LTCStep`
2,442 → **2,306 allocs/op (−5.6%)** and `UnrollBackward`
33,963 → **31,983 (−5.8%)**. The gain is deliberately reported as the
single-digit number it is: each fused site saves exactly one graph node
plus one backward intermediate tensor, and the structure accounts close
exactly (`LTCStep` 68 sites × 2 allocations = 136 = the difference;
`UnrollBackward` 396 sites × 5 = 1,980). What remains is `tensor.New`'s
fixed per-node overhead (addressed in the next section) — this is the
measured structural ceiling for fusion on this engine, the answer the
earlier "cost/fragility" question was waiting for.

### Embedded shape backing — the phase-10 allocation cut

The fixed per-node overhead tracked since phase 6 as roadmap item #12:
every graph node pays for its forward output tensor, and every tensor
construction paid twice — once for the `Tensor` with its `Data` buffer,
once more for the `Shape` `[]int` slice on the heap. Profiling put the
share at 64.9% of the remaining allocations (re-measured 60.4% just
before the fix). v0.4.0 removed the `Shape` half by embedding a `[4]int`
shape buffer (embedded backing — an inline `shapeBuf` in the struct):
`Shape` stays the `[]int` single source of truth, but for rank ≤ 4 it
points into the struct itself, at zero heap allocation; ranks beyond 4 —
only `serialize`'s wire stream carries tensors up to rank 8 — fall back
to a copied heap slice, so compatibility is preserved.

The decision trail is worth keeping, because the honest numbers argue
against *and* for the change at once. Prototyping measured the five
benchmarks at **−18~−26% allocs/op** (deterministic; the eight
tensor-operator benchmarks each lost exactly one shape allocation, −25%
where the count went 4→3), but wall-clock landed inside **±a few % of
noise** in interleaved A/B runs and `B/op` rose ~3% (the embedded buffer
enlarges every `Tensor`): the benefit is allocation count and GC
hygiene, not throughput. Two implementations were on the table: option
①, a value-type shape field, would remove the same allocation but break
all **233** `.Shape` accesses plus **7** direct writes — an API rupture
for the same gain; option ②, the embedded backing, touches ~10 internal
sites and breaks nothing (read paths untouched, exported field type
unchanged). The library owner chose ②; the sole API addition is the
exported `Tensor.Reshape`, the sanctioned replacement for direct
`t.Shape = …` writes (negative dimensions panic). The same v0.4.0
API-hygiene pass deleted `tensor.Stack` (3D output no op consumes, zero
in-library callers) and moved the ownership-contract `SumToShapeTake`
into autograd internals (above).

### The fused LTC unfold kernel — one node per Step (stage 16)

`autograd.FusedOp(data, parents, backward)` is the engine's escape hatch
from the fixed op set: the caller computes the forward value, and the
closure owns the node's **entire** backward — every `addGrad` to the
parents and the accumulation order they fire in are the closure's
contract, bit for bit. It costs one heap closure per node, negligible at
the intended rate of one node per RNN step.

The LTC's `Step` is where that rate pays: its `unfolds` ODE iterations
used to record ~`6·units` graph nodes per unfold, and stage-15 profiling
showed ~80% of the step's wall clock was per-node interpreter overhead,
not math. Stage 16 replaces the whole unfolds loop with **one** `FusedOp`
node (`nn/ltc_fused.go`): the forward replays the identical float32
operation sequence in a single loop nest (the same per-element kernels,
the same `mul32` FMA barriers wherever a product feeds an accumulation),
and the backward is a hand-written VJP that replays the replaced
subgraph's backward sweep contribution for contribution. The contract is
bit-identity against the pre-fusion graph path, not a tolerance — finite
values and NaN positions match exactly (the one archived exception, the
payload/sign of a NaN produced by *two* NaN operands, is in
[pitfalls.md](pitfalls.md)); it is gated by a differential matrix over
cell shapes, loss shapes and adversarial non-finite inputs, and the
public API is unchanged. One subtlety worth knowing exists, no more:
the fused node is paired with a second, parentless `FusedOp` (the `hvN`
delivery node) that emits one dense state-gradient contribution at
exactly the topological slot the graph path's `Hadamard(cmT, v)` node
completed in — that is how contributions arriving from *outside* the
kernel (the output affine, chained inputs, loss-side terms) interleave
with the kernel's deliveries bit-identically.

Measured (`-benchtime=200x`, this machine): `LTCStep` ~203 µs / 2,122
allocs → **~87 µs / 236 allocs/op**, `UnrollBackward` ~2.8–3.3 ms /
28,273 allocs → **~1.33 ms / 6,662** — about 2.1–2.3× wall clock, with
allocation counts down ~89%/76%. The remat benchmarks shrink along
(`UnrollRemat` ~3.3–3.6 ms, `UnrollRematCfC` ~1.9–2.2 ms), because
their recompute sweeps run the same fused steps.

Stage 19 moved the boundary outward twice. **19a internalized the
sensory drive** and the `numConst`/`denBase` assemblies (~`10·inDim+2`
further nodes per step, the largest remaining interpreter share): the
parent list grows 9 → 12 (`inputs`, `vleak`, `sMu`, `sSigma`, `sWm`
enter the kernel), the `hvN` delivery slot is *re-derived* rather than
assumed — everything that used to append during `numConst`'s subtree
expansion now appends during the `inputs` parent's expansion, at the
same post-`hvN` position, so the three documented interleavings carry
over mechanically — and a Step now records **34 nodes at any
dimensions** (15 op nodes + 19 leaves; 84 before, 626 pre-fusion;
pinned by `TestLTCFusedNodeAccount19a`). **19b allocates the VJP's
scratch planes once per backward** and reuses them across unfolds and
presynaptic rows (18 planes), with the delivery/reuse boundary stated
explicitly: reused scratch is never handed to `addGrad`, while every
delivered gradient buffer stays freshly allocated — `addGrad`'s
first-contribution ownership transfer is part of the bitwise contract
(the latest unfold's accumulator row is copy-adopted, −0 signs
included). The remat fold classes are untouched (pinned by
`TestRematFusedLTCFoldClasses`). Measured (`-benchtime=200x`):
`LTCStep` ~87 µs / 236 → **~78 µs / 77 allocs / 23 KB**,
`UnrollBackward` ~1.33 ms / 6,662 → **~1.29 ms / 3,750 / 558 KB**,
`UnrollRemat` ~3.45 ms / 16,000 → **~3.4 ms / 9,373**,
`UnrollRematCfC` ~0.83 ms / 5,000 → **~0.84 ms / 4,829** (the CfC was
not touched this round — its numbers stand from stage 18).

### The fused CfC step — the whole step as one node (stage 18)

Stage 18 applies the same recipe to the CfC (`nn/cfc_fused.go`), with
three structural differences worth knowing. First, the closed form has
no unfolds, so the kernel fuses the *entire* per-step subgraph — both
drives, the `g`/`a` assembly, the `decayRate` cap chain and the
exprel `decayFactor` — not a sub-loop: `66 + 14·(inDim+units)` graph
nodes (190 at `inDim=1, units=8`, 346 at `4, 16`) become **24 nodes at
any dimensions** (9 op nodes + 15 leaves), with only the input affine,
the four softplus constraints and the output affine staying
graph-level. Second, there is **no `hvN` delivery node**: the LTC's
dense `cmT` contribution sat unfolds-deep where external contributions
could interleave, but the CfC's three state-gradient contribution
classes are mutually contiguous under either record-high association,
so one atomic VJP suffices. Third, the exprel branch is value
selection rather than control flow — both branches are evaluated and
mask-added in the graph's order, so there is no branch divergence to
replay. The bitwise contract (forward and gradients, chained/stacked
topologies included) and the `mul32` discipline are the LTC kernel's;
`h` sits first in the parent list, which keeps the CfC spine-free and
`UnrollRemat` at its two-forwards-one-backward ideal. Measured
(`-benchtime=200x`): `CfCStep` ~83.5 µs / 1,066 allocs → **~34 µs /
52**, `UnrollRematCfC` ~1.86 ms / 22,106 → **~0.83 ms / 5,000**. Two
deliberate behavioral tightenings ride along — upfront state-shape
validation (broadcast-compatible wrong `h` shapes the graph path
silently broadcast now panic, as the `Cell` contract always required)
and a pre-delivery panic point (a caught panic leaves the fused kernel
clean; discard the graph after a `recover` regardless) — both
documented in [cfc.md](cfc.md).

### Rematerialized BPTT — `nn.UnrollRemat` (stage 16)

The memory model above makes BPTT cost O(T × per-step graph): the whole
sequence stays in the graph until `Backward`. `nn.UnrollRemat` trades
recomputation for that memory — gradient checkpointing — while keeping
the gradients **bit-for-bit identical** to `Unroll` + `loss.Backward()`.
Two passes: the first walks the sequence step by step,
[`Detach`](api.md)ing every step's output and input state so the live
graph never exceeds one step, then builds the loss once over the
detached outputs and backwards it — yielding exactly the whole-graph
backward's per-step output seeds. The second recomputes chunk-sized
units of steps in reverse and backpropagates the saved seeds through
them, threading the state gradient through the detached boundary states.

The hard part is *order*, not math: `float32` addition is not
associative, and the whole-graph backward folds each leaf's
contributions in an order fixed by the graph's DFS shape and by which
step outputs the loss visits first. Conceptually there are three fold
classes — the per-step state subgraph folds in strictly reverse step
order; the output affine (e.g. `outW`/`outB`) folds in the loss's
reverse visit order over the seeded steps; and the LTC's `cm` "spine"
folds run by run of the visit order's record highs. A one-step
structural probe sorts every swept leaf into its class, and the sweeps
(rest, plus a σ sweep for spine-class cells like the LTC, plus an
affine pass for non-ascending losses) replay each class's fold — the
association `((A+r₁)+r₂)` vs `(A+(r₁+r₂))` differs by one rounding, so
replaying the *sequence* is what makes the result bitwise rather than
merely close. The full argument and the sweep structure are in
`nn/remat.go`'s godoc; this paragraph is the map, not the territory.

Memory semantics: peak graph memory drops to O(chunkSize × per-step
graph) plus O(T) small detached tensors (one `[batch, StateSize()]`
tensor per step) — measured at T = 512: **~0.65 MB retained vs ~8.3 MB
live** for the full unroll (~13×). The worst case is honest: a loss
visiting the outputs in an adversarial order (a descending visit with a
small chunkSize is the extreme) forces recompute units to merge up to
O(T) long, so peak memory returns to the full-unroll figure — paid on
top of the sweeps' recomputation. In that corner `UnrollRemat` costs
strictly more peak memory AND more compute than one whole-graph
backward; that is the price of bitwise fidelity, not a bug. The caller
contracts, all probe-enforced with panics: `params` must list **every**
trainable leaf the cell's `Step` consumes (an unlisted leaf would
silently accumulate a 2–3× wrong gradient across sweeps); the cell's
per-step graph structure must be a pure function of `(x, h)` — a
value-dependent branch is the one case the probe cannot see and drifts
~1–2 ULP, archived in [pitfalls.md](pitfalls.md); and a loss closing
over a step-consumed leaf must visit every loss-side consumer of it
after the seeded outputs (spell penalties data-first:
`Add(data, penalty)`). A classification cache was considered and
deliberately vetoed — no sound cache key exists, and a stale hit would
silently skip a multi-class panic (the two loss-graph walks do share
one `TopoOrder` computation). A runnable recipe with chunk-size guidance is
[cookbook.md](cookbook.md#13-long-sequence-training-chunked-bptt-with-unrollremat).

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

### Stage-19 hygiene: pooled traversal scratch and fixed-array broadcast shapes

Two internal allocation cuts landed alongside 19a/19b, both invisible to
behavior (zero API or numeric change, gated by the differential suites):

- **`Backward`'s DFS scratch is pooled.** The topological traversal's
  visited set and topo buffer used to be allocated per call — and remat
  runs several traversals per recompute unit. One shared pair is now
  pooled between `Backward` and `TopoOrder`, so a plain `Backward`
  allocates no traversal scratch at all (`UnrollBackward` B/op −5.1%
  from this alone). The safety contract is stated in the code: a
  *nested* `Backward`/`TopoOrder` during an outer traversal observes the
  in-use flag and falls back to fresh scratch (the pre-pool behavior);
  the release runs deferred, so a panicked traversal returns the pool
  pristine; no graph node is kept alive by the scratch — only capacity
  is retained. And `TopoOrder` never hands pooled storage to its
  caller: its result slice is always freshly allocated.
- **Broadcast shapes for the two fresh-shape arms are fixed arrays.**
  `broadcastShapeFresh`'s outer-product and 1D-lift arms used to
  allocate their result shape as a slice; they now return a value-type
  `[2]int` that never reaches the heap — 3 → 2 allocs/op on those arms,
  with exact floor assertions in `tensor/broadcast_shape_test.go`. The
  shapes were already fresh copies on those arms, so nothing observable
  moved.

### The graph is the memory model

Every intermediate tensor stays alive — referenced by its node — until
`Backward` completes. Memory therefore scales with the number of ops
per iteration, not just with parameter size. One LTC `Step` unrolls
`unfolds` ODE iterations into the graph; since the phase-6 synapse
vectorization each iteration is O(units) activation blocks, and since
the phase-9 sparse contraction the presynaptic reduction is a
`+0`-seeded fold ended by normalizing MatMuls ([ltc.md](ltc.md)) —
down from O(units²) per-synapse nodes, and from the dense
`[units², units]` indicator matrices phase 9 eliminated (those
materialized O(units³) float32s, at construction *and* at load time;
see [persistence.md](persistence.md)). The phase-7 backward overhaul
(above) halved the per-node allocation count, and the phase-8
Sigmoid–Hadamard fusion took it further still, and the phase-10
embedded shape backing (above) removed one more allocation per tensor
construction. Measured now (stage 19, `-benchtime=200x`): `LTCStep`
**77 allocs/op** and `UnrollBackward` **3,750** — cumulative **−99% and
−97%** from the original per-synapse loop (7,360 / 120,163); the
stage-16 fused unfold kernel was the big step (~2.1–2.3× wall clock
against the pre-fusion graph path), stage 19 moved the sensory path
into the kernel and cut allocations a further 67%/44%. The phase-9 step is an honest trade: allocs went
*up* ~43%/~30% over the phase-8 values (2,306 / 31,983; the fold's
per-stage cloning) but ns/op went *down* ~21%/~13% (independent
red-team rerun), because the dense indicator MatMuls' O(units³) idle
inner loops over zero rows and their large backward allocations are
gone — allocation counts were traded away for useless compute, and
wall-clock is a net benefit. The same change removed the
*construction-time* memory cliff: a fully-wired `units = 1024` cell
now allocates ~32 MB (measured: 36.4 MiB for `NewLTC`, 32.4 MiB for
`NewCfC`; red-team re-verification agrees), not the old ~8 GiB of
indicators. Graph memory still grows with `units · unfolds · sequence
length`, so keep all three modest on this engine; a `CfC` step
([cfc.md](cfc.md)) avoids the `unfolds` factor entirely — one fused
node per step since stage 18 (52 allocs/op) — and
`nn.UnrollRemat` (above) caps peak graph memory at O(chunk) for long
sequences.

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
