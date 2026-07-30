# The LTC cell: paper ↔ code

> English | [中文](zh/ltc.md)

**Summary:** a line-by-line correspondence between the Liquid Time-Constant
equations (Hasani et al., AAAI 2021) and `nn/ltc.go`, including the
semi-implicit Euler algebra, the full parameter table, the `ts` contract and
wiring semantics.

**Audience:** engineers who need to trust (or modify) the cell's numerics;
readers of the paper who want to find each symbol in the code.

References: R. Hasani, M. Lechner, A. Amini, D. Rus, R. Grosu,
[*Liquid Time-constant Networks*](https://ojs.aaai.org/index.php/AAAI/article/view/17017)
(AAAI 2021), and the reference implementation
[`mlech26l/ncps`](https://github.com/mlech26l/ncps).

## The ODE

Each neuron is a leaky membrane with an input-dependent time constant:

```
cm · dv/dt = −gleak · (v − vleak) + Σⱼ actⱼ · (erevⱼ − v)

    actⱼ = wⱼ · sigmoid( σⱼ · (v_pre,ⱼ − μⱼ) )
```

- `v` — membrane potential (the hidden state), one per unit.
- `cm` — membrane capacitance; `gleak`/`vleak` — leak conductance and its
  reversal potential.
- Synapse `j` from presynaptic source `v_pre` activates with a *sigmoidal
  activation function* centered at `μⱼ` with steepness `σⱼ`, scaled by
  weight `wⱼ`; `(erevⱼ − v)` is the driving force toward the synapse's
  reversal potential (excitatory for `erev = +1`, inhibitory for `−1`).

The `LTC` type doc comment (`nn/ltc.go:15-29`) states the same model in the
paper's alternative form `dv/dt = −(1/tau + f(v, I))·v + f(v, I)·A`.

## Paper symbol → code line map (nn/ltc.go)

| Paper concept | Code | Lines |
|---|---|---|
| ODE statement + integration scheme (doc) | `LTC` type comment | 15–29 |
| Constructor, validation | `NewLTC` | 75–92 |
| Parameter init ranges | `NewLTC` literal | 108–134 |
| `eps` (denominator guard) = `1e-8` | `ltcEps` const; `eps` field init | 12–13; 109 |
| Construction-time graph constants (folded masks, sparse reduction indicators) | `NewLTC` literal; `sumIndicator`; `reversalIndicator` | 128–137; 160–168; 174–182 |
| Trainable parameter set (13 tensors) | `Parameters()` | 189–196 |
| One RNN step | `Step` | 202–251 |
| `ts` contract check (positive & finite) | `Step` guard | 207–209 |
| Affine input map `inputs = x⊙inW + inB` | `Step` | 216 |
| Softplus constraints on `cm, w, sW` (+`gleak`); mask folded into the weights once per Step | `Step` | 221–224 |
| Sensory synaptic currents (once per step) | `synapses(inputs, sMu, sSigma, sWm, denReduceS, numReduceS)` | 227 |
| Step-invariant recurrent parameter rows, sliced once | `rows(c.mu/c.sigma/wM)` (helper `rows` at 287–293) | 231–233 |
| Step-constant membrane terms (`eps` hoisted into `denBase`) | `numConst`, `denBase` | 238–239 |
| ODE unfold loop (`unfolds` substeps) | `for t := 0; t < c.unfolds; t++` | 242–247 |
| Recurrent currents (recomputed per substep) | `synapsesRows(v, muRs, sigRs, wmRs, denReduceR, numReduceR)` | 243 |
| `num = cm_t⊙v + gleak⊙vleak + Σ actⱼ·erevⱼ` | `num := …` | 245 |
| `den = cm_t + gleak + Σ actⱼ (+ eps)` | `denBase` + `denR` | 239, 246 |
| `v ← num / (den + eps)` | `v = autograd.Div(num, Add(denBase, denR))` | 246 |
| Affine output map `out = v⊙outW + outB` | `Step` | 249 |
| `cm_t = softplus(cm)·unfolds/ts` with overflow-safe scaling | `scaledCapacitance` | 267–280 |
| Per-presynaptic activation block + two indicator MatMul contractions | `synapses` / `synapsesRows` | 318–331 / 336–358 |

### Synaptic drive vectorization

`synapses`/`synapsesRows` (`nn/ltc.go:318-358`) compute the currents per
presynaptic neuron as one vector block each, instead of a per-synapse-pair
loop. Row `i` of each parameter matrix still parameterizes the synapses
*from* neuron `i`:

```go
// One [batch, units] activation block per presynaptic neuron i
// (synapsesRows, nn/ltc.go:341-346):
preCol := Col(pre, i)                              // [batch, 1]
z := Hadamard(sigRs[i], Sub(preCol, muRs[i]))      // σᵢⱼ·(v_pre,i − μᵢⱼ)
blocks[i] = Hadamard(Sigmoid(z), wmRs[i])          // × wᵢⱼ·maskᵢⱼ

// The blocks concatenate to [batch, pre·units]; two MatMuls against
// sparse construction-time indicators contract the presynaptic axis
// (nn/ltc.go:347-357):
den = MatMul(flat, denReduce)                      // den[:,j] = Σᵢ blocksᵢ[:,j]
num = MatMul(flat, numReduce)                      // num[:,j] = Σᵢ blocksᵢ[:,j]·erev[i,j]
```

Three structural changes relative to the original per-synapse loop:

- **Mask out of the hot path.** The wiring masks fold into the
  positivity-constrained weights with one matrix Hadamard per Step —
  `wm = softplus(w)⊙mask` (`nn/ltc.go:223-224`) — instead of one masked
  multiply per synapse per substep.
- **Indicator-matrix contractions.** `denReduce`/`numReduce` are sparse
  `[pre·units, units]` indicator matrices (指示矩阵) built once at
  construction (`sumIndicator`, `nn/ltc.go:160-168`); `numReduce`
  additionally bakes the constant ±1 reversal potentials into its
  nonzeros (`reversalIndicator`, `nn/ltc.go:174-182`), so `erev` never
  appears as a per-synapse graph node. MatMul skips zero entries, so
  each contraction costs O(batch·pre·units) despite the indicator's
  `[pre·units, units]` extent.
- **Recurrent rows sliced once per Step.** `mu`, `sigma` and the masked
  weight matrix are Step-invariant; `rows` (`nn/ltc.go:287-293`) slices
  them once and the unfold loop reuses the rows, keeping
  `units·(unfolds−1)` `SliceRow` nodes per matrix out of the graph.

**Equivalence is ULP-level at the Step level — not bitwise.** In
isolation, the vectorized drive is bit-for-bit identical to the old
per-synapse loop (regression-tested with strict `==` by
`TestLTCSynapsesVectorizedEquivalence` in `nn/ltc_test.go`: mask ∈ {0,1}
and ascending contraction order reproduce the old `Add` chain exactly,
with no rounding-order change). The whole `Step`, however, hoists `eps`
and the Step-constant terms (`gleak⊙vleak`, the sensory currents) out
of the unfold loop, which changes `float32` association order. An
independent red-team oracle — the pre-rewrite `ltc.go` extracted from
git history, differential-tested on 13 randomized configurations —
measures a maximum difference of **1.79e-7** forward and **1.19e-7** on
all-parameter BPTT gradients: ULP-level, benign, but not "bitwise".

Measured cost reduction (same machine, `-benchtime=100x`, re-verified):
`LTCStep` 7,360 → **3,440 allocs/op (−53.3%)**; `UnrollBackward`
120,163 → **68,688 allocs/op (−42.8%)**. Graph nodes per unfold drop
from O(units²) per-synapse nodes to O(units) vector blocks plus the two
contractions; the example's first-iteration loss remains `0.690761`,
bit-identical to the pre-rewrite value.

## The semi-implicit Euler, derived

The reference integrates the ODE with a semi-implicit (backward) Euler
step: the leak and synaptic driving forces are evaluated at the *new*
voltage `v_{k+1}`, which makes the update an exact division instead of an
exponential. With substep length `dt = ts/unfolds`:

```
cm · (v_{k+1} − v_k)/dt = −gleak·(v_{k+1} − vleak) + Σⱼ actⱼ·(erevⱼ − v_{k+1})
```

Collect `v_{k+1}` on the left:

```
(cm/dt)·v_{k+1} + gleak·v_{k+1} + (Σⱼ actⱼ)·v_{k+1}
        = (cm/dt)·v_k + gleak·vleak + Σⱼ actⱼ·erevⱼ
```

```
            (cm/dt)·v_k + gleak·vleak + Σⱼ actⱼ·erevⱼ      num
v_{k+1}  =  ───────────────────────────────────────────  = ───────
             cm/dt + gleak + Σⱼ actⱼ                        den
```

and the code computes `v ← num / (den + eps)` elementwise
(`nn/ltc.go:245-246`), with `cm/dt = softplus(cm)·unfolds/ts` built by
`scaledCapacitance`. All quantities are per-unit vectors, so every
operation is elementwise or a broadcast; the degenerate case (all wiring
masks zero) reduces exactly to a leaky integrator
`v ← (a·v + b·vleak)/(a + b + eps)` with `a = softplus(cm)·unfolds/ts`,
`b = softplus(gleak)` — regression-tested in closed form by
`TestLTCZeroMasksLeakyIntegrator` in `nn/ltc_test.go`.

Because the solve is implicit in `v`, the update is stable for large `ts`
(the state relaxes toward its steady state instead of exploding), which is
what makes the cell's variable-step regime usable.

## Parameter table

All ranges follow the ncps reference implementation. "Softplus" means the
raw parameter is unconstrained and enters the ODE as `softplus(raw)`, the
reference's `implicit_param_constraints` mode — positivity without
optimizer-side clipping.

| Parameter | Shape | Init | Constraint | Role |
|---|---|---|---|---|
| `gleak` | `[units]` | U(0.001, 1) | softplus | leak conductance |
| `vleak` | `[units]` | U(−0.2, 0.2) | unconstrained | leak reversal potential |
| `cm` | `[units]` | U(0.4, 0.6) | softplus, scaled by `unfolds/ts` | membrane capacitance |
| `mu` | `[units, units]` | U(0.3, 0.8) | unconstrained | recurrent synapse activation centers |
| `sigma` | `[units, units]` | U(3, 8) | unconstrained | recurrent synapse steepness |
| `w` | `[units, units]` | U(0.001, 1) | softplus | recurrent synaptic weights |
| `sMu` | `[inDim, units]` | U(0.3, 0.8) | unconstrained | sensory synapse centers |
| `sSigma` | `[inDim, units]` | U(3, 8) | unconstrained | sensory synapse steepness |
| `sW` | `[inDim, units]` | U(0.001, 1) | softplus | sensory synaptic weights |
| `inW`, `inB` | `[inDim]` | 1, 0 | unconstrained | per-feature affine input map |
| `outW`, `outB` | `[units]` | 1, 0 | unconstrained | per-unit affine state→output map |
| `erev` | `[units, units]` | random ±1 | **fixed — not trainable** | recurrent reversal potentials |
| `sErev` | `[inDim, units]` | random ±1 | **fixed — not trainable** | sensory reversal potentials |

`Parameters()` returns the 13 trainable tensors; `erev`/`sErev` are
deliberately absent (`nn/ltc.go:189-196`). Learning the reversal potentials
would let synapses flip between excitatory and inhibitory polarity,
degrading the LTC into an ordinary plastic network — the ±1 sign pattern
is structural, drawn once at construction.

Known deviation from the reference: `inW`/`outW` initialize to exactly 1
and `inB`/`outB` to 0, where ncps uses U(0.9, 1.1)/U(−0.1, 0.1). Both maps
are trainable, so the difference washes out during training; it only
affects step zero.

## The time span `ts`

`Step(x, h, ts)` integrates the ODE over a time span of `ts` units, in
`unfolds` substeps (`dt = ts/unfolds`). This is the "liquid" part of the
network: the caller drives time, event-driven and variable per step —
small `ts` barely advances the membranes, large `ts` relaxes them toward
steady state. It corresponds to ncps's `elapsed_time`.

**Contract: `ts` must be positive and finite.** `NaN`, `+Inf`, `-Inf`,
zero and negative values panic (`nn/ltc.go:207-209`):

```go
_, _ = cell.Step(x, nil, 0.01)  // fine: fast dynamics
_, _ = cell.Step(x, nil, 10.0)  // fine: near steady state
cell.Step(x, nil, math.Inf(1))  // panics: infinite time span is rejected
cell.Step(x, nil, math.NaN())   // panics
```

Finiteness domains of the implementation (`scaledCapacitance`,
`nn/ltc.go:267-280`):

| `ts` range | behavior |
|---|---|
| `ts ≳ 1e-3` (any real training regime) | the overflow guard is bit-identical to the naive `softplus(cm)·unfolds/ts` — a red-team sweep over `ts ∈ [1e-3, 100]` measured max diff `0`; full physical fidelity |
| down to ≈ `1e-38` | still the true ODE; the first bit-level deviations from the naive formula appear around `ts ≈ 1e-37 – 1e-38` (the hard overflow clamp engages once `unfolds/ts` exceeds `MaxFloat32`, i.e. `ts ≲ 1.8e-38`), far below any schedule |
| `0 < ts ≲ 1e-38` | `unfolds/ts` exceeds `MaxFloat32`; the scale is clamped and the capacitance product is capped by a smooth differentiable min. Outputs stay **finite** (verified at `ts = 1e-40` and `1e-300`), but this is a finiteness-only domain, not physical — no sane schedule goes here |
| huge `ts` (`1e300` tested) | scale → 0, state relaxes to its steady state; finite |

## Wiring

`Wiring` gates individual synapses with binary masks (`nn/wiring.go`):

- **sensory mask** `[inDim, units]` — synapses from inputs to neurons;
- **recurrent mask** `[units, units]` — synapses between neurons;
- entry `[i, j]` is the synapse **from** input/neuron `i` **to** neuron `j`
  (row `i` of the parameter matrices parameterizes presynaptic neuron `i`).

Constructors:

| Constructor | semantics |
|---|---|
| `FullyConnected(inDim, units)` | every synapse exists (all ones); also the default when `NewLTC` receives `nil` |
| `RandomSparse(inDim, units, sensoryP, recurrentP, rng)` | each synapse exists independently with probability `p` (Bernoulli per entry). No connectivity guarantee: at low `p`, neurons can end up isolated. Probabilities must lie in `[0, 1]` — `NaN`, negative and `>1` values panic, as do dimensions `< 1`. |

Masks are immutable after construction: fields are unexported, the
full-mask accessors `Sensory()`/`Recurrent()` return deep copies, and the
hot-path row accessors used inside `Step` read the pristine mask directly.
Verified by the red team: tampering with any returned tensor leaves cell
outputs bit-identical.

A sparse cell, driven at several time scales (seed 42, deterministic):

```go
rng := rand.New(rand.NewSource(42))
wiring := nn.RandomSparse(2, 6, 0.8, 0.5, rng) // sensory 11/12, recurrent 16/36 synapses
cell := nn.NewLTC(2, 6, wiring, 6, rng)        // 13 trainable parameter tensors

x := autograd.Var(tensor.Uniform(rng, -1, 1, 3, 2))
out, h := cell.Step(x, nil, 0.1) // batch 3, zero initial state
_, h = cell.Step(x, h, 1.0)      // thread the state; larger ts -> further relaxed
ys, hN := nn.Unroll(cell, []*autograd.Variable{x, x, x}, nil, 0.1)
```

## Relation to the ncps reference

| ncps concept | lnn counterpart |
|---|---|
| LTC layer with `units`, `unfolds` (default 6) | `NewLTC(inDim, units, wiring, unfolds, rng)` |
| ODE solver: semi-implicit Euler over `unfolds` substeps | identical scheme (`Step` loop, `nn/ltc.go:242-247`) |
| `implicit_param_constraints` (softplus positivity) | softplus on `cm`, `gleak`, `w`, `sW` |
| parameter init ranges | adopted verbatim (table above) |
| fixed ±1 reversal potentials from wiring | `erev`/`sErev`, not in `Parameters()` |
| `elapsed_time` per step (variable-step/event-driven training) | `ts float64` argument of `Step`/`Unroll` (float64 chosen to match ncps and to keep `unfolds/ts` safe for tiny `ts`) |
| sensory vs recurrent synapse split | `sMu/sSigma/sW/sErev` vs `mu/sigma/w/erev` + wiring masks |
| CfC (closed-form continuous-time) cell | **implemented** as `nn.CfC` (`nn/cfc.go`): the same ODE and the same 13-parameter synapse parameterization, advanced by the Lemma 1 closed-form solution instead of this Euler loop — see [cfc.md](cfc.md) |
