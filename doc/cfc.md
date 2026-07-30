# The CfC cell: closed-form continuous time

> English | [中文](zh/cfc.md)

**Summary:** the paper-to-code correspondence of `nn/cfc.go` — the
Closed-form Continuous-time cell (Hasani et al., *Nature Machine
Intelligence* 2022): the very membrane ODE this library's LTC integrates
numerically, advanced over a step's time span by its Lemma 1 closed-form
(closed-form solution, 闭式解) instead of an ODE solver loop — including
the exprel stabilization, the `ts` contract, the shared LTC
parameterization, and the deliberate deviation from the official
MLP-backbone cells.

**Audience:** readers of the paper who want each symbol in the code;
engineers choosing between `LTC` and `CfC`.

References: R. Hasani, M. Lechner, A. Amini, D. Rus, R. Grosu,
[*Closed-form continuous-time neural networks*](https://doi.org/10.1038/s42256-022-00556-7),
**Nature Machine Intelligence 4, 992–1003 (2022)**,
DOI [10.1038/s42256-022-00556-7](https://doi.org/10.1038/s42256-022-00556-7),
arXiv [2106.13898](https://arxiv.org/abs/2106.13898) — note this is
*Nature Machine Intelligence*, not *Nature Communications*. Reference
implementations: [`mlech26l/ncps`](https://github.com/mlech26l/ncps)
(`ncps/torch/cfc_cell.py`) and [`raminmh/CfC`](https://github.com/raminmh/CfC)
(`torch_cfc.py`).

**A naming warning:** the expansion "liquid cubic activation" for the CfC
acronym that appears in some search-engine summaries belongs to **no
official source**. A six-way grep across both paper versions (arXiv
preprint and NMI published version) and both official repositories
returns zero hits for "liquid cubic" — the phrase does not occur in the
paper, the code, or the authors' materials. It was not adopted here; do
not let summary generators mislead you.

## The ODE is exactly the LTC's

The CfC keeps the LTC membrane ODE — the one [ltc.md](ltc.md) derives —
written in the form that exposes the closed-form solution:

```
cm · dv/dt = −gleak·(v − vleak) + Σⱼ actⱼ·(erevⱼ − v)
           = −G·(v − A)

G = gleak + Σⱼ actⱼ                                   total conductance
A = (gleak·vleak + Σⱼ actⱼ·erevⱼ) / G                 instantaneous reversal state
actⱼ = softplus(wⱼ)·sigmoid(σⱼ·(v_pre − μⱼ))·maskⱼ    exactly the LTC activation
```

`G` and `A` aggregate exactly the currents the LTC's `den`/`num`
accumulate ([ltc.md](ltc.md)), per unit.

## Lemma 1: the closed-form solution

Freeze the activations over the step — the Lemma 1 approximation: the
input integral is evaluated at the current input, so `G` and `A` are
piecewise constant — and the ODE is linear in `v` with the exact
solution over a time span `ts` (the paper's Theorem 1 / Lemma 1 /
Eq. (8)):

```
v_new = A + (v − A)·e^{−κ·ts},   κ = G/cm
      = v + (A − v)·F(B),        B = κ·ts,   F(B) = 1 − e^{−B}
```

`F ∈ [0, 1]`, so `v_new` is a convex combination of the old state `v`
and the instantaneous reversal state `A`: the state stays bounded with
no solver unfolds at all. The code follows the second form
(`nn/cfc.go`, `Step`, lines 175–217):

| quantity | code | lines |
|---|---|---|
| `G = gleak + Σ actⱼ` | `g := autograd.Add(gleak, autograd.Add(denS, denR))` | 201 |
| `A = (gleak·vleak + Σ actⱼ·erevⱼ) / (G + eps)` | `a := autograd.Div(…, autograd.Add(g, epsV))` | 203–206 |
| `B = κ·ts`, overflow/sign-capped | `b := c.decayRate(g, cm, epsV, ts)` | 208 |
| `F(B) = 1 − e^{−B}`, exprel-stabilized | `f := c.decayFactor(b)` | 210 |
| `v_new = v + (A − v)·F` | `vNew := autograd.Add(h, autograd.Hadamard(autograd.Sub(a, h), f))` | 213 |

One honest code-level deviation from the paper's bare equations: the
divisors are guarded, `κ = G/(cm + eps)` and `A = …/(G + eps)` with
`eps = 1e-8`. Both divisors are bounded away from zero by softplus
positivity in any real training regime, so the guard is numerically
invisible; it exists to keep adversarial parameter draws from producing
`Inf`/`NaN` gradients.

## Algorithm 1: compiling the LTC to closed form

The paper's Algorithm 1 compiles an LTC into its closed-form update
synapse by synapse, allowing arbitrary sparse adjacency. lnn mirrors
that structure: `drive()` (`nn/cfc.go:226-249`) walks presynaptic
neuron `i`'s row of the parameter matrices, gates it with wiring mask
row `i`, and accumulates the `num`/`den` currents — the same convention
as the LTC's `synapses()` and the same binary wiring masks, so
`NewCfC(inDim, units, wiring, rng)` accepts exactly the `Wiring`
topologies `NewLTC` does (`nil` means fully connected).

## Relation to the LTC: same ODE, two integrators

`CfC` and `LTC` discretize the **same** ODE; the difference is the
integrator:

| | `LTC` | `CfC` |
|---|---|---|
| scheme | semi-implicit Euler over `unfolds` substeps ([ltc.md](ltc.md)) | analytic integrator (解析积分器): the Lemma 1 closed form in one step |
| graph cost per RNN step | grows with `unfolds` | constant in the time span — no substep loop |
| constructor | `NewLTC(inDim, units, wiring, unfolds, rng)` | `NewCfC(inDim, units, wiring, rng)` — no `unfolds` |
| everything else | shared: the 13 trainable tensors, init ranges, fixed ±1 reversal potentials excluded from `Parameters()`, the `ts` contract, the `Cell` interface — `nn.Unroll` drives both unchanged |

Because the parameter draw order is identical, **the same seed gives
bit-identical initialization** across the two cells (verified: all 13
parameters plus the ±1 reversal potentials compare equal).

Convergence (red-team, independent oracle): over a fixed time horizon
the difference between CfC and LTC trajectories converges with
first-order `p ≈ 1.0` — halving `ts` halves the gap, as expected of two
first-order discretizations of one ODE. Over a *single* step the
difference shrinks faster, `~O(ts²)` as `ts → 0` (measured in `/tmp`:
log-log slope 1.40 at `ts = 0.4` tightening to 1.95 at `ts = 0.025`,
matching the red team's `p ≈ 1.89`).

**Trade-off vs the official implementation.** The official CfC cells
(`ncps`, `raminmh/CfC`) ship an MLP-backbone variant: feed-forward
`ff1`/`ff2` heads behind a sigmoid time gate stand in for the
closed-form solution as a learned proxy. This library's convention is
synapses + wiring + fixed ±1 reversal potentials (see
[ltc.md](ltc.md)), so the CfC here implements the paper's
*equation-level* closed form — the red team's equation-by-equation
audit ruled it "closer to the equations than the official pure mode".
If you need to reproduce MLP-backbone outputs from those repositories
bit-for-bit, this is not that cell; if you want the ODE's closed-form
solution with this library's LTC parameterization, it is.

## The exprel stabilization

The famous closed-form-CT trap is the raw quotient `(1 − e^{−B})/B` at
`B → 0`: `1 − e^{−B}` cancels to 0 in finite precision and dividing by
`B` yields garbage (and a dead gradient). `decayFactor`
(`nn/cfc.go:299-324`) sidesteps it by computing the whole product
`F(B) = B·exprel(B)` with a per-element branch:

| branch | formula | why |
|---|---|---|
| `B < 1e-2` | Taylor: `B − B²/2 + B³/6 − B⁴/24` | dropped remainder `≤ B⁵/120 < 8.3e-13`, far below `float32` epsilon; `dF/dB = 1 − B + B²/2 − B³/6 → 1`, so the gradient survives `B → 0` |
| `B ≥ 1e-2` | direct `1 − e^{−B}` | no catastrophic cancellation left at this scale; for huge `B`, `e^{−B}` underflows to exactly 0 and `F` saturates at 1 (`v_new → A`) |

The division by `B` of the exprel quotient is cancelled **analytically**
against the outer `B` factor before anything enters the graph — there is
**no divide-by-`B` node** to guard at all. The branch mask is a
per-element constant built from `B`'s data, so gradients flow through
exactly the active branch, and the two branches agree to `~1e-10` in
value and slope at the threshold (red-team scan of 8001 points crossing
`1e-2`: jump `≤ 2.98e-8`; regression-tested by
`TestCfCExprelBoundaryContinuity`).

`B` itself is protected upstream in `decayRate` (`nn/cfc.go:262-275`):
the time scale is computed in `float64` and capped at `1e30` before
conversion, and the conductance ratio gets the same smooth
differentiable cap the LTC uses for its capacitance scaling
(`cap(k) = k − softplus(k − hi)`), which also keeps `B` non-negative —
a negative decay rate would turn `e^{−B}` into a blow-up.

## The time span `ts`

The contract is the LTC's: `ts` must be positive and finite; `NaN`,
`±Inf`, zero and negative values panic (`nn/cfc.go:178-180`). Behavior
at the extremes:

| `ts` | behavior |
|---|---|
| normal regime (`ts ≳ 1e-3`) | full closed-form fidelity; caps are bit-identical to the bare algebra |
| tiny (`1e-40` tested) | `B ≈ 0`, `F ≈ B`, so `v_new` is bit-identical to `v` — the correct `dt → 0` semantics |
| huge (`1e300` tested) | scale capped at `1e30`, decay `e^{−B} → 0`, state relaxes to `A` (steady state); finite |

## Parameter table

Isomorphic to the LTC's (same shapes, same init ranges, same softplus
constraints — see [ltc.md](ltc.md) for the derivation of each range):

| Parameter | Shape | Init | Constraint | Role |
|---|---|---|---|---|
| `gleak` | `[units]` | U(0.001, 1) | softplus | leak conductance |
| `vleak` | `[units]` | U(−0.2, 0.2) | unconstrained | leak reversal potential |
| `cm` | `[units]` | U(0.4, 0.6) | softplus | membrane capacitance (no `unfolds/ts` scaling — `ts` enters through `B = κ·ts`) |
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

`Parameters()` returns the same 13 trainable tensors as the LTC
(`nn/cfc.go:162-169`); `erev`/`sErev` are excluded for the same
structural reason as in the LTC — learning them would flip synapse
polarity.

## A complete training loop

`CfC` satisfies the `Cell` interface, so this is
`examples/ltc-sequence` with the cell swapped and the ODE substep loop
gone — training uses the `optimizer` package (see
[training.md](training.md)) with caller-owned gradient clipping:

```go
package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/nn"
	"github.com/qorm/LNN/optimizer"
	"github.com/qorm/LNN/tensor"
)

func main() {
	rng := rand.New(rand.NewSource(42))

	const (
		inDim   = 1
		units   = 8
		seqLen  = 12
		batch   = 16
		iters   = 250
		lr      = 0.05
		maxNorm = 1.0 // global gradient-norm clip
		ts      = 1.0 // time span per step
	)

	// No unfolds parameter: the closed-form solution advances the full
	// time span in a single step.
	cell := nn.NewCfC(inDim, units, nil, rng)
	readout := nn.NewLinear(units, 1, rng)
	params := nn.ParametersOf(cell, readout)
	opt := optimizer.NewSGD(lr)

	fmt.Printf("CfC accumulator task: inDim=%d units=%d seqLen=%d batch=%d, %d trainable tensors\n",
		inDim, units, seqLen, batch, len(params))

	var first, last float64
	for it := 0; it < iters; it++ {
		// Fresh random sequences every iteration (online SGD):
		// bounded accumulator s_t = clip(s_{t-1} + 0.25*u_t, -1, 1).
		xs := make([]*autograd.Variable, seqLen)
		targets := make([]*autograd.Variable, seqLen)
		state := make([]float32, batch)
		for t := 0; t < seqLen; t++ {
			xb := make([]float32, batch)
			yb := make([]float32, batch)
			for b := 0; b < batch; b++ {
				u := float32(1)
				if rng.Intn(2) == 0 {
					u = -1
				}
				xb[b] = u
				s := state[b] + 0.25*u
				if s > 1 {
					s = 1
				} else if s < -1 {
					s = -1
				}
				state[b] = s
				yb[b] = s
			}
			xs[t] = autograd.Var(tensor.FromData(xb, batch, inDim))
			targets[t] = autograd.Var(tensor.FromData(yb, batch, 1))
		}

		ys, _ := nn.Unroll(cell, xs, nil, ts)
		var acc *autograd.Variable
		for t, y := range ys {
			diff := autograd.Sub(readout.Forward(y), targets[t])
			sq := autograd.Hadamard(diff, diff)
			if t == 0 {
				acc = sq
			} else {
				acc = autograd.Add(acc, sq)
			}
		}
		loss := autograd.Scale(autograd.MeanAll(acc), 1/float32(seqLen))

		for _, p := range params {
			p.ZeroGrad()
		}
		loss.Backward()

		// Global gradient-norm clipping (caller-owned, as with the LTC):
		// scale gradients in place, then let the optimizer step.
		var norm2 float64
		for _, p := range params {
			if p.Grad == nil {
				continue
			}
			for _, g := range p.Grad.Data {
				norm2 += float64(g) * float64(g)
			}
		}
		if norm := math.Sqrt(norm2); norm > maxNorm {
			s := float32(maxNorm / norm)
			for _, p := range params {
				if p.Grad == nil {
					continue
				}
				for i := range p.Grad.Data {
					p.Grad.Data[i] *= s
				}
			}
		}
		opt.Step(params)

		if it == 0 {
			first = float64(loss.Value())
		}
		last = float64(loss.Value())
		if it%50 == 0 || it == iters-1 {
			fmt.Printf("iter %3d  loss=%.6f\n", it, loss.Value())
		}
	}
	fmt.Printf("first=%.6f last=%.6f\n", first, last)
}
```

Actual output (Go 1.26, seed 42 — deterministic; each `loss` is
measured *before* that iteration's update):

```
CfC accumulator task: inDim=1 units=8 seqLen=12 batch=16, 15 trainable tensors
iter   0  loss=0.620651
iter  50  loss=0.048169
iter 100  loss=0.041556
iter 150  loss=0.042028
iter 200  loss=0.021624
iter 249  loss=0.029091
first=0.620651 last=0.029091
```

The loss falls from `0.620651` to `0.029091` (−95%) on a task that
requires cross-step memory.

## Verification trail

- **Red-team verdict: faithful and credible.** Equation-level audit
  against the NMI published version (Theorem 1 / Eq. (8) / Lemma 1 /
  Algorithm 1 all cross-checked) plus 10/10 numeric adversarial tests:
  threshold-crossing scan of 8001 points (jump `≤ 2.98e-8`), extreme
  `ts` finiteness (8/8), masked-synapse gradients exactly zero (9/9),
  all-parameter gradcheck with zero failures.
- **In-repo regression tests** (`nn/cfc_test.go`):
  `TestCfCGradcheckAllParameters` (finite-difference check over all 13
  parameters, max relative error `8.63e-3` — within `float32`
  central-difference noise), `TestCfCZeroMasksPureLeakClosedForm`
  (all-zero wiring reduces to the pure-leak closed form, `1e-4`),
  `TestCfCDecayFactorExprelStability` and
  `TestCfCExprelBoundaryContinuity` (the `1e-2` branch boundary),
  `TestCfCStepTinyTsFixedPoint` (`ts = 1e-40` leaves `v`
  bit-identical), `TestCfCStepRejectsBadTs` (five illegal-`ts`
  classes panic), `TestCfCDeterministicSameSeed`,
  `TestCfCParametersExcludeErev`.
- **Residual (informational, not a defect):** `erev`/`sErev` enter the
  graph as `Var` leaves, which creates dead gradients — `Parameters()`
  excludes them, so optimizers never touch them; the cost is a small
  amount of wasted backward work. The LTC bakes its reversal
  potentials into construction-time `Const` indicator matrices and
  pays nothing; the CfC could adopt the same trick.
