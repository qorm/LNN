# Shapes and broadcasting

> English | [中文](zh/shapes-and-broadcasting.md)

**Summary:** lnn uses dense row-major shapes with an explicitly enumerated
broadcasting subset (five rules, anything else panics), and its reduction
output shapes are *not* fully symmetric — read the tables below before
relying on one.

**Audience:** anyone writing forward or loss code; this is the reference
for every "what shape comes out?" question.

## Layout and ranks

`Shape []int` + flat row-major `Data []float32`; element `(i, j)` of an
`[m, n]` tensor is `Data[i*n+j]`. The library is 1D/2D-focused:

| 2D only (panic on other ranks) | any rank |
|---|---|
| `MatMul`, `Transpose`, `Rows`, `Cols`, `ConcatCol`, `SliceCol`, `SliceRow`, `SoftmaxRows`, `LogSoftmaxRows`, `SumRows`, `SumCols` | `Add`, `Sub`, `Hadamard`, `Scale`, `Neg`, `Apply`, `Tanh`, `Sigmoid`, `Exp`, `Log`, `Pow`, `Softplus`, `Clip`, `SumAll`, `MeanAll` |

`MatMul` is matrix multiplication only: `[m, k] × [k, n] → [m, n]`, with a
panic on rank or inner-dimension mismatch.

## Broadcasting rules (complete list)

Binary elementwise ops (`Add`, `Sub`, `Hadamard`) accept exactly these
combinations — implemented as a `switch` in `tensor/ops.go`
(`broadcastShape`), not general NumPy-style broadcasting:

| # | a | b | result | example |
|---|---|---|---|---|
| 1 | shape S | identical shape S | S | `[2,3] ⊕ [2,3] → [2,3]` |
| 2 | scalar (any tensor with exactly one element) | anything | shape of b | `[1] ⊕ [2,3] → [2,3]`; `[1,1]` is also a scalar |
| 3 | anything | scalar | shape of a | `[2,3] ⊕ [1] → [2,3]` |
| 4 | `[m, n]` | row vector `[n]` or `[1, n]` (n must match) | `[m, n]` | `[2,3] ⊕ [3] → [2,3]` |
| 5 | row vector | `[m, n]` | `[m, n]` | `[3] ⊕ [2,3] → [2,3]` |
| 6 | `[m, 1]` column vector | `[m, n]` (same row count) | `[m, n]` | `[2,1] ⊕ [2,3] → [2,3]` |
| 7 | `[m, n]` | `[m, 1]` column vector | `[m, n]` | `[2,3] ⊕ [2,1] → [2,3]` |
| 8 | `[m, 1]` | row vector `[n]` or `[1, n]` | **outer product** `[m, n]` | `[2,1] ⊗ [3] → [2,3]` |
| 9 | row vector | `[m, 1]` | **outer product** `[m, n]` | `[3] ⊗ [2,1] → [2,3]` |

Every other combination panics:

```
tensor: shapes [2 3] and [2 2] are not broadcastable
```

Notably, two 1D tensors of *different* lengths never broadcast (`[2] ⊕ [3]`
panics), and equal element counts with different layouts never do either
(`[1,6] ⊕ [2,3]` panics).

### Output shape promotions

Two consequences of the implementation to know about:

- **1D vs 1D results are promoted to `[1, n]`.** `Add([n], [n])` returns
  shape `[1, n]`, not `[n]` (rule 1 resolves to `[n]`, and the binary-op
  driver then promotes 1D outputs to 2D).
- **Scalar × scalar yields `[1,1]`, not `[1]`.** `[1] ⊕ [1]` resolves via
  rule 1 to `[1]` and is then promoted to `[1,1]` by the 1D-output rule
  above. Only genuinely 0-dimensional operands (shape `[]`, i.e.
  `tensor.New()`) produce a `[1]` result, and mixing `[]` with `[1]`
  operands is order-dependent.

All verified at a glance (each line is a real call on `m := [2,3]`):

```go
tensor.Add(m, m).Shape                        // [2 3]   (rule 1)
tensor.Add(m, scalar).Shape                   // [2 3]   (rule 2/3)
tensor.Add(m, rowVec).Shape                   // [2 3]   (rule 4/5)
tensor.Add(m, colVec).Shape                   // [2 3]   (rule 6/7)
tensor.Hadamard(colVec, rowVec).Shape         // [2 3]   (rule 8/9, outer product)
tensor.Add(tensor.New(3), tensor.New(3)).Shape // [1 3]  (1D⊕1D promoted)
```

## Reduction shapes (quick reference)

For input `[m, n]`:

| op | axis | output shape | notes |
|---|---|---|---|
| `SumAll` | all | `[1]` | `0` for an empty tensor |
| `MeanAll` | all | `[1]` | **panics** on an empty tensor (mean of zero elements is undefined) |
| `SumRows` | 0 (over rows) | **`[1, n]`** | stays 2D so it re-broadcasts against `[m, n]` |
| `SumCols` | 1 (over columns) | **`[m]`** | 1D |
| `SoftmaxRows` / `LogSoftmaxRows` | per row | `[m, n]` | max-subtracted, numerically stable; zero-column input yields an empty result of the same shape |
| `SliceRow(i)` | — | `[1, n]` | copy |
| `SliceCol(from, to)` | — | `[m, to-from]` | copy; panics on invalid/empty ranges |
| `autograd.Col(i)` | — | `[m, 1]` | `SliceCol(i, i+1)` |
| `autograd.GatherRows(a, idx)` | — | `[m]` (1D) | `len(idx)` must equal the row count |
| `ConcatCol(...)` | cols | `[m, Σn]` | all inputs share `m` |
| `Transpose` | — | `[n, m]` | |
| `MatMul(a, b)` | — | `[m, n]` | from `[m, k] × [k, n]` |
| `Stack(k tensors of shape S)` | new lead | `[k, S...]` | **experimental:** produces 3D for 2D inputs, which no other op consumes (they panic on rank ≠ 2) |

### The honest part: SumRows and SumCols are asymmetric

`SumRows` returns `[1, n]` while `SumCols` returns `[m]` — one keeps a
matrix, the other drops to a vector. There is no mathematical reason; it is
historical: the conventions grew with the code, and both `autograd`
backward passes and `nn` internals are built on them (e.g. the
`LogSoftmaxRows` gradient reshapes a `SumCols` result to `[m, 1]`, while
bias gradients rely on `SumRows` staying row-broadcastable). Changing
either shape now would be an API-breaking change across the module, so it
is tracked as a separate evaluation item rather than fixed in place
(roadmap — see [pitfalls.md](pitfalls.md)). Until then: check the table
above, and prefer `autograd` ops (which handle the reductions internally)
over manual `tensor` reductions in new code.

The same lineage explains the 1D⊕1D → `[1, n]` promotion: leaf gradients
of 1D parameters still come back in the leaf's own shape (`SumToShape`
below reduces to the *operand* shape), so the promotion rarely leaks into
user code — but it does show in intermediate `Data.Shape` values.

## Backward reductions: SumToShape

Broadcasting ops reduce output gradients back to operand shapes with
`tensor.SumToShape(grad, shape)`. For a gradient of shape `[m, n]`:

| target shape | reduction | result |
|---|---|---|
| `[m, n]` | identity (clone) | `[m, n]` |
| `[1]` (scalar) | `SumAll` | `[1]` |
| `[n]` | column sums (`SumRows`, reshaped to 1D) | `[n]` |
| `[1, n]` | column sums | `[1, n]` |
| `[m, 1]` | row sums (`SumCols`, reshaped) | `[m, 1]` |
| anything else | — | **panic** |

This is what makes leaf gradients always match leaf shapes even when the
forward pass broadcast them (including 1D leaves whose forward output was
promoted to `[1, n]`).
