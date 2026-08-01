# Profile-guided optimization (PGO)

> English | [中文](zh/pgo.md)

**Summary:** what Go's profile-guided optimization does and does not buy
for binaries built on this library — with numbers measured on the
repository's own benchmark suite, the single compiler decision they all
hinge on, and the workflow for profiling *your* binary (the only place a
profile can live: LNN is a library and ships no profile).

**Audience:** users chasing throughput; anyone wondering whether a
`default.pgo` belongs next to their `main` package.

Toolchain facts in this guide come from the Go team's PGO
documentation, <https://go.dev/doc/pgo>. The measurement commands are
packaged as the `make pgo-profile` / `make pgo-bench` targets (see
below).

## The short version

- PGO exists since Go 1.20; since Go 1.21 `go build` picks up a
  `default.pgo` file in the main package's directory automatically
  (`-pgo=auto`); `-pgo=<path>` selects a profile explicitly
  ([go.dev/doc/pgo](https://go.dev/doc/pgo)).
- As of Go 1.22 the Go team reports **around 2–14 %** improvement on a
  representative set of Go programs (same source). Treat that — not the
  biggest number below — as the expectation for a real application.
- Measured here (Go 1.26.5, Apple M4): when the profile flips one
  specific inlining decision, elementwise tensor ops drop **−57 ~ −70 %
  ns/op** and the `nn` cell benchmarks **≈ −4 ~ −6 %** (a smaller share
  than the pre-fusion −7 %: the fused LTC kernel no longer routes its
  hot loop through the inlined wrappers); when it does not, the gain is
  **≈ 0**. Both outcomes are reproduced below — the effect is real but
  *bimodal*, and it is not under the user's direct control.
- PGO never changes numerics: allocations are identical op-for-op and
  the `examples/cfc-sequence` output is bit-identical with and without
  the profile (verified below).

## Why a library ships no profile

PGO is applied **per main package**: the profile is either a
`default.pgo` next to *your* `package main`, or a path passed to
`go build -pgo=`. A library has no main package, so there is nowhere a
shipped profile would attach. And the optimization is whole-program:
per the Go documentation, "PGO in Go applies to the entire program …
including packages in dependencies. This means that the unique way your
application uses a dependency impacts the optimizations applied to that
dependency." The profile that helps is the one captured from *your*
workload — which only you have.

## Measured on this repository

Environment: `go1.26.5 darwin/arm64`, Apple M4 (10 cores, 16 GB),
macOS 26.5.2 (Darwin 25.5.0). Method: profile collected from the two `nn`
benchmarks (`go test ./nn -run '^$' -bench 'BenchmarkLTCStep|BenchmarkUnrollBackward'
-benchtime=2s -cpuprofile=…`), then the full 17-benchmark suite run
with `-pgo` on and off, interleaved A/B/A/B, `-count=3` per pass (n = 6
per cell), `-benchmem`. All 17 benchmarks, mean ns/op (measured on the
post-fusion tree; the `tensor`/`autograd` baseline column agrees with
this table's pre-fusion edition within run-to-run noise, and the two
headline `nn` rows are the fused-kernel baselines):

| benchmark | baseline | PGO | Δ ns/op | allocs |
|---|---:|---:|---:|---|
| tensor/AddBroadcastRow | 30,970 | 9,388 | **−69.7 %** | unchanged |
| tensor/Hadamard | 20,288 | 8,679 | **−57.2 %** | unchanged |
| autograd/ChainForwardBackward | 401,864 | 319,478 | **−20.5 %** | unchanged |
| autograd/DivDenLoop | 297,555 | 273,970 | −7.9 % | unchanged |
| autograd/GatherRowsBackward | 10,590 | 9,765 | −7.8 % (t = 1.1, within baseline spread) | unchanged |
| nn/UnrollPeakMemory512 | 69,342,589 | 64,999,033 | −6.3 % | unchanged |
| nn/UnrollRematCfC | 1,869,180 | 1,755,819 | −6.1 % (t = 1.9, within spread) | unchanged |
| nn/UnrollRemat | 3,447,318 | 3,241,136 | −6.0 % | unchanged |
| nn/UnrollRematPeakMemory512 | 165,574,634 | 156,075,163 | −5.7 % | unchanged |
| nn/LTCStep | 90,618 | 87,395 | −3.6 % | unchanged |
| nn/UnrollBackward | 1,376,367 | 1,327,337 | −3.6 % | unchanged |
| tensor/MatMul64 | 76,079 | 74,341 | −2.3 % | unchanged |
| tensor/MatMul128 | 647,442 | 636,798 | −1.6 % | unchanged |
| tensor/SumCols | 7,378 | 7,313 | −0.9 % | unchanged |
| tensor/SoftmaxRows | 44,036 | 45,828 | +4.1 % (within spread) | unchanged |
| tensor/SumRows | 8,262 | 7,405 | −10.4 % (one baseline outlier; medians 7,642 → 7,411) | unchanged |
| tensor/Transpose | 10,482 | 11,093 | +5.8 % (within run-to-run spread) | unchanged |

Welch t-statistics (n = 6 vs 6) for the five big movers:
AddBroadcastRow 64.3, Hadamard 73.7, ChainForwardBackward 10.7,
UnrollRemat 5.6, DivDenLoop 5.1 — all clearly significant. LTCStep
(2.1) and UnrollBackward (2.9) are marginal; GatherRowsBackward (1.1),
UnrollRematCfC (1.9), the MatMul pair (~1.6) and
SoftmaxRows/SumRows/SumCols/Transpose move less than their own spread.

### The catch: one bistable inlining decision

Every gain above traces back to a single compiler event. The
elementwise wrappers (`tensor.Add`, `tensor.Sub`, `tensor.Hadamard`,
…) call `broadcastBinary(a, b, closure)` (`tensor/ops.go:185`), whose
inner loop calls the closure **indirectly once per element**. With the
profile applied, `broadcastBinary` is marked hot, its inline budget is
raised, and `-gcflags=-m` shows it disappearing into all three wrappers
(`Add` at `ops.go:272`, `Sub` at 281, `Hadamard` at 291):

```
tensor/ops.go:272:24: inlining call to broadcastBinary
tensor/ops.go:272:24: inlining call to Add.func1
```

One indirect call per element removed: on the 128×128 benchmark that is
16,384 calls, 31.0 µs → 9.4 µs (≈ 1.9 → 0.57 ns/element). The
benchmarks whose hot loops are wrapper arithmetic gain in proportion:
`ChainForwardBackward` (16 layers of `Add(Hadamard(v, w), x)`) −20.5 %,
`DivDenLoop` −7.9 %. The `nn` cell benchmarks gain less than the −7 %
measured before stage 16: the fused LTC kernel (`nn/ltc_fused.go`)
executes the ODE unfolds as one `FusedOp` node that no longer routes
through the broadcast wrappers at all, so only the non-fused remainder
of the step benefits (LTCStep/UnrollBackward −3.6 %), while the remat
pair — whose recompute sweeps rebuild ordinary per-step graphs — keeps
a larger share (−6 %).

But the decision is bistable. Three profiles were collected with the
*identical* command minutes apart, plus a merge of all three
(`go tool pprof -proto a b c`):

| profile | `broadcastBinary` inlined? | measured effect |
|---|---|---|
| #1 (2 s) | **yes** (all three wrappers) | the table above |
| #2 (2 s, same command) | **yes** (all three wrappers) | — (not re-measured) |
| #3 (5 s) | no | — |
| merged #1+#2+#3 | no | ≈ 0 (LTCStep and Hadamard baseline-level) |

With a profile from the second world the picture flips: the compiler
spends the hot budget *inside* `broadcastBinary` (inlining
`broadcastShapeFresh`, cost ≈1190, and `bcastMode`, per the `-d
pgodebug=1` trace), and no `inlining call to broadcastBinary` line
appears at the wrappers — consistent with `broadcastBinary`'s cost
(811 standalone) being inflated past even the expanded hot-path budget
(2000) once those callees are absorbed. Which of the two worlds you get
depends on incidental sample distribution one level deeper in the call
tree — sampling luck, not anything the profile collector controls. The
Go documentation's own
warning applies verbatim: "*microbenchmarks are usually bad candidates
for PGO profiling*" — the table above applies a benchmark profile back
to the same benchmarks; a profile from a real application may land in
either world.

End-to-end, on this library's own example: `examples/cfc-sequence`
runs in ≈ 0.30 s with and without PGO — the toy workload is below
timing resolution, and its stdout is **bit-identical**
(`first loss 0.620651 -> final loss 0.029091`). Binary size grew
2,652,082 → 2,666,898 bytes (+0.6 %, the documented "slightly larger
binaries due to additional function inlining").

## Workflow: profile your own binary

1. **Collect a CPU profile from real (or realistic) load.** Any pprof
   CPU profile works — `runtime/pprof.StartCPUProfile`, or
   `net/http/pprof`'s `/debug/pprof/profile?seconds=30` for a service.
   Production load is best; the Go documentation stresses that
   representativeness matters more than duration.
2. **Build with it.** Either drop it as `default.pgo` in the main
   package's directory (picked up automatically since Go 1.21), or pass
   it explicitly: `go build -pgo=/path/to/profile.pprof`. The Go team
   recommends *committing* `default.pgo` for reproducible builds —
   profiles are source-stable (renames and edits degrade matches
   gracefully) and portable across GOOS/GOARCH.
3. **Verify what you got.** Re-run your own A/B measurement. To check
   whether the big elementwise win described above fired for your
   build, grep the compiler's inline decisions:
   `go build -pgo=… -gcflags='-m' ./... 2>&1 | grep 'inlining call to broadcastBinary'` —
   a match means it fired.
4. **Re-profile periodically**, especially after large refactors; stale
   profiles degrade gracefully (unmatched code just loses the extra
   optimization), they never miscompile.

### Reproducing the numbers above in this repo

```
make pgo-profile                       # writes /tmp/lnn.pprof (override PGOFILE=…)
make bench      > /tmp/base.txt        # baseline
make pgo-bench  > /tmp/pgo.txt         # same suite with the profile applied
benchstat /tmp/base.txt /tmp/pgo.txt   # or eyeball the two files
```

The profile is a throwaway artifact outside the repository; nothing is
committed. Note the bimodality described above: if your collected
profile lands in the "no inlining" world, re-collect or expect the
modest end of the range.

## Caveats

- **Expect 2–14 %, not 68 %.** The headline elementwise numbers are
  real but rest on one inlining coin-flip; the Go team's fleet-wide
  figure is the honest prior for an arbitrary application.
- **PGO is not a substitute for algorithmic work** — it shuffles
  inlining and layout; it does not change what your code computes.
  Here: same allocations, same bits out.
- **Builds cost more** the first time (every package is rebuilt against
  the profile; cached afterwards), and binaries grow slightly.
- An unrepresentative profile should not make your program slower — the
  [Go documentation's FAQ](https://go.dev/doc/pgo) ("Will PGO with an
  unrepresentative profile make my program slower than no PGO?") answers
  literally "*It should not.*" It just optimizes the wrong (cold)
  functions; the hot parts should not get slower.

## Why this repository ships no `default.pgo`

- The library has no main package; the examples are the only mains,
  and their runtime (≈ 0.3 s) is below PGO's timing resolution.
- A committed profile is a binary blob that would need re-collection as
  the code and the Go toolchain evolve — for a measured benefit of zero
  on the only binaries in the repo.
- Consumers get strictly more from a profile of *their* workload; the
  workflow above is the deliverable, not a blob.
