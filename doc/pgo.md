# Profile-guided optimization (PGO)

> English | [中文](zh/pgo.md)

**Summary:** what Go's profile-guided optimization does and does not buy
for binaries built on this library — with numbers measured on the
repository's own benchmark suite, the few compiler decisions they hinge
on, and the workflow for profiling *your* binary (the only place a
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
- Measured here (Go 1.26.5, Apple M4) on the v0.6.0 tree: a profile that
  flips the inlining decision for **all three** broadcast wrappers drops
  the elementwise ops **−57 ~ −69 % ns/op** (`AddBroadcastRow` −69 %,
  `Hadamard` −58 %; re-measured at the v0.6.0 resample, unchanged from
  the pre-fusion numbers). A **fresh** profile collected on this tree
  flips the decision for `Hadamard` alone — `Hadamard` still gets −57 %,
  while `Add`/`Sub` stay ≈ 0 — and a merged profile flips none (≈ 0
  everywhere). The `nn` fused-kernel benchmarks get **≈ 0 to −3 %**
  under *every* profile: the fused LTC/CfC kernels route almost no
  wrapper arithmetic, so the fusion closed most of what PGO used to give
  the cell steps. The effect is real but *multi-state*, and the state is
  not under the user's direct control.
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
macOS 26.5.2 (Darwin 25.5.0). All numbers below come from the v0.6.0
resample, run on a clean `git archive HEAD` snapshot of the release tree
(isolated from concurrent worktree edits) in one session with the three
states interleaved **A/B/C/A/B/C** (`-count=3` per pass, n = 6 per
cell, `-benchmem`): **A** = no PGO, **B** = the *old* stage-16 profile
(the one this table originally shipped, kept in order to answer the
"stale profile" question), **C** = a *fresh* profile collected on this
tree with the standard command (`go test ./nn -run '^$' -bench
'BenchmarkLTCStep|BenchmarkUnrollBackward' -benchtime=2s
-cpuprofile=…`). Reported values are the **median** ns/op of the six
per-cell runs — a deliberate change from the earlier edition's mean,
adopted because the median is robust to the single-session outliers that
marked the previous table (stage 19's LTCStep row). Allocations are
identical op-for-op in every state (verified below):

| benchmark | baseline | old profile | fresh profile | Δ old | Δ fresh | allocs |
|---|---:|---:|---:|---:|---:|---|
| tensor/AddBroadcastRow | 31,966 | 9,918 | 32,330 | **−69.0 %** | +1.1 % (ns) | unchanged |
| tensor/Hadamard | 21,402 | 9,066 | 9,165 | **−57.6 %** | **−57.2 %** | unchanged |
| autograd/ChainForwardBackward | 416,292 | 319,454 | 416,041 | **−23.3 %** | −0.1 % (ns) | unchanged |
| autograd/DivDenLoop | 298,902 | 289,278 | 301,741 | −3.2 % (ns) | +0.9 % (ns) | unchanged |
| autograd/GatherRowsBackward | 10,220 | 10,344 | 10,400 | +1.2 % (ns) | +1.8 % (ns) | unchanged |
| nn/UnrollPeakMemory512 | 65,169,079 | 63,463,448 | 63,760,520 | −2.6 % (ns) | −2.2 % (ns) | unchanged |
| nn/UnrollRematCfC | 884,883 | 874,745 | 882,294 | −1.1 % (ns) | −0.3 % (ns) | unchanged |
| nn/UnrollRemat | 3,391,914 | 3,306,071 | 3,338,338 | −2.5 % (t = 2.7) | −1.6 % (ns) | unchanged |
| nn/UnrollRematPeakMemory512 | 167,316,423 | 160,131,072 | 160,577,974 | −4.3 % (ns) | −4.0 % (ns) | unchanged |
| nn/LTCStep | 82,980 | 83,015 | 80,632 | +0.0 % (ns) | −2.8 % (ns) | unchanged |
| nn/UnrollBackward | 1,361,163 | 1,333,687 | 1,318,196 | −2.0 % (ns) | −3.2 % (t = 3.2) | unchanged |
| tensor/MatMul64 | 77,739 | 79,414 | 78,792 | +2.2 % (ns) | +1.4 % (ns) | unchanged |
| tensor/MatMul128 | 665,644 | 662,934 | 667,760 | −0.4 % (t = 2.2) | +0.3 % (ns) | unchanged |
| tensor/SumCols | 7,650 | 7,684 | 7,758 | +0.4 % (ns) | +1.4 % (ns) | unchanged |
| nn/CfCStep | 36,309 | 36,162 | 35,734 | −0.4 % (ns) | −1.6 % (t = 2.5) | unchanged |
| tensor/SoftmaxRows | 46,322 | 46,661 | 46,041 | +0.7 % (ns) | −0.6 % (ns) | unchanged |
| tensor/SumRows | 7,831 | 7,820 | 7,836 | −0.1 % (ns) | +0.1 % (ns) | unchanged |
| tensor/Transpose | 10,674 | 11,042 | 10,734 | +3.5 % (ns) | +0.6 % (ns) | unchanged |

Welch t-statistics (n = 6 vs 6). Under the **old** profile the three big
elementwise movers are all clearly significant: AddBroadcastRow 39.6,
Hadamard 33.5, ChainForwardBackward 20.1 (p < 0.001 each); UnrollRemat
2.7 and MatMul128 2.2 are marginal. Under the **fresh** profile
`Hadamard` is the only big mover (t = 33.1, p < 0.001); UnrollBackward
(t = 3.2, p < 0.01) and CfCStep (t = 2.5) are small marginal-to-
significant wins. Every other row moves less than its own spread —
including LTCStep under both profiles (t = 0.0 and t = 1.1). One caveat:
`ChainForwardBackward` is **bimodal** under a fresh profile — ≈ 0 in the
column above (the sample behind this table), but a different fresh
collection from the same command delivered the full −23 % in a second
session. That row's gain follows a deeper call path than the wrapper
set, so it is not stable across fresh samples (see the three-world
section below).

**The stale-profile question, answered by re-collection.** The old
profile's effect on the tensor/autograd rows is unchanged: it still
flips the same three wrappers and still delivers −57 ~ −69 % — exactly
what the Go FAQ's graceful-degradation promise predicts for code that
has not changed. On the `nn` rows the old profile now costs nothing:
LTCStep is +0.0 % (the stage-19 +13.5 % adverse trend did **not**
reproduce and reads as a single-session outlier, not a stale-profile
effect), and every fused-kernel row is ≈ 0 to −3 %. A *fresh* profile
on this tree lands in a partial inlining world — `Hadamard` yes,
`Add`/`Sub` no (mechanism below) — so the elementwise win a new consumer
can expect from a fresh profile is `Hadamard` −57 % rather than the full
−57 ~ −69 %, and the `nn` rows stay ≈ 0 to −3 %.

### The catch: an inlining decision with three worlds

Every gain above traces back to a single compiler event. The
elementwise wrappers (`tensor.Add`, `tensor.Sub`, `tensor.Hadamard`,
…) call `broadcastBinary(a, b, closure)` (`tensor/ops.go:195`), whose
inner loop calls the closure **indirectly once per element**. With a
profile that marks the wrapper hot, its inline budget is raised and
`-gcflags=-m` shows `broadcastBinary` disappearing into the wrapper
(`Add` at `ops.go:291`, `Sub` at 300, `Hadamard` at 310):

```
tensor/ops.go:291:24: inlining call to broadcastBinary
tensor/ops.go:291:24: inlining call to Add.func1
```

One indirect call per element removed: on the 128×128 benchmark that is
16,384 calls, 31.0 µs → 9.9 µs (≈ 1.95 → 0.61 ns/element). The
benchmarks whose hot loops are wrapper arithmetic gain in proportion:
`ChainForwardBackward` (16 layers of `Add(Hadamard(v, w), x)`) −23.3 %,
`DivDenLoop` −3.2 % (ns). The `nn` cell benchmarks have almost nothing
left to gain: the fused LTC kernel (`nn/ltc_fused.go`) and the fused CfC
step execute their ODE unfolds as single `FusedOp` nodes that route no
wrapper arithmetic at all, and stage 19 internalized the sensory path
too — so the step kernels measure ≈ 0 under every profile (LTCStep
+0.0 %/−2.8 %, CfCStep −0.4 %/−1.6 %), and the unroll/remat kernels,
which still build ordinary per-step graphs, keep only a small −2 ~ −3 %.

The decision is not a single switch but three related ones. The old
stage-16 profile marks **all three wrappers hot** and inlines
`broadcastBinary` (cost 834, inlined into Add/Sub/Hadamard at
`ops.go:291/300/310`). A fresh profile on this tree marks **`Hadamard`
hot but not `Add`/`Sub`** — the nn benchmarks it was collected from
build Hadamard-heavy step graphs but no longer route through the
broadcast wrappers. One path into the Hadamard-only world — the one
observed in the sample behind this table's fresh column: `broadcastBinary`
absorbs its own hot callees (`New`, `bcastMode`, `IsScalar`, per the
`-d pgodebug=1` trace), inflating its cost from 834 to 956, which then
fits only the hot wrapper's budget. A second fresh collection reached
the same Hadamard-only world with cost still 834, so the cost inflation
is one route in, not the gate. Four fresh collections minutes apart,
plus a merge:

| profile | `broadcastBinary` inlined? | measured effect |
|---|---|---|
| old (stage-16) | **yes** (all three wrappers) | the table's Δ old column |
| fresh #1 (2 s) | **yes** (Hadamard only) | the table's Δ fresh column |
| fresh #2 (2 s, same command) | **yes** (Add + Hadamard) | — (not re-measured) |
| fresh #3 (2 s) | **yes** (Hadamard only) | — |
| merged fresh #1+#2+#3 | no | ≈ 0 (AddBroadcastRow, Hadamard, SoftmaxRows all baseline-level) |

So on the v0.6.0 tree a fresh profile reliably delivers the `Hadamard`
win (−57 %), sometimes also `Add`, and a merged profile lands in the
no-inlining world (≈ 0). Which wrappers you get depends on incidental
sample distribution one level deeper in the call tree — sampling luck,
not anything the profile collector controls. The three-world framing
describes the **tensor** rows; the one `autograd` mover
(`ChainForwardBackward`) does not follow the wrapper set — a collection
that inlined only `Hadamard` still delivered the full −23 % on that row
in a second session, so its fresh-profile outcome is sample-dependent
(its gain follows a deeper call path the profile happens to mark hot).
The Go documentation's own
warning applies verbatim: "*microbenchmarks are usually bad candidates
for PGO profiling*" — the table above applies a benchmark profile back
to the same benchmarks; a profile from a real application may land in
any of the three worlds.

End-to-end, on this library's own example: `examples/cfc-sequence`
runs in ≈ 0.14 s in every state — the toy workload is below timing
resolution — and its stdout is **bit-identical** with no PGO, the old
profile, and the fresh profile (sha256-equal; `first loss 0.620651 ->
final loss 0.029091`). Binary size: 2,718,962 bytes without PGO,
2,733,906 with the old profile (+0.5 %, the documented "slightly larger
binaries due to additional function inlining"), and 2,717,602 with the
fresh profile (−0.05 %, effectively unchanged — the partial world adds
almost no code).

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
   whether the elementwise win fired for your build, grep the
   compiler's inline decisions:
   `go build -pgo=… -gcflags='-m' ./... 2>&1 | grep 'inlining call to broadcastBinary'`.
   A match at the `Hadamard` call site (`ops.go:310`) is what −57 %
   needs; matches at `Add`/`Sub` (291/300) add the rest. Treat grep
   hits as a **lower bound**: an autograd chain row can still win
   without any `Add`/`Sub` hit (its gain follows a deeper call path),
   so a Hadamard-only match does not mean the whole autograd line stays
   flat. If nothing matches, your profile landed in the no-inlining
   world — re-collect or expect the conservative end of the range.
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
committed. Note the three-world behavior described above: a fresh
profile on this tree lands in the Hadamard-only world (the table's
Δ fresh column); if yours lands in the no-inlining world, re-collect or
expect the conservative end of the range.

## Caveats

- **Expect 2–14 %, not 68 %.** The headline elementwise numbers are
  real but rest on a few related inlining coin-flips; on this tree a
  fresh profile reliably lands the `Hadamard` one but not always the
  others. The Go team's fleet-wide figure is the honest prior for an
  arbitrary application.
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
  and their runtime (≈ 0.14 s) is below PGO's timing resolution.
- A committed profile is a binary blob that would need re-collection as
  the code and the Go toolchain evolve — for a measured benefit of zero
  on the only binaries in the repo.
- Consumers get strictly more from a profile of *their* workload; the
  workflow above is the deliverable, not a blob.
- If your application's hot loop is elementwise tensor arithmetic, PGO
  is worth a try (the `Hadamard` win is robust to a fresh profile); if
  it is the fused `nn` cells, expect ≈ 0 to −3 % and rely on the fusion
  itself, not PGO. Re-sample when you upgrade the toolchain or after a
  large refactor.
