# Persistence: Save, Load, and the LNNS format

> English | [中文](zh/persistence.md)

**Summary:** the `serialize` package persists tensors in a compact,
versioned, little-endian binary format (magic `"LNNS"`), and `nn` builds
six Save/Load functions on top of it for the LTC and CfC cells and the
Linear layer. Loading treats its input as an untrusted byte stream: every
failure is an error (never a panic), size claims are validated before any
buffer is allocated, and a hostile stream can allocate only in proportion
to the bytes it actually delivers.

**Audience:** engineers checkpointing trained models, or anyone who wants
the wire format and its safety contract spelled out byte by byte.

## The two layers

| layer | package | role |
|---|---|---|
| tensor streams | `github.com/qorm/LNN/serialize` | the wire format itself: write/read a slice of `*tensor.Tensor`, or the `Data` of a parameter slice. Exposed separately so the format can be audited on its own. |
| model persistence | `github.com/qorm/LNN/nn` | `SaveLTC`/`LoadLTC`, `SaveCfC`/`LoadCfC`, `SaveLinear`/`LoadLinear` — a kind byte + a small header + one tensor stream, with model-level validation (masks, reversal potentials, `unfolds` bound). |

## API overview

The six model-level functions:

| function | signature | stream contents |
|---|---|---|
| `nn.SaveLTC(w, c)` / `nn.LoadLTC(r)` | `func(io.Writer, *LTC) error` / `func(io.Reader) (*LTC, error)` | kind `0`, header `inDim, units, unfolds`, 17 tensors |
| `nn.SaveCfC(w, c)` / `nn.LoadCfC(r)` | `func(io.Writer, *CfC) error` / `func(io.Reader) (*CfC, error)` | kind `1`, header `inDim, units`, 17 tensors |
| `nn.SaveLinear(w, l)` / `nn.LoadLinear(r)` | `func(io.Writer, *Linear) error` / `func(io.Reader) (*Linear, error)` | kind `2`, no header, `W` and `B` |

The four stream-level building blocks in `serialize`:

| function | role |
|---|---|
| `serialize.WriteTensors(w, ts)` / `serialize.ReadTensors(r)` | write/read a slice of tensors in the wire format |
| `serialize.WriteParameters(w, params)` / `serialize.LoadParameters(r, params)` | write the `Data` of `[]*autograd.Variable`; read values back **in place** (below) |

### Saving and loading a whole model

```go
// Save a trained cell (and a readout layer) to files.
f, _ := os.Create("cfc.model")
err := nn.SaveCfC(f, cell) // check errors on every call in real code
f.Close()

// Load it back — the seed of any RNG involved is irrelevant.
r, _ := os.Open("cfc.model")
loaded, err := nn.LoadCfC(r)
r.Close()
// loaded.Step(x, h, ts) now produces bit-identical output to cell.Step.
```

### Checkpointing parameters during training

`serialize.LoadParameters` copies values back into the given variables
**in place**: the `*autograd.Variable` pointers keep their identity, so
every graph edge that references them stays valid — you can checkpoint
mid-training without rebuilding the model or the graph:

```go
var buf bytes.Buffer
err := serialize.WriteParameters(&buf, params)   // snapshot p.Data of each param
// ... later, same process or a fresh one with identically shaped params ...
err = serialize.LoadParameters(&buf, params)     // values restored in place
for _, p := range params {
    p.ZeroGrad() // see the stale-Grad contract below
}
```

Two contracts come with it:

- **All shapes are validated before anything is copied.** A count
  mismatch or any shape mismatch is an error, and a failing load leaves
  every parameter exactly as it was.
- **Stale `Grad` is deliberately preserved.** The load overwrites each
  parameter's `Data` and leaves its `Grad` untouched: gradients
  accumulated on an earlier graph survive as stale values. A caller that
  reuses the variables in a new graph calls `ZeroGrad` first — exactly as
  before any training step ([training.md](training.md)).

## A complete example: train → save → load → resume

The program below trains a `CfC` on the same bounded-accumulator task as
`examples/cfc-sequence` ([cfc.md](cfc.md)), saves the cell and readout,
loads them into *fresh* models built with a *different* seed, verifies
bit-identical `Step` output, and resumes training from the checkpoint —
with the optimizer package ([training.md](training.md)) and caller-owned
gradient clipping:

```go
package main

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/nn"
	"github.com/qorm/LNN/optimizer"
	"github.com/qorm/LNN/serialize"
	"github.com/qorm/LNN/tensor"
)

const (
	inDim   = 1
	units   = 8
	seqLen  = 12
	batch   = 16
	lr      = 0.05
	maxNorm = 1.0 // global gradient-norm clip
	ts      = 1.0
)

func main() {
	rng := rand.New(rand.NewSource(42))

	cell := nn.NewCfC(inDim, units, nil, rng)
	readout := nn.NewLinear(units, 1, rng)
	params := nn.ParametersOf(cell, readout)
	opt := optimizer.NewSGD(lr)

	fmt.Println("== phase 1: train 60 iterations, then save ==")
	train(cell, readout, params, opt, rng, 60, 0)

	const modelFile = "cfc.model"
	const paramFile = "readout.params"
	saveFile(modelFile, func(w io.Writer) error { return nn.SaveCfC(w, cell) })
	saveFile(paramFile, func(w io.Writer) error { return serialize.WriteParameters(w, readout.Parameters()) })
	fi1, _ := os.Stat(modelFile)
	fi2, _ := os.Stat(paramFile)
	fmt.Printf("saved cfc.model (%d bytes) + readout.params (%d bytes)\n", fi1.Size(), fi2.Size())

	fmt.Println("== phase 2: load into fresh models (different seed) ==")
	rng2 := rand.New(rand.NewSource(123)) // seed is irrelevant: Load overwrites every RNG-derived field
	loaded := openModel(modelFile)
	readout2 := nn.NewLinear(units, 1, rng2)
	must(serialize.LoadParameters(openStream(paramFile), readout2.Parameters()))
	fmt.Println("LoadCfC + LoadParameters: ok")

	// The loaded cell is bit-for-bit the saved cell.
	x := autograd.Var(tensor.Uniform(rand.New(rand.NewSource(7)), -1, 1, 4, inDim))
	out1, _ := cell.Step(x, nil, 0.5)
	out2, _ := loaded.Step(x, nil, 0.5)
	same := true
	for i := range out1.Data.Data {
		if math.Float32bits(out1.Data.Data[i]) != math.Float32bits(out2.Data.Data[i]) {
			same = false
		}
	}
	fmt.Printf("bit-identical Step output after load: %v\n", same)

	fmt.Println("== phase 3: resume training from the checkpoint ==")
	params2 := nn.ParametersOf(loaded, readout2)
	opt2 := optimizer.NewSGD(lr)
	for _, p := range params2 {
		p.ZeroGrad()
	}
	train(loaded, readout2, params2, opt2, rng, 60, 60)

	fmt.Println("== hostile streams are errors, never panics ==")
	raw, _ := os.ReadFile(modelFile)
	if _, err := nn.LoadLTC(bytes.NewReader(raw)); err != nil {
		fmt.Printf("LTC loader on a CfC stream -> %v\n", err)
	}
	if _, err := nn.LoadCfC(bytes.NewReader(raw[:len(raw)/2])); err != nil {
		fmt.Printf("truncated stream           -> %v\n", err)
	}
	bad := append([]byte(nil), raw...)
	bad[13] = 99 // version byte of the tensor stream (kind + 2 int32 header + "LNNS" magic)
	if _, err := nn.LoadCfC(bytes.NewReader(bad)); err != nil {
		fmt.Printf("unknown format version     -> %v\n", err)
	}
}

// train runs iters iterations of the bounded-accumulator task, printing the
// loss (measured before each printed iteration's update) with a global
// iteration offset for readable resume output.
func train(cell nn.Cell, readout *nn.Linear, params []*autograd.Variable, opt optimizer.Optimizer, rng *rand.Rand, iters, offset int) {
	for it := 0; it < iters; it++ {
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

		if it%20 == 0 || it == iters-1 {
			fmt.Printf("iter %3d  loss=%.6f\n", it+offset, loss.Value())
		}
	}
}

func saveFile(path string, fn func(io.Writer) error) {
	f, err := os.Create(path)
	must(err)
	must(fn(f))
	must(f.Close())
}

func openStream(path string) *bytes.Reader {
	raw, err := os.ReadFile(path)
	must(err)
	return bytes.NewReader(raw)
}

func openModel(path string) *nn.CfC {
	c, err := nn.LoadCfC(openStream(path))
	must(err)
	return c
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "FATAL:", err)
		os.Exit(1)
	}
}
```

Actual output (Go 1.26, seed 42 — deterministic; each `loss` is measured
*before* that iteration's update):

```
== phase 1: train 60 iterations, then save ==
iter   0  loss=0.620651
iter  20  loss=0.184146
iter  40  loss=0.158232
iter  59  loss=0.078601
saved cfc.model (1859 bytes) + readout.params (71 bytes)
== phase 2: load into fresh models (different seed) ==
LoadCfC + LoadParameters: ok
bit-identical Step output after load: true
== phase 3: resume training from the checkpoint ==
iter  60  loss=0.054492
iter  80  loss=0.045694
iter 100  loss=0.041556
iter 119  loss=0.031060
== hostile streams are errors, never panics ==
LTC loader on a CfC stream -> nn: stream holds model kind 1 (CfC), not LTC (kind 0)
truncated stream           -> serialize: tensor 6: truncated stream: claims 256 data bytes but only 176 remain: unexpected EOF
unknown format version     -> serialize: unsupported format version 99 (this build reads version 1)
```

The resumed run continues exactly where training left off: because `SGD`
is stateless and the data-generating RNG stream is uninterrupted, the
resumed trajectory is bit-identical to uninterrupted training — the
`iter 100` loss `0.041556` matches the same-iteration printout of the
full 250-iteration run in `examples/cfc-sequence` exactly. (Stateful
optimizers — `Momentum`, `Adam` — do not persist their state in the
stream; to resume them exactly, snapshot and restore their per-parameter
buffers yourself, or accept a brief re-warmup.)

## Wire format

All integers are little-endian (`encoding/binary.LittleEndian`); floats
are IEEE-754 `float32`, little-endian.

### Tensor stream (the `"LNNS"` format)

| field | type | notes |
|---|---|---|
| magic | `[4]byte` | exactly `L N N S`; anything else is "not an lnn tensor stream" |
| version | `uint8` | `1` (the exported `serialize.Version`); other values are rejected |
| count | `uint32` | number of tensors; the stream encodes *exactly* this many — trailing bytes after the last payload are rejected as corruption |
| then, `count` times: rank | `uint8` | `0 ≤ rank ≤ 8` |
| shape | `[rank]int64` | each dimension `≥ 0` |
| data | `[size]float32` | `size` = product of shape; row-major, little-endian |

### Model stream

A model stream is a one-byte kind tag, a small fixed header, and a tensor
stream blob:

| field | type | notes |
|---|---|---|
| kind | `uint8` | `0` = LTC, `1` = CfC, `2` = Linear; loading with the wrong function is a precise error, not a misparse |
| header | `int32`s | LTC: `inDim, units, unfolds`; CfC: `inDim, units`; Linear: none |
| blob | — | the tensor stream above (`"LNNS"`, version, count, data) |

Header values must fit `int32` on the write side and be `≥ 1` on the read
side; `unfolds` is additionally bounded by the load limit `1024` (below).

### Tensor order inside a model blob

Fixed and hand-written per model type — deliberately not reflection over
struct fields, so the format is auditable line by line and stable across
refactors that rename private fields. Both cells carry the same 17
tensors; they differ only in the header:

| index | tensor | shape |
|---|---|---|
| 0 | sensory mask | `[inDim, units]` |
| 1 | recurrent mask | `[units, units]` |
| 2–14 | the 13 trainable parameters, in `Parameters()` order: `gleak, vleak, cm, mu, sigma, w, sMu, sSigma, sW, inW, inB, outW, outB` | `[units]` / `[units, units]` / `[inDim, units]` / `[inDim]` per the parameter table in [ltc.md](ltc.md) |
| 15 | `erev` (recurrent reversal potentials) | `[units, units]` |
| 16 | `sErev` (sensory reversal potentials) | `[inDim, units]` |

Linear carries two tensors: `W` `[in, out]` and `B` `[out]` (the layer's
dimensions live in `W`'s shape; the header is just the kind byte).

## The untrusted-stream safety contract

The rest of the library reports misuse by panicking: its inputs come from
the program itself, so a bad shape is a bug in the caller. Serialization
is the deliberate exception. A load path consumes bytes from *outside*
the program — files, networks, checkpoints from other versions — which
may be corrupt, truncated, or outright hostile. **Every failure on the
read path is returned as an error, never a panic**, and a hostile stream
can allocate only in proportion to the bytes it actually delivers:

**Fixed limits, validated before any allocation.** Claimed ranks,
dimensions and counts are checked first, with overflow-safe
multiplication (`math/bits.Mul64`, the same discipline as
`tensor.Size`):

| limit | value | meaning |
|---|---|---|
| `maxElems` | `2^30` float32s | one tensor's payload ≤ 4 GiB |
| `maxCount` | `2^20` tensors | tensors per stream |
| `maxRank` | `8` | axes per tensor (the library's ops are 1D/2D-focused) |
| `maxUnfolds` | `1024` | **load-path only**, `LoadLTC`: checked before the blob is even parsed, so a hostile `unfolds` cannot bill any construction or stepping work. `NewLTC`'s runtime contract is unchanged (it still requires only `unfolds >= 1`): a constructor's inputs come from your own code, a load's input is an untrusted byte stream that gets no vote on its own resource budget |

A stream claiming a `1<<62`-wide dimension is rejected with an error,
not serviced with a petabyte-sized `make()` — regression-tested with
allocation counts (`TestHostileDimDoesNotAllocate`,
`TestHostileCountDoesNotAllocate`: no more than 50 small allocations
for the whole hostile decode).

**Two allocation strategies by reader capability.**

- *Readers that report their remaining length* (the `Len()` method of
  `bytes.Buffer`, `bytes.Reader`, `strings.Reader`): every payload claim
  is checked against the bytes actually left, so an oversized or
  truncated claim is rejected **before** its buffer is allocated, and a
  fitting claim is serviced by a single full-size `make` (the fast
  path).
- *Readers without a length* (`io.Pipe`, `net.Conn`, `gzip.Reader`):
  nothing can be proven up front, so payload buffers use progressive
  allocation — they start small (at most one 4,096-float32 chunk,
  16 KiB) and grow only as bytes arrive. A stream claiming `2^30`
  elements but stopping after its 18-byte header peaks at a few chunks
  of memory (~33 KiB — against a 4 GiB `make` before hardening) and
  fails with `io.ErrUnexpectedEOF`. Peak allocation stays proportional
  to the delivered bytes; a complete stream still ends with all
  elements in a single slice.

**Model-level validation, in order.** The load functions check, all
before constructing any cell: the kind byte (exact match — cross-loading
is a named error); header dims `≥ 1` and `unfolds` within `[1, 1024]`;
tensor count (exactly 17 for cells, 2 for Linear); mask shapes matching
the header and every mask entry exactly `0` or `1`; and the reversal
potentials — every `erev`/`sErev` entry must be **exactly `+1` or `−1`**
(a bitwise comparison, so `NaN`, `±Inf`, `0` and fractions like `2.5` are
all rejected). The constructors fix those signs and training excludes the
potentials from `Parameters()`, so a stream carrying anything else
describes a cell `NewLTC`/`NewCfC` could never have produced, and is
refused. Shape mismatches against an existing model (`LoadParameters`,
`copyFields`) are validated **before any value is copied**, so a failing
load leaves the destination exactly as it was.

**Fuzzed.** Mutation fuzzing pins the contract: `TestMutatedTensorStreamsNeverPanic`
and `TestMutatedModelStreamsNeverPanic` (bit flips, deletions, inserts,
block swaps). The red team ran 7,500 mutants against the original
implementation (0 panics, 0 silent misdecodes) and a further 1,200
mutants × both reader classes after the resource-exhaustion hardening
(again 0 panics).

## Bit-exactness

`Load` reconstructs a cell by running its constructor with a throwaway
RNG and then overwriting every RNG-derived field from the stream, so the
result is **independent of the seed**: for identical inputs, a loaded
cell produces bit-identical `Step` outputs and `Parameters()` values
(compared at the `Float32bits` level, `NaN` and `−0` included). Values
are copied into the fresh cell's existing storage; nothing about the
stream aliases into the returned cell. For the LTC one extra step
matters: the constructor bakes the reversal potentials into sparse
reduction-indicator matrices at construction, so `LoadLTC` rebuilds those
indicators in place from the *streamed* `erev`/`sErev` before returning —
which is also why an all-flipped ±1 pattern loads fine (the check accepts
any ±1 assignment, not just the ones the constructor samples).

The write path handles in-memory tensors the caller owns; it still
returns errors rather than panicking (nil tensors, a `Shape`/`Data`
disagreement, rank above 8, count above the limit), so a Save loop can
report I/O failures uniformly.

## Versioning: honest about the future

`version = 1` is the only layout this build reads. A stream with any
other version byte fails with `unsupported format version N (this build
reads version 1)` — future versions will **error out rather than
mis-parse** an unknown layout. If the wire format ever changes, the
version bumps and `ReadTensors` grows explicit old-layout support; there
is no silent best-effort decoding. The format version is exported as
`serialize.Version` for exactly this kind of check on the caller side.

---

See [training.md](training.md) for the loop this plugs into,
[pitfalls.md](pitfalls.md) for the user-facing safety boundary, and
[architecture.md](architecture.md) for the layers `serialize` sits on
(it imports only `tensor` and `autograd`).
