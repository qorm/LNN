package nn

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/serialize"
	"github.com/qorm/LNN/tensor"
)

// This file adds model-level persistence (Save/Load) for the LTC and CfC
// cells and the Linear layer, on top of the versioned tensor stream of the
// serialize package.
//
// Model stream layout (little-endian):
//
//	kind    uint8      0 = LTC, 1 = CfC, 2 = Linear
//	header  int32s     LTC: inDim, units, unfolds; CfC: inDim, units; Linear: none
//	blob    tensors    the serialize wire format ("LNNS", version, count,
//	                  data, and — v2 — a trailing CRC-32C checksum over the
//	                  whole blob; v1 blobs without the checksum still load)
//
// The model envelope (kind byte + header) sits OUTSIDE the blob's checksum,
// so its integrity is enforced by the model-level validation instead: the
// kind byte is matched exactly, and the header dims are checked against the
// limits and against the streamed mask shapes (a flipped dim that survives
// the limits fails the mask-shape check). One residual window stays open:
// `unfolds` is checked against its [1, 1024] load limit but has no cross-
// check against the streamed tensors, so a single-bit flip that lands inside
// that range silently loads as a different-but-valid cell (a checkpoint with
// more or fewer ODE sub-steps than intended). This is inherent to keeping
// the envelope outside the checksum — integrity, not authenticity, and the
// checksum cannot see bytes it does not cover. Detecting it would need a
// whole-stream checksum covering the header too; see doc/persistence.md.
//
// Tensor order inside the blob is fixed and hand-written per model type —
// deliberately not reflection over struct fields, so the format is auditable
// line by line and stable across refactors that rename private fields:
//
//	LTC, CfC:  sensory mask, recurrent mask,
//	           the 13 trainable parameters in Parameters() order,
//	           erev, sErev
//	Linear:    W, B
//
// # Error contract
//
// Like the rest of serialize, the load path treats its input as an untrusted
// external byte stream and reports every failure — wrong kind, wrong tensor
// count, shape mismatch, non-binary wiring mask, truncation, trailing bytes —
// as an error rather than a panic. This is the documented exception to the
// library's panic-on-misuse contract: a checkpoint is exactly the kind of
// input the program does not control.
//
// # Bit-exactness
//
// Load reconstructs a cell by running its constructor with a throwaway RNG
// and then overwriting every RNG-derived field from the stream, so the
// result is independent of the seed and, for identical inputs, produces
// bit-identical Step outputs and Parameters(). Values are copied into the
// fresh cell's existing storage; nothing about the stream aliases into the
// returned cell.
//
// Model kind tags, written as the first byte of every model stream.
const (
	kindLTC uint8 = iota
	kindCfC
	kindLinear
)

// Tensor counts per model blob, mirroring the fixed field order above. Both
// cells carry the same 17 tensors; they differ only in the int32 header (the
// LTC writes unfolds, the CfC does not).
const (
	ltcTensorCount    = 2 + 13 + 2 // masks + trainable parameters + erev/sErev
	cfcTensorCount    = 2 + 13 + 2
	linearTensorCount = 2
)

// maxUnfolds caps the unfolds value LoadLTC accepts from a stream. Value:
// 1024. Real LTC configurations use 1-16 ODE substeps, so 1024 is far beyond
// any plausible deployment; a hostile stream, by contrast, turns a gigantic
// unfolds into a CPU-exhaustion loop the moment the loaded cell steps (red
// team F2: unfolds=1<<20 cost 2.26 s per load+step, extrapolating to ~38
// minutes for a single Step at 1<<30).
//
// The cap is deliberately load-only: NewLTC's runtime contract is unchanged
// (it still requires only unfolds >= 1, panicking otherwise), because a
// constructor's inputs come from the caller's own code — an extreme unfolds
// there is a bug the caller controls — while a load's input is an untrusted
// byte stream that gets no vote on its own resource budget.
const maxUnfolds = 1024

// maxUnits and maxInDim cap the units and inDim values LoadLTC and LoadCfC
// accept from a stream. Value: 2048 each (raised from 256 by the item-#14
// sparse contraction; the F1 history and the original 256 derivation are
// kept below).
//
// # History: the 256 cap (red team sweep F1, the v0.2.0 release blocker)
//
// The constructors used to materialize the synaptic reduction indicators as
// dense [pre*units, units] matrices: two of units^3 float32s on the
// recurrent side and two of inDim*units^2 on the sensory side, so load-time
// memory was O(units^3) while the header that controls it is only 9-13
// bytes. Quantified at the old caps (units = inDim = 256):
//
//	persistent  2*(units^3 + inDim*units^2)*4B = 2*(256^3 + 256*256^2)*4B
//	            = 256 MiB of indicator matrices per loaded cell
//	transient   plus one indicator rebuild while Load re-baked the numerator
//	            indicators from the streamed polarities: at most
//	            max(units^3, inDim*units^2)*4B = 64 MiB, freed right after
//	            the copy — a bounded worst-case peak of ~320 MiB.
//
// Before these caps, F1 loaded a legal units=512 stream (5 MB of delivered
// bytes) at the cost of 1,560 MB of allocations (311x amplification), and a
// minimal units=4096 attack stream made the process attempt
// 2*4096^3*4B ≈ 550 GB of recurrent indicators until the operating system
// killed it outright — worse than a panic, and a direct breach of this
// file's and the serialize package's contract that a hostile stream
// allocates only in proportion to the bytes it actually delivers. units and
// inDim were the twin face maxUnfolds had already covered for the time
// axis.
//
// # Re-derivation after item #14 (sparse contraction)
//
// The root-cause fix (item #14) replaced the dense indicators with the
// sparse fold of ltc.go's contract: no [units^2, units] tensor is ever
// materialized, in the constructor OR the load path (the transient indicator
// rebuild above is gone too — the numerator coefficients are row views of
// the streamed erev storage). Load-time memory is now O(units^2), dominated
// by the parameter matrices the stream itself delivers. Worst case, fully
// wired, with U = units = inDim (the largest legitimate header):
//
//	stream   masks 2 + mu/sigma/w 3 + sMu/sSigma/sW 3 + erev/sErev 2
//	         = 10*U^2 float32 resident as the parsed tensors  = 40*U^2 B
//	cell     the same 10*U^2 parameters and masks, plus the ident
//	         identity U^2 and the two wiring plans 2*U^2 int32
//	         = 52*U^2 B
//	peak     stream + cell held together during copyFields
//	         = 92*U^2 B = 92*2048^2 B ≈ 368 MiB at the cap
//
// — the same ~320 MiB budget class as the old regime at units=256, with 8x
// the capacity. A minimal attack stream (inDim = 1) delivers ~20*U^2 bytes
// and peaks at ~1.5x that: allocation stays in proportion to delivered
// bytes, the F1 contract now honored at the root rather than bandaged.
// units=4096 (the old jetsam PoC) would still peak at ~1.4 GiB, so the cap
// stays — the header check still fires before the blob is parsed — but at
// 2048 instead of 256.
//
// Like maxUnfolds, the caps are deliberately load-only: NewLTC/NewCfC's
// runtime contracts are unchanged (dims >= 1 remains the only requirement),
// because a constructor's inputs come from the caller's own code — an
// extreme size there is the caller's own allocation decision (and with the
// sparse contraction a units=1024 fully-wired constructor costs ~32 MB, not
// the old ~8 GiB cliff) — while a load's input is an untrusted byte stream
// that gets no vote on its own resource budget.
const (
	maxUnits = 2048
	maxInDim = 2048
)

// headerWriter writes the kind byte and the int32 header fields, capturing
// the first I/O error for a single report at the end.
type headerWriter struct {
	w   io.Writer
	err error
}

func (hw *headerWriter) write(b []byte) {
	if hw.err != nil {
		return
	}
	_, hw.err = hw.w.Write(b)
}

func (hw *headerWriter) u8(v uint8) {
	hw.write([]byte{v})
}

// i32 writes a non-negative header dimension, rejecting values that do not
// fit the format's int32 fields.
func (hw *headerWriter) i32(v int) {
	if hw.err != nil {
		return
	}
	if v < 0 || int64(v) > math.MaxInt32 {
		hw.err = fmt.Errorf("nn: header value %d does not fit the format's int32 fields", v)
		return
	}
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(v))
	hw.write(b[:])
}

// headerReader is the read side of headerWriter, normalizing both EOF
// flavors to io.ErrUnexpectedEOF (a header that ends mid-field is truncated).
type headerReader struct {
	r   io.Reader
	err error
}

func (hr *headerReader) read(b []byte) {
	if hr.err != nil {
		return
	}
	if _, err := io.ReadFull(hr.r, b); err != nil {
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			hr.err = io.ErrUnexpectedEOF
		} else {
			hr.err = err
		}
	}
}

func (hr *headerReader) u8() uint8 {
	var b [1]byte
	hr.read(b[:])
	return b[0]
}

func (hr *headerReader) i32() int {
	var b [4]byte
	hr.read(b[:])
	return int(int32(binary.LittleEndian.Uint32(b[:])))
}

func kindName(k uint8) string {
	switch k {
	case kindLTC:
		return "LTC"
	case kindCfC:
		return "CfC"
	case kindLinear:
		return "Linear"
	default:
		return "unknown"
	}
}

// readKind consumes the kind byte and requires it to equal want, giving
// cross-loaded streams (an LTC stream fed to LoadCfC, ...) a precise error.
func readKind(hr *headerReader, want uint8) error {
	kind := hr.u8()
	if hr.err != nil {
		return fmt.Errorf("nn: reading model kind: %w", hr.err)
	}
	if kind != want {
		return fmt.Errorf("nn: stream holds model kind %d (%s), not %s (kind %d)", kind, kindName(kind), kindName(want), want)
	}
	return nil
}

// wiringFromStream validates the two mask tensors read from a model stream
// and rebuilds a Wiring from them. The binary-mask and shape invariants are
// checked here and reported as errors, so newWiring's assertion panics can
// never be reached by a hostile stream.
func wiringFromStream(sensory, recurrent *tensor.Tensor, inDim, units int) (*Wiring, error) {
	if !shapeIs(sensory.Shape, inDim, units) {
		return nil, fmt.Errorf("nn: sensory mask shape %v does not match [inDim=%d, units=%d]", sensory.Shape, inDim, units)
	}
	if !shapeIs(recurrent.Shape, units, units) {
		return nil, fmt.Errorf("nn: recurrent mask shape %v does not match [units=%d, units=%d]", recurrent.Shape, units, units)
	}
	for _, v := range sensory.Data {
		if v != 0 && v != 1 {
			return nil, fmt.Errorf("nn: sensory mask entry %v outside {0, 1}", v)
		}
	}
	for _, v := range recurrent.Data {
		if v != 0 && v != 1 {
			return nil, fmt.Errorf("nn: recurrent mask entry %v outside {0, 1}", v)
		}
	}
	return newWiring(sensory, recurrent), nil
}

// checkReversals validates a streamed reversal-potential tensor (erev or
// sErev): every entry must be exactly +1 or -1. The comparison is bitwise ==,
// so NaN, ±Inf, 0 and fractional values like 2.5 are all rejected. The
// constructors fix these signs and training excludes the potentials from
// Parameters(), so a stream carrying anything else describes a cell
// NewLTC/NewCfC could never have produced — accepting it would let a hostile
// checkpoint mint excitatory/inhibitory patterns outside the model's state
// space (red team F4).
func checkReversals(name string, t *tensor.Tensor) error {
	for i, v := range t.Data {
		if v != 1 && v != -1 {
			return fmt.Errorf("nn: %s[%d] = %v is not a reversal potential (must be exactly +1 or -1)", name, i, v)
		}
	}
	return nil
}

// copyFields copies src into dst elementwise. It validates every shape pair
// before copying anything, so a stream that mismatches late leaves the early
// fields exactly as they were.
func copyFields(src, dst []*tensor.Tensor) error {
	for i := range src {
		if !tensor.SameShape(src[i], dst[i]) {
			return fmt.Errorf("nn: model tensor %d shape mismatch: stream has %v, model has %v", i, src[i].Shape, dst[i].Shape)
		}
	}
	for i := range src {
		copy(dst[i].Data, src[i].Data)
	}
	return nil
}

// throwawayRNG returns the construction RNG the Load functions pass to the
// cell constructors. Its seed is deliberately irrelevant: every value the
// constructors draw is overwritten from the stream before the cell is
// returned. A non-nil RNG is required only because the constructors draw
// during construction.
func throwawayRNG() *rand.Rand { return rand.New(rand.NewSource(0)) }

// ltcTensors returns c's tensors in stream order: the wiring masks, the 13
// trainable parameters (Parameters() order) and the reversal potentials.
// SaveLTC and LoadLTC both define the format through this single ordering.
func ltcTensors(c *LTC) []*tensor.Tensor {
	ts := make([]*tensor.Tensor, 0, ltcTensorCount)
	ts = append(ts, c.wiring.sensoryMask, c.wiring.recurrentMask)
	for _, p := range c.Parameters() {
		ts = append(ts, p.Data)
	}
	ts = append(ts, c.erev.Data, c.sErev.Data)
	return ts
}

// SaveLTC writes c to w. The stream captures every field fixed at
// construction time — hyperparameters, wiring masks, the 13 trainable
// parameters and the fixed +/-1 reversal potentials — so LoadLTC
// reproduces the cell bit for bit.
//
// Errors: any I/O failure writing the header or the tensor blob is
// returned (wrapped with its location), along with the serialize
// writer's validation errors (see serialize.WriteTensors); for a cell
// produced by NewLTC the latter cannot fire. It never panics on I/O or
// stream concerns; c must be a live cell (a nil c panics as any nil
// dereference would — programmer error, per the library's
// panic-on-misuse convention).
func SaveLTC(w io.Writer, c *LTC) error {
	hw := &headerWriter{w: w}
	hw.u8(kindLTC)
	hw.i32(c.inDim)
	hw.i32(c.units)
	hw.i32(c.unfolds)
	if hw.err != nil {
		return fmt.Errorf("nn: writing LTC header: %w", hw.err)
	}
	return serialize.WriteTensors(w, ltcTensors(c))
}

// LoadLTC reads a stream written by SaveLTC and returns an equivalent
// cell: for the same input and state, Step produces bit-identical
// outputs, independently of the RNG used to construct the destination.
//
// Errors (never panics — the stream is untrusted input, the documented
// exception to the library's panic-on-misuse convention): a cross-model
// stream (wrong kind byte), a truncated or corrupt header, dims below 1,
// an unfolds value outside [1, maxUnfolds] (1024, a load-only limit),
// units or inDim above maxUnits / maxInDim (2048 each, load-only
// limits), every serialize.ReadTensors failure (bad magic, unknown
// version, truncation — surfaced as io.ErrUnexpectedEOF — hostile size
// claims, and a v2 checksum mismatch), a wrong tensor count, mask shapes
// or non-binary mask entries, and reversal potentials that are not exactly
// +1 or -1. All header limits are checked before the blob is parsed, so a
// hostile header bills no parsing work; see doc/persistence.md for the
// byte-level contract.
func LoadLTC(r io.Reader) (*LTC, error) {
	hr := &headerReader{r: r}
	if err := readKind(hr, kindLTC); err != nil {
		return nil, err
	}
	inDim, units, unfolds := hr.i32(), hr.i32(), hr.i32()
	if hr.err != nil {
		return nil, fmt.Errorf("nn: reading LTC header: %w", hr.err)
	}
	if inDim < 1 || units < 1 {
		return nil, fmt.Errorf("nn: LTC header has invalid dims in=%d units=%d", inDim, units)
	}
	if unfolds < 1 {
		return nil, fmt.Errorf("nn: LTC header has invalid unfolds=%d", unfolds)
	}
	// All three size claims are checked before the blob is even parsed, so
	// a hostile header cannot bill any parsing or construction work:
	// unfolds would become a CPU-exhaustion loop (see maxUnfolds), and
	// units/inDim would become O(units^2) parameter allocations far above
	// the documented load budget (see maxUnits/maxInDim, red team sweep F1).
	if unfolds > maxUnfolds {
		return nil, fmt.Errorf("nn: LTC header has unfolds=%d, exceeding the load limit %d", unfolds, maxUnfolds)
	}
	if units > maxUnits {
		return nil, fmt.Errorf("nn: LTC header has units=%d, exceeding the load limit %d", units, maxUnits)
	}
	if inDim > maxInDim {
		return nil, fmt.Errorf("nn: LTC header has inDim=%d, exceeding the load limit %d", inDim, maxInDim)
	}
	ts, err := serialize.ReadTensors(r)
	if err != nil {
		return nil, err
	}
	if len(ts) != ltcTensorCount {
		return nil, fmt.Errorf("nn: LTC stream holds %d tensors, want %d", len(ts), ltcTensorCount)
	}
	wiring, err := wiringFromStream(ts[0], ts[1], inDim, units)
	if err != nil {
		return nil, err
	}
	// Reversal potentials sit at the blob's tail; validate them before the
	// cell is built, so no cell (even the throwaway) exists for a stream
	// outside the model's state space.
	if err := checkReversals("erev", ts[ltcTensorCount-2]); err != nil {
		return nil, err
	}
	if err := checkReversals("sErev", ts[ltcTensorCount-1]); err != nil {
		return nil, err
	}
	cell := NewLTC(inDim, units, wiring, unfolds, throwawayRNG())

	dsts := make([]*tensor.Tensor, 0, 15)
	for _, p := range cell.Parameters() {
		dsts = append(dsts, p.Data)
	}
	dsts = append(dsts, cell.erev.Data, cell.sErev.Data)
	if err := copyFields(ts[2:], dsts); err != nil {
		return nil, err
	}
	// The numerator coefficients are row views sharing the erev/sErev Data
	// arrays that copyFields just overwrote with the streamed polarities, so
	// the contraction picks them up with no rebuild (the pre-#14 design
	// re-materialized a dense [pre*units, units] indicator here).
	return cell, nil
}

// cfcTensors returns c's tensors in stream order, mirroring ltcTensors. The
// reversal potentials are plain tensors now (baked into the numerator
// indicators at construction), so they go into the stream directly; the
// order itself is unchanged.
func cfcTensors(c *CfC) []*tensor.Tensor {
	ts := make([]*tensor.Tensor, 0, cfcTensorCount)
	ts = append(ts, c.wiring.sensoryMask, c.wiring.recurrentMask)
	for _, p := range c.Parameters() {
		ts = append(ts, p.Data)
	}
	ts = append(ts, c.erev, c.sErev)
	return ts
}

// SaveCfC writes c to w. As with the LTC, the stream captures the
// wiring masks, all 13 trainable parameters and the reversal
// potentials; the CfC has no unfolds, so the header is just inDim and
// units.
//
// Errors: exactly as SaveLTC — I/O failures (wrapped with their
// location) plus the serialize writer's validation errors, which cannot
// fire for a cell produced by NewCfC. It never panics on I/O or stream
// concerns; c must be a live cell.
func SaveCfC(w io.Writer, c *CfC) error {
	hw := &headerWriter{w: w}
	hw.u8(kindCfC)
	hw.i32(c.inDim)
	hw.i32(c.units)
	if hw.err != nil {
		return fmt.Errorf("nn: writing CfC header: %w", hw.err)
	}
	return serialize.WriteTensors(w, cfcTensors(c))
}

// LoadCfC reads a stream written by SaveCfC and returns an equivalent
// cell with bit-identical Step behavior, independently of the RNG used
// to construct the destination.
//
// Errors (never panics — the stream is untrusted input): the same
// classes as LoadLTC minus unfolds (the CfC has none) — wrong kind
// byte, truncated or corrupt header, dims below 1, units or inDim above
// maxUnits / maxInDim (2048 each, load-only limits), every
// serialize.ReadTensors failure (bad magic, unknown version,
// truncation as io.ErrUnexpectedEOF, hostile size claims, and a v2
// checksum mismatch), a wrong tensor count, bad wiring masks, and
// reversal potentials that are not exactly +1 or -1. As in LoadLTC, the
// numerator coefficients are row
// views of the erev/sErev storage, so overwriting the streamed
// polarities updates them in place. See doc/persistence.md.
func LoadCfC(r io.Reader) (*CfC, error) {
	hr := &headerReader{r: r}
	if err := readKind(hr, kindCfC); err != nil {
		return nil, err
	}
	inDim, units := hr.i32(), hr.i32()
	if hr.err != nil {
		return nil, fmt.Errorf("nn: reading CfC header: %w", hr.err)
	}
	if inDim < 1 || units < 1 {
		return nil, fmt.Errorf("nn: CfC header has invalid dims in=%d units=%d", inDim, units)
	}
	// Checked before the blob is even parsed, the same defense as LoadLTC:
	// the CfC constructor bills the same O(units^2) parameter matrices, so a
	// hostile header is refused before any parsing or construction work
	// (see maxUnits/maxInDim, red team sweep F1).
	if units > maxUnits {
		return nil, fmt.Errorf("nn: CfC header has units=%d, exceeding the load limit %d", units, maxUnits)
	}
	if inDim > maxInDim {
		return nil, fmt.Errorf("nn: CfC header has inDim=%d, exceeding the load limit %d", inDim, maxInDim)
	}
	ts, err := serialize.ReadTensors(r)
	if err != nil {
		return nil, err
	}
	if len(ts) != cfcTensorCount {
		return nil, fmt.Errorf("nn: CfC stream holds %d tensors, want %d", len(ts), cfcTensorCount)
	}
	wiring, err := wiringFromStream(ts[0], ts[1], inDim, units)
	if err != nil {
		return nil, err
	}
	if err := checkReversals("erev", ts[cfcTensorCount-2]); err != nil {
		return nil, err
	}
	if err := checkReversals("sErev", ts[cfcTensorCount-1]); err != nil {
		return nil, err
	}
	cell := NewCfC(inDim, units, wiring, throwawayRNG())

	dsts := make([]*tensor.Tensor, 0, 15)
	for _, p := range cell.Parameters() {
		dsts = append(dsts, p.Data)
	}
	dsts = append(dsts, cell.erev, cell.sErev)
	if err := copyFields(ts[2:], dsts); err != nil {
		return nil, err
	}
	// The numerator coefficients are row views sharing the erev/sErev Data
	// arrays that copyFields just overwrote with the streamed polarities, so
	// the contraction picks them up with no rebuild (same as LoadLTC).
	return cell, nil
}

// SaveLinear writes l to w. The header carries only the kind byte; the
// layer dimensions live in W's shape.
//
// Errors: any I/O failure writing the kind byte or the tensor blob is
// returned, along with the serialize writer's validation errors (see
// serialize.WriteTensors). It never panics on I/O or stream concerns;
// l must be a live layer.
func SaveLinear(w io.Writer, l *Linear) error {
	hw := &headerWriter{w: w}
	hw.u8(kindLinear)
	if hw.err != nil {
		return fmt.Errorf("nn: writing Linear header: %w", hw.err)
	}
	return serialize.WriteTensors(w, []*tensor.Tensor{l.W.Data, l.B.Data})
}

// LoadLinear reads a stream written by SaveLinear and returns a layer
// owning fresh tensors (nothing aliases the reader's bytes).
//
// Errors (never panics — the stream is untrusted input): a wrong kind
// byte, every serialize.ReadTensors failure (bad magic, unknown
// version, truncation as io.ErrUnexpectedEOF, hostile size claims, and
// a v2 checksum mismatch), a tensor count other than 2, a weight that is
// not 2D, and a bias shape that does not match the weight's column
// count. Unlike the cell loaders it has no header dims to bound: the
// layer's size is whatever W's streamed shape says, subject to
// serialize's per-tensor limits.
func LoadLinear(r io.Reader) (*Linear, error) {
	hr := &headerReader{r: r}
	if err := readKind(hr, kindLinear); err != nil {
		return nil, err
	}
	ts, err := serialize.ReadTensors(r)
	if err != nil {
		return nil, err
	}
	if len(ts) != linearTensorCount {
		return nil, fmt.Errorf("nn: Linear stream holds %d tensors, want %d", len(ts), linearTensorCount)
	}
	wT, bT := ts[0], ts[1]
	if len(wT.Shape) != 2 {
		return nil, fmt.Errorf("nn: Linear weight must be 2D, stream has shape %v", wT.Shape)
	}
	if !shapeIs(bT.Shape, wT.Shape[1]) {
		return nil, fmt.Errorf("nn: Linear bias shape %v does not match weight columns %d", bT.Shape, wT.Shape[1])
	}
	return &Linear{W: autograd.Var(wT), B: autograd.Var(bT)}, nil
}
