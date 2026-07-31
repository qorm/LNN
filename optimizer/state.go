package optimizer

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/serialize"
	"github.com/qorm/LNN/tensor"
)

// This file adds optimizer-state persistence on top of the serialize
// package's versioned tensor stream, closing the checkpoint gap: nn's
// Save/Load functions and serialize.WriteParameters restore model
// parameters, but Momentum's velocity buffers and Adam's moment estimates,
// update counts and bias-correction powers lived only in memory, so
// resuming training from a checkpoint silently reset them — for Adam, that
// throws away the learning-rate adaptation of every prior step (its bias
// correction restarts as if t were 0). SaveState/LoadState make a resumed
// trajectory bit-identical to an uninterrupted one.
//
// # Wire format
//
// All integers are little-endian (encoding/binary.LittleEndian); floats are
// IEEE-754 float32, little-endian:
//
//	magic    [4]byte  "LNO1"
//	version  uint8    1
//	kind     uint8    0 = SGD, 1 = Momentum, 2 = Adam
//	count    uint32   number of parameter records (one per parameter, in
//	                 the order of the params slice given to SaveState)
//	repeated count times (the record section; empty for SGD):
//	  present uint8   0 = no state for this parameter, 1 = state follows
//	  Adam only, when present:
//	    t    uint32   update count
//	    pow1 float32  Beta1^t, saved bit for bit
//	    pow2 float32  Beta2^t, saved bit for bit
//	blob     tensors  serialize.WriteTensors over the state buffers, in
//	                 parameter order: one velocity tensor per present
//	                 Momentum parameter, m then v per present Adam
//	                 parameter, and an (empty) stream of zero tensors for
//	                 SGD — so every state stream, whatever its kind, ends
//	                 in one counted tensor blob that is self-framing and
//	                 rejects trailing bytes
//
// The record section + tensor blob split mirrors nn's model streams (a
// header plus one serialize blob): every size claim lives inside the blob,
// where serialize's audited limits and allocation discipline validate it
// before any buffer is allocated — this file adds no tensor format of its
// own. Each state tensor carries the shape of its parameter's Data, so
// LoadState validates it dimension by dimension.
//
// Hyperparameters (LR, Mu, Beta1/Beta2/Eps) are deliberately NOT in the
// stream: they are exported fields the caller sets on the destination
// optimizer, exactly as at construction. For Adam the saved pow1/pow2 are
// checked bit for bit against this optimizer's Beta1^t/Beta2^t, so loading
// a stream saved under different betas fails as corruption rather than
// silently resuming with inconsistent bias correction.
//
// # Error contract
//
// The load path treats its input as an untrusted byte stream, with the
// same discipline as the serialize package: every failure — bad magic,
// unknown version, kind mismatch, count mismatch, presence flag outside
// {0, 1}, shape mismatch, inconsistent Adam counters, truncation, trailing
// bytes — is returned as an error, never a panic. All records and tensors
// are parsed and validated BEFORE any state is applied (the
// validate-all-then-apply order of nn/save.go's copyFields), so a failing
// LoadState leaves the destination optimizer exactly as it was. Truncated
// streams surface io.ErrUnexpectedEOF, transparently from serialize's blob
// reader where the truncation sits inside the blob.
//
// # Load semantics
//
// LoadState overwrites the destination optimizer's state for exactly the
// parameters given: a present record replaces any existing buffer (bit for
// bit), an absent record deletes the parameter's entry. Stale keys for
// variables NOT in params are deliberately left in place — after a load
// the optimizer's state is precisely what the stream describes for params,
// plus whatever the optimizer already knew about other variables. This
// mirrors serialize.LoadParameters' stale-Grad contract: honest disclosure
// rather than silent cleanup. Construct a fresh optimizer and load into it
// for a state that is exactly the stream and nothing else.
//
// State is keyed by the index into the given params slice: pointers do not
// survive across processes, so LoadState attaches the i-th record to
// params[i]. The same parameter order must be given to Save and Load.

// stateMagic identifies an optimizer state stream. It is distinct from
// serialize's "LNNS" so the two stream types can never be confused: the
// state stream embeds tensor blobs, it is not one.
var stateMagic = [4]byte{'L', 'N', 'O', '1'}

// stateVersion is the state format version this build writes and reads.
const stateVersion uint8 = 1

// Optimizer kind tags, written as the kind byte of every state stream.
const (
	kindSGD uint8 = iota
	kindMomentum
	kindAdam
)

// maxT caps the Adam update count LoadState accepts from a stream. Value:
// 2^24 (16,777,216 updates). The pow consistency check recomputes Beta^t
// as t sequential float32 multiplications — the exact rounding path
// Adam.Step maintains — so the streamed t is also a CPU claim on the load
// path: a 13-byte record claiming t = 2^32-1 must not bill four billion
// multiplications. The cap is deliberately load-only (the same asymmetry
// as serialize's maxUnfolds and nn's maxUnits): Adam.Step's runtime
// contract is unchanged, because a step count there is the caller's own
// training history, while a load's input is an untrusted byte stream that
// gets no vote on its own resource budget. 2^24 steps is far beyond any
// realistic run in this library — each step is a full graph backward.
const maxT = 1 << 24

// stateWriter encodes the stream header and record section by hand with a
// fixed scratch discipline — no binary.Write (which reflects), mirroring
// serialize's writer: accumulate the first I/O error, report once.
type stateWriter struct {
	w   io.Writer
	err error
}

func (sw *stateWriter) write(b []byte) {
	if sw.err != nil {
		return
	}
	_, sw.err = sw.w.Write(b)
}

func (sw *stateWriter) u8(v uint8) { sw.write([]byte{v}) }

func (sw *stateWriter) u32(v uint32) {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	sw.write(b[:])
}

func (sw *stateWriter) f32(v float32) { sw.u32(math.Float32bits(v)) }

// stateReader decodes the header and record section, normalizing both EOF
// flavors into io.ErrUnexpectedEOF: a stream that ends mid-structure is
// truncated, whatever the underlying reader reports about it (the same
// normalization as serialize and nn/save.go's header readers).
type stateReader struct {
	r   io.Reader
	err error
}

func (sr *stateReader) read(b []byte) {
	if sr.err != nil {
		return
	}
	if _, err := io.ReadFull(sr.r, b); err != nil {
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			sr.err = fmt.Errorf("truncated stream: %w", io.ErrUnexpectedEOF)
		} else {
			sr.err = err
		}
	}
}

func (sr *stateReader) u8() uint8 {
	var b [1]byte
	sr.read(b[:])
	return b[0]
}

func (sr *stateReader) u32() uint32 {
	var b [4]byte
	sr.read(b[:])
	return binary.LittleEndian.Uint32(b[:])
}

func (sr *stateReader) f32() float32 { return math.Float32frombits(sr.u32()) }

// stateKindOf returns the stream kind tag of a supported optimizer.
func stateKindOf(o Optimizer) (uint8, bool) {
	switch o.(type) {
	case *SGD:
		return kindSGD, true
	case *Momentum:
		return kindMomentum, true
	case *Adam:
		return kindAdam, true
	}
	return 0, false
}

func stateKindName(k uint8) string {
	switch k {
	case kindSGD:
		return "SGD"
	case kindMomentum:
		return "Momentum"
	case kindAdam:
		return "Adam"
	}
	return "unknown"
}

// betaPow returns beta^t computed as t sequential float32 multiplications
// from 1 — the exact rounding path Adam.Step maintains in its pow1/pow2
// fields, so the result agrees with the optimizer's running power bit for
// bit. The early zero exit is exact but narrow: it only engages for betas
// whose running product reaches a true zero (0.5, say: 0.5^t = 2^-t
// underflows to exactly 0 at t = 150, and once zero it stays zero, so no
// later rounding can occur). Other betas settle into a subnormal FIXED
// POINT instead: 0.9 pins at 0x00000004 from t ~= 965 (4 ULP * 0.9 = 3.6
// ULP rounds back to 4 ULP) and 0.999 pins at 0x000001f4 from
// t ~= 96,900, so for them the exit never fires and the loop runs all t
// multiplications — bounded by maxT on the load path — arriving at the
// same bits Adam.Step's running product pins at. The behavior is correct
// either way, and since the stream stores pow1/pow2 bit for bit, betaPow
// is only ever a load-time consistency check (it also catches a beta
// mismatch); the stored power is what gets restored.
func betaPow(beta float32, t int) float32 {
	p := float32(1)
	for i := 0; i < t && p != 0; i++ {
		p *= beta
	}
	return p
}

// countToUint32 bounds a parameter count to the stream format's uint32 count
// field. The sole caller passes len(params), which cannot be negative and
// cannot reach 2^32 under any realistic allocation; the guard is extracted
// here so its rejection semantics — the exact boundary, the error wording,
// and the treatment of negatives — are pinnable by direct unit tests, since
// a genuinely oversized params slice (2^32 pointers, 32 GiB) can never exist
// in a test. Negative inputs (unreachable from len() but expressible on the
// pure function) wrap to >= 2^63 under the uint64 conversion and are
// rejected by the same comparison.
func countToUint32(n int) (uint32, error) {
	if uint64(n) > math.MaxUint32 {
		return 0, fmt.Errorf("%d parameters exceed the stream count limit %d", n, uint32(math.MaxUint32))
	}
	return uint32(n), nil
}

// checkStateParams guards the parameter slice both paths index into.
func checkStateParams(op string, params []*autograd.Variable) error {
	for i, p := range params {
		if p == nil || p.Data == nil {
			return fmt.Errorf("optimizer: %s: parameter %d has no data", op, i)
		}
	}
	return nil
}

// SaveState writes o's per-parameter state for params to w, keyed by index
// into the slice (pointers do not survive across processes; LoadState
// attaches the i-th record to params[i], so the same order must be given
// to both). The stream carries only per-parameter state buffers — velocity
// for Momentum; m, v, the update count t and the bias-correction powers
// pow1/pow2 for Adam, saved bit for bit so a resumed run continues with
// exactly the adaptation the interrupted run had accumulated.
// Hyperparameters are not saved: set them on the destination optimizer
// before LoadState. For SGD the save is an identity (the record section is
// empty), but the stream is still written — magic, kind, count and an
// empty tensor blob — so every kind round-trips through one uniform,
// self-framing format.
//
// SaveState fails with a descriptive error on I/O failure, on a nil
// parameter or one without Data, on an unsupported optimizer type
// (anything but *SGD, *Momentum or *Adam), and on an Adam update count
// that does not fit the format's uint32 field. It never panics and never
// mutates o or params. Writing the same state twice yields byte-identical
// streams.
func SaveState(w io.Writer, o Optimizer, params []*autograd.Variable) error {
	kind, ok := stateKindOf(o)
	if !ok {
		return fmt.Errorf("optimizer: SaveState: unsupported optimizer type %T: state persistence covers SGD, Momentum and Adam", o)
	}
	if err := checkStateParams("SaveState", params); err != nil {
		return err
	}
	count, err := countToUint32(len(params))
	if err != nil {
		return fmt.Errorf("optimizer: SaveState: %w", err)
	}
	sw := &stateWriter{w: w}
	sw.write(stateMagic[:])
	sw.u8(stateVersion)
	sw.u8(kind)
	sw.u32(count)

	ts := make([]*tensor.Tensor, 0, len(params))
	switch o := o.(type) {
	case *SGD:
		// Stateless: the record section is empty and the blob carries
		// zero tensors. The header still names the kind and the parameter
		// count, so the stream stays self-describing and kind-checkable.
	case *Momentum:
		for _, p := range params {
			if v := o.velocity[p]; v != nil {
				sw.u8(1)
				ts = append(ts, &tensor.Tensor{Shape: p.Data.Shape, Data: v})
			} else {
				sw.u8(0) // absent in the map: recorded as absent, not fabricated as zeros
			}
		}
	case *Adam:
		for i, p := range params {
			st := o.state[p]
			if st == nil {
				sw.u8(0)
				continue
			}
			if st.t < 0 || uint64(st.t) > math.MaxUint32 {
				return fmt.Errorf("optimizer: SaveState: parameter %d: update count %d does not fit the format's uint32 field", i, st.t)
			}
			sw.u8(1)
			sw.u32(uint32(st.t))
			sw.f32(st.pow1)
			sw.f32(st.pow2)
			ts = append(ts,
				&tensor.Tensor{Shape: p.Data.Shape, Data: st.m},
				&tensor.Tensor{Shape: p.Data.Shape, Data: st.v})
		}
	}
	if sw.err != nil {
		return fmt.Errorf("optimizer: writing state header: %w", sw.err)
	}
	// WriteTensors validates each buffer's shape against its Data before
	// encoding (so a velocity resized behind the optimizer's back is a
	// valued error, not a corrupt stream) and frames the whole blob with
	// the serialize magic, version and count.
	return serialize.WriteTensors(w, ts)
}

// LoadState restores into o the per-parameter state written by SaveState,
// attaching the stream's i-th record to params[i]. For a stream saved
// under the same hyperparameters and a matching parameter order, the
// optimizer afterwards steps bit-identically to the one that was saved —
// a resumed training trajectory equals the uninterrupted one element for
// element (see the resume tests). Any corruption, version skew, kind or
// count mismatch, shape mismatch, inconsistent Adam counter, truncation
// or trailing byte is an error; all of the stream is validated before any
// state is applied, so a failing load leaves o exactly as it was.
//
// The load overwrites o's state for the given parameters — a present
// record replaces any existing buffer, an absent record deletes the
// entry — and deliberately leaves entries for variables NOT in params
// untouched: after the load, o's state is exactly what the stream
// describes for params plus any stale keys it already held (the same
// honest-disclosure contract as serialize.LoadParameters' stale Grad).
// Construct a fresh optimizer to load into for a clean state.
//
// The destination optimizer supplies the hyperparameters; for Adam the
// streamed pow1/pow2 must equal this optimizer's Beta1^t/Beta2^t bit for
// bit, so a stream saved under different betas fails as corruption rather
// than resuming with inconsistent bias correction. An Adam update count
// above maxT (2^24) is rejected before the blob is even parsed — a
// load-only limit bounding the pow recomputation cost, with Adam.Step's
// runtime contract unchanged. SGD carries no state; loading an SGD
// stream validates the stream and changes nothing.
//
// It never panics on stream contents (the stream is untrusted input,
// the documented exception domain): truncation surfaces as
// io.ErrUnexpectedEOF (transparently from the blob reader when it sits
// inside the blob), and every other failure is a descriptive error.
// o must be a supported optimizer and params must carry Data, as for
// SaveState.
func LoadState(r io.Reader, o Optimizer, params []*autograd.Variable) error {
	kind, ok := stateKindOf(o)
	if !ok {
		return fmt.Errorf("optimizer: LoadState: unsupported optimizer type %T: state persistence covers SGD, Momentum and Adam", o)
	}
	if err := checkStateParams("LoadState", params); err != nil {
		return err
	}
	sr := &stateReader{r: r}
	var m [4]byte
	sr.read(m[:])
	if sr.err != nil {
		return fmt.Errorf("optimizer: reading state header: %w", sr.err)
	}
	if m != stateMagic {
		return fmt.Errorf("optimizer: bad magic % x, want % x (\"LNO1\"): not an optimizer state stream", m[:], stateMagic[:])
	}
	version := sr.u8()
	streamKind := sr.u8()
	count := sr.u32()
	if sr.err != nil {
		return fmt.Errorf("optimizer: reading state header: %w", sr.err)
	}
	// Rejected, never guessed — the same rule and message shape as the
	// serialize package's version gate.
	if version != stateVersion {
		if version > stateVersion {
			return fmt.Errorf("optimizer: unsupported state format version %d (this build reads version %d): the stream was written by a newer version of the library; update this build to read it", version, stateVersion)
		}
		return fmt.Errorf("optimizer: unsupported state format version %d (this build reads version %d): no earlier layout exists, the stream is corrupt or forged", version, stateVersion)
	}
	if streamKind != kind {
		return fmt.Errorf("optimizer: state stream kind %d (%s) does not match optimizer kind %d (%s)",
			streamKind, stateKindName(streamKind), kind, stateKindName(kind))
	}
	if count != uint32(len(params)) {
		return fmt.Errorf("optimizer: state stream holds %d parameter records, destination has %d parameters", count, len(params))
	}

	// Parse the record section into staging; the optimizer is not touched
	// until every record, tensor and counter below has validated.
	type stateRecord struct {
		present bool
		t       int     // Adam: update count
		pow1    float32 // Adam: Beta1^t, as saved
		pow2    float32 // Adam: Beta2^t, as saved
	}
	recs := make([]stateRecord, count)
	presentCount := 0
	// SGD's record section is empty by definition: the header's count
	// still names the parameters the stream was taken over, but no record
	// bytes follow it (and reading any would consume the blob instead).
	if kind != kindSGD {
		for i := range recs {
			present := sr.u8()
			if sr.err != nil {
				return fmt.Errorf("optimizer: parameter %d: reading presence flag: %w", i, sr.err)
			}
			switch present {
			case 0:
			case 1:
				recs[i].present = true
				presentCount++
			default:
				return fmt.Errorf("optimizer: parameter %d: presence flag %d outside {0, 1}: the stream is corrupt", i, present)
			}
			if !recs[i].present || kind != kindAdam {
				continue
			}
			t := sr.u32()
			pow1 := sr.f32()
			pow2 := sr.f32()
			if sr.err != nil {
				return fmt.Errorf("optimizer: parameter %d: reading Adam counters: %w", i, sr.err)
			}
			// Checked before the blob is even parsed, on the same
			// discipline as serialize's size limits: t bounds the pow
			// recomputation in the validation phase below.
			if uint64(t) > maxT {
				return fmt.Errorf("optimizer: parameter %d: update count %d exceeds the load limit %d", i, t, maxT)
			}
			recs[i].t = int(t)
			recs[i].pow1 = pow1
			recs[i].pow2 = pow2
		}
	}

	ts, err := serialize.ReadTensors(r)
	if err != nil {
		return err
	}
	tensorsPerPresent := 0
	switch kind {
	case kindMomentum:
		tensorsPerPresent = 1 // velocity
	case kindAdam:
		tensorsPerPresent = 2 // m, v
	}
	if want := presentCount * tensorsPerPresent; len(ts) != want {
		return fmt.Errorf("optimizer: state blob holds %d tensors, want %d (%d present parameters, %d tensors each)",
			len(ts), want, presentCount, tensorsPerPresent)
	}

	// Validate every shape and counter before applying anything, so a
	// stream that mismatches late leaves the optimizer exactly as it was.
	adam, _ := o.(*Adam)
	k := 0
	for i := range recs {
		if !recs[i].present {
			continue
		}
		shape := params[i].Data.Shape
		switch kind {
		case kindMomentum:
			if !tensor.SameShape(ts[k], params[i].Data) {
				return fmt.Errorf("optimizer: parameter %d: velocity shape mismatch: stream has %v, parameter has %v", i, ts[k].Shape, shape)
			}
		case kindAdam:
			if !tensor.SameShape(ts[k], params[i].Data) {
				return fmt.Errorf("optimizer: parameter %d: moment m shape mismatch: stream has %v, parameter has %v", i, ts[k].Shape, shape)
			}
			if !tensor.SameShape(ts[k+1], params[i].Data) {
				return fmt.Errorf("optimizer: parameter %d: moment v shape mismatch: stream has %v, parameter has %v", i, ts[k+1].Shape, shape)
			}
			// Bit-for-bit by construction when the betas agree: betaPow
			// replays Step's sequential product. A mismatch means the
			// stream is corrupt or was saved under other hyperparameters.
			if want := betaPow(adam.Beta1, recs[i].t); math.Float32bits(recs[i].pow1) != math.Float32bits(want) {
				return fmt.Errorf("optimizer: parameter %d: stream pow1 is 0x%08x but Beta1^%d = 0x%08x for this optimizer's Beta1=%v: the stream is corrupt or was saved with different Adam hyperparameters",
					i, math.Float32bits(recs[i].pow1), recs[i].t, math.Float32bits(want), adam.Beta1)
			}
			if want := betaPow(adam.Beta2, recs[i].t); math.Float32bits(recs[i].pow2) != math.Float32bits(want) {
				return fmt.Errorf("optimizer: parameter %d: stream pow2 is 0x%08x but Beta2^%d = 0x%08x for this optimizer's Beta2=%v: the stream is corrupt or was saved with different Adam hyperparameters",
					i, math.Float32bits(recs[i].pow2), recs[i].t, math.Float32bits(want), adam.Beta2)
			}
		}
		k += tensorsPerPresent
	}

	// Apply. ReadTensors decodes each payload into a fresh slice owned by
	// nobody else, so the buffers are adopted without copying: nothing of
	// the caller's reader aliases into the optimizer's state.
	k = 0
	switch kind {
	case kindSGD:
		// Stateless: the validated stream changes nothing.
	case kindMomentum:
		mom := o.(*Momentum)
		if mom.velocity == nil {
			mom.velocity = make(map[*autograd.Variable][]float32, len(params))
		}
		for i := range recs {
			if recs[i].present {
				mom.velocity[params[i]] = ts[k].Data
				k++
			} else {
				delete(mom.velocity, params[i])
			}
		}
	case kindAdam:
		if adam.state == nil {
			adam.state = make(map[*autograd.Variable]*adamState, len(params))
		}
		for i := range recs {
			if recs[i].present {
				adam.state[params[i]] = &adamState{
					m:    ts[k].Data,
					v:    ts[k+1].Data,
					t:    recs[i].t,
					pow1: recs[i].pow1,
					pow2: recs[i].pow2,
				}
				k += 2
			} else {
				delete(adam.state, params[i])
			}
		}
	}
	return nil
}
