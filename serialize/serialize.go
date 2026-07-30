// Package serialize persists tensors and model parameters in a compact,
// versioned binary format. It is the storage layer behind nn's Save/Load
// functions, exposed separately so the stream format can be audited on its
// own.
//
// # Wire format
//
// All integers are little-endian (encoding/binary.LittleEndian):
//
//	magic   [4]byte   "LNNS"
//	version uint8     1
//	count   uint32    number of tensors
//	repeated count times:
//	  rank  uint8
//	  shape [rank]int64   (each dimension >= 0)
//	  data  [size]float32 (IEEE-754, size = product of shape)
//
// A stream encodes exactly count tensors: trailing bytes after the last
// tensor's data are rejected as corruption.
//
// # Error contract (the exception domain)
//
// The rest of lnn reports misuse by panicking: its inputs come from the
// program itself, so a bad shape is a bug in the caller. Serialization is the
// deliberate exception. A load path consumes bytes from outside the program
// — files, networks, checkpoints from other versions — which may be corrupt,
// truncated, or outright hostile. Every failure on the read path is therefore
// returned as an error, never a panic, and a hostile stream must never drive
// an unbounded allocation: claimed ranks, dimensions and counts are validated
// against fixed limits and the element count is checked with overflow-safe
// multiplication (math/bits.Mul64, the same discipline as tensor.Size) BEFORE
// any buffer is allocated. A stream claiming a 1<<62-wide dimension is
// rejected with an error, not serviced with a petabyte-sized make(). When the
// reader exposes its remaining length (bytes.Buffer, bytes.Reader, ...) the
// claimed payload is additionally checked against it, so an oversized claim
// is rejected even when it fits the global limit.
//
// The write path, by contrast, handles in-memory tensors the caller owns;
// still, it returns errors rather than panicking, so a Save loop can report
// I/O failures uniformly.
package serialize

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"math/bits"

	"lnn/autograd"
	"lnn/tensor"
)

// Version is the format version this build writes and reads. Bump it (and
// teach ReadTensors the old layout) if the wire format ever changes.
const Version uint8 = 1

var magic = [4]byte{'L', 'N', 'N', 'S'}

// Read-path limits. They exist to turn hostile size claims into errors before
// an allocation happens; no sane model comes anywhere near them. maxElems
// caps a single tensor at 2^30 float32s (4 GiB of payload), maxCount caps a
// stream at 2^20 tensors, and maxRank caps the shape at 8 axes (the library's
// ops are 1D/2D-focused; storage supports more, but not this many).
const (
	maxElems uint64 = 1 << 30
	maxCount uint64 = 1 << 20
	maxRank         = 8
)

// chunk is the number of float32s encoded/decoded per scratch-buffer pass,
// bounding the transient memory of large tensor payloads to 4*chunk bytes.
const chunk = 4096

// writer accumulates I/O errors so the encode path reads top-to-bottom and
// reports once, mirroring the bufio.Scanner style. It encodes by hand with
// fixed scratch buffers: no binary.Write (which reflects over its argument),
// keeping the library reflection-free.
type writer struct {
	w   io.Writer
	err error
}

func (bw *writer) write(b []byte) {
	if bw.err != nil {
		return
	}
	_, bw.err = bw.w.Write(b)
}

func (bw *writer) u8(v uint8) {
	var b [1]byte
	b[0] = v
	bw.write(b[:])
}

func (bw *writer) u32(v uint32) {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	bw.write(b[:])
}

func (bw *writer) i64(v int64) {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], uint64(v))
	bw.write(b[:])
}

// reader counts the bytes it consumes so that, when the underlying reader
// knows its total length (the Len() method of bytes.Buffer, bytes.Reader and
// strings.Reader), every claimed payload can be checked against the bytes
// actually remaining before a buffer of that size is allocated.
type reader struct {
	r        io.Reader
	total    int64 // stream length, or -1 if the reader does not report one
	consumed int64
}

func newReader(r io.Reader) *reader {
	rd := &reader{r: r, total: -1}
	if l, ok := r.(interface{ Len() int }); ok {
		rd.total = int64(l.Len())
	}
	return rd
}

// remaining returns the unread byte count, or -1 when unknown.
func (rd *reader) remaining() int64 {
	if rd.total < 0 {
		return -1
	}
	return rd.total - rd.consumed
}

// full reads exactly len(b) bytes, normalizing both EOF flavors into
// io.ErrUnexpectedEOF (a stream that ends mid-structure is truncated,
// whatever the underlying reader reports about it).
func (rd *reader) full(b []byte) error {
	n, err := io.ReadFull(rd.r, b)
	rd.consumed += int64(n)
	if err == nil {
		return nil
	}
	if err == io.EOF || err == io.ErrUnexpectedEOF {
		return fmt.Errorf("truncated stream: %w", io.ErrUnexpectedEOF)
	}
	return err
}

// WriteTensors writes ts to w in the package's wire format. It fails with a
// descriptive error on I/O failure, on a nil tensor, or on an in-memory
// tensor whose Shape and Data disagree (checked with the same overflow-safe
// multiplication as tensor.Size, but returning an error instead of panicking).
func WriteTensors(w io.Writer, ts []*tensor.Tensor) error {
	if uint64(len(ts)) > maxCount {
		return fmt.Errorf("serialize: %d tensors exceed the stream count limit %d", len(ts), maxCount)
	}
	bw := &writer{w: w}
	bw.write(magic[:])
	bw.u8(Version)
	bw.u32(uint32(len(ts)))
	for i, t := range ts {
		if t == nil {
			return fmt.Errorf("serialize: tensor %d is nil", i)
		}
		if err := bw.tensor(t, i); err != nil {
			return err
		}
	}
	if bw.err != nil {
		return fmt.Errorf("serialize: writing stream: %w", bw.err)
	}
	return nil
}

// tensor encodes one tensor, rank then shape then float32 payload, after
// validating that the shape is representable and consistent with Data.
func (bw *writer) tensor(t *tensor.Tensor, idx int) error {
	rank := len(t.Shape)
	if rank > maxRank {
		return fmt.Errorf("serialize: tensor %d: rank %d exceeds the maximum rank %d", idx, rank, maxRank)
	}
	var n uint64 = 1
	for _, dim := range t.Shape {
		if dim < 0 {
			return fmt.Errorf("serialize: tensor %d: negative dimension %d in shape %v", idx, dim, t.Shape)
		}
		hi, lo := bits.Mul64(n, uint64(dim))
		if hi != 0 || lo > math.MaxInt64 {
			return fmt.Errorf("serialize: tensor %d: shape %v overflows the element count", idx, t.Shape)
		}
		n = lo
	}
	if n != uint64(len(t.Data)) {
		return fmt.Errorf("serialize: tensor %d: shape %v implies %d elements but Data holds %d",
			idx, t.Shape, n, len(t.Data))
	}
	bw.u8(uint8(rank))
	for _, dim := range t.Shape {
		bw.i64(int64(dim))
	}
	var enc [chunk * 4]byte
	for data := t.Data; len(data) > 0 && bw.err == nil; {
		k := len(data)
		if k > chunk {
			k = chunk
		}
		for i, f := range data[:k] {
			binary.LittleEndian.PutUint32(enc[i*4:], math.Float32bits(f))
		}
		bw.write(enc[:4*k])
		data = data[k:]
	}
	return bw.err
}

// ReadTensors reads a stream written by WriteTensors and returns its tensors.
// Corrupt, truncated, unknown-version or hostile streams fail with a
// descriptive error; nothing in the stream can trigger an allocation beyond
// the documented limits (see the package doc).
func ReadTensors(r io.Reader) ([]*tensor.Tensor, error) {
	rd := newReader(r)
	var m [4]byte
	if err := rd.full(m[:]); err != nil {
		return nil, fmt.Errorf("serialize: reading stream magic: %w", err)
	}
	if m != magic {
		return nil, fmt.Errorf("serialize: bad magic % x, want % x (\"LNNS\"): not an lnn tensor stream", m[:], magic[:])
	}
	var vb [1]byte
	if err := rd.full(vb[:]); err != nil {
		return nil, fmt.Errorf("serialize: reading format version: %w", err)
	}
	if vb[0] != Version {
		return nil, fmt.Errorf("serialize: unsupported format version %d (this build reads version %d)", vb[0], Version)
	}
	var cb [4]byte
	if err := rd.full(cb[:]); err != nil {
		return nil, fmt.Errorf("serialize: reading tensor count: %w", err)
	}
	count := uint64(binary.LittleEndian.Uint32(cb[:]))
	if count > maxCount {
		return nil, fmt.Errorf("serialize: stream claims %d tensors, exceeding the count limit %d", count, maxCount)
	}
	ts := make([]*tensor.Tensor, count)
	for i := range ts {
		t, err := readTensor(rd, i)
		if err != nil {
			return nil, err
		}
		ts[i] = t
	}
	// The format is self-framing (an explicit count), so anything after the
	// last tensor's payload is corruption rather than "another message".
	var probe [1]byte
	if err := rd.full(probe[:]); err == nil {
		return nil, fmt.Errorf("serialize: unexpected trailing byte(s) after tensor %d", count-1)
	} else if !errors.Is(err, io.ErrUnexpectedEOF) {
		return nil, fmt.Errorf("serialize: checking for trailing data: %w", err)
	}
	return ts, nil
}

// readTensor decodes tensor idx, validating every claim before allocating:
// rank against maxRank, dimensions against negativity, and the total element
// count against both overflow (bits.Mul64) and maxElems. When the stream
// length is known, the payload claim is finally checked against the bytes
// actually left, so an oversized claim errors out before make() runs.
func readTensor(rd *reader, idx int) (*tensor.Tensor, error) {
	var rb [1]byte
	if err := rd.full(rb[:]); err != nil {
		return nil, fmt.Errorf("serialize: tensor %d: reading rank: %w", idx, err)
	}
	rank := int(rb[0])
	if rank > maxRank {
		return nil, fmt.Errorf("serialize: tensor %d: rank %d exceeds the maximum rank %d", idx, rank, maxRank)
	}
	shape := make([]int, rank)
	var n uint64 = 1
	for d := 0; d < rank; d++ {
		var db [8]byte
		if err := rd.full(db[:]); err != nil {
			return nil, fmt.Errorf("serialize: tensor %d: reading shape: %w", idx, err)
		}
		dim := int64(binary.LittleEndian.Uint64(db[:]))
		if dim < 0 {
			return nil, fmt.Errorf("serialize: tensor %d: negative dimension %d at axis %d", idx, dim, d)
		}
		hi, lo := bits.Mul64(n, uint64(dim))
		if hi != 0 {
			return nil, fmt.Errorf("serialize: tensor %d: shape prefix %v times dimension %d overflows the element count",
				idx, shape[:d], dim)
		}
		if lo > maxElems {
			return nil, fmt.Errorf("serialize: tensor %d: shape claims %d elements, exceeding the limit %d", idx, lo, maxElems)
		}
		n = lo
		shape[d] = int(dim)
	}
	if rem := rd.remaining(); rem >= 0 && n*4 > uint64(rem) {
		return nil, fmt.Errorf("serialize: tensor %d: truncated stream: claims %d data bytes but only %d remain: %w",
			idx, n*4, rem, io.ErrUnexpectedEOF)
	}
	data := make([]float32, n)
	var enc [chunk * 4]byte
	for done := uint64(0); done < n; {
		k := n - done
		if k > chunk {
			k = chunk
		}
		buf := enc[:4*k]
		if err := rd.full(buf); err != nil {
			return nil, fmt.Errorf("serialize: tensor %d: reading data: %w", idx, err)
		}
		for i := uint64(0); i < k; i++ {
			data[done+i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[4*i:]))
		}
		done += k
	}
	return &tensor.Tensor{Shape: shape, Data: data}, nil
}

// WriteParameters writes the Data tensors of params (in order) to w. It is
// WriteTensors over p.Data with nil guards, named for the checkpoint use case.
func WriteParameters(w io.Writer, params []*autograd.Variable) error {
	ts := make([]*tensor.Tensor, len(params))
	for i, p := range params {
		if p == nil || p.Data == nil {
			return fmt.Errorf("serialize: parameter %d has no data", i)
		}
		ts[i] = p.Data
	}
	return WriteTensors(w, ts)
}

// LoadParameters reads a stream written by WriteParameters and copies the
// values back into params IN PLACE: the *autograd.Variable pointers keep
// their identity, so every graph edge that references them stays valid and
// graph ownership is preserved. A count mismatch or any shape mismatch is an
// error; all shapes are validated before anything is copied, so a failing
// load leaves every parameter exactly as it was.
func LoadParameters(r io.Reader, params []*autograd.Variable) error {
	for i, p := range params {
		if p == nil || p.Data == nil {
			return fmt.Errorf("serialize: parameter %d has no data", i)
		}
	}
	ts, err := ReadTensors(r)
	if err != nil {
		return err
	}
	if len(ts) != len(params) {
		return fmt.Errorf("serialize: parameter count mismatch: stream holds %d tensors, destination has %d variables",
			len(ts), len(params))
	}
	for i, t := range ts {
		if !tensor.SameShape(t, params[i].Data) {
			return fmt.Errorf("serialize: parameter %d shape mismatch: stream has %v, destination has %v",
				i, t.Shape, params[i].Data.Shape)
		}
	}
	for i, t := range ts {
		copy(params[i].Data.Data, t.Data)
	}
	return nil
}
