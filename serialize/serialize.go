// Package serialize persists tensors and model parameters in a compact,
// versioned binary format. It is the storage layer behind nn's Save/Load
// functions, exposed separately so the stream format can be audited on its
// own.
//
// # Wire format
//
// All integers are little-endian (encoding/binary.LittleEndian):
//
//	magic     [4]byte   "LNNS"
//	version   uint8     2 (v1 and v2 are both read; see "Format versioning")
//	count     uint32    number of tensors
//	repeated count times:
//	  rank  uint8
//	  shape [rank]int64   (each dimension >= 0)
//	  data  [size]float32 (IEEE-754, size = product of shape)
//	checksum  [4]byte   v2 only: CRC-32C (Castagnoli) of every byte above —
//	                    magic, version, count, and each tensor's rank, shape
//	                    and data
//
// A stream encodes exactly count tensors followed, in v2, by the 4-byte
// checksum: trailing bytes after the last tensor's data (v1) or after the
// checksum (v2) are rejected as corruption.
//
// # Integrity checksum (v2)
//
// v2 appends a CRC-32C (Castagnoli polynomial, the same code safetensors
// uses, hardware-accelerated on modern CPUs) computed over the entire stream.
// It is a damage-detection mechanism, not a cryptographic authenticator: any
// single-byte flip in a v2 stream almost certainly changes the recomputed
// value and the load is refused with a checksum-mismatch error, which makes
// accidental corruption — a flipped bit, a truncated file, bit rot — reliably
// observable. A party that can write the stream can always recompute the
// checksum, so it carries no security weight against a malicious writer; the
// resource-exhaustion defenses (the fixed limits and the
// validate-before-allocate discipline below) are what stand against hostile
// streams, not the checksum. It also does not localize where the damage is:
// the load path is all-or-nothing (validate everything, then apply), so a
// single whole-stream checksum is all the format needs — per-tensor checksums
// would localize the damage for no consumer, at the cost of 4 bytes per tensor
// and a second code path.
//
// The checksum is computed and verified incrementally: the writer hashes each
// byte as it writes it and appends the digest at the end; the reader hashes
// each byte as it parses it and compares against the stored value after the
// last tensor. Neither side ever buffers the whole stream, so the
// unknown-length reader discipline below is untouched. The checksum
// deliberately sits at the tail, AFTER every structural validation: hostile
// size claims (rank/shape/count above the limits) are still refused in the
// header, before any allocation, without waiting for the end of the stream —
// the checksum is a corruption detector, not the allocation gate.
//
// # Error contract (the exception domain)
//
// The rest of LNN reports misuse by panicking: its inputs come from the
// program itself, so a bad shape is a bug in the caller. Serialization is the
// deliberate exception. A load path consumes bytes from outside the program
// — files, networks, checkpoints from other versions — which may be corrupt,
// truncated, or outright hostile. Every failure on the read path is therefore
// returned as an error, never a panic, and a hostile stream can allocate only
// in proportion to the bytes it actually delivers:
//
//   - Claimed ranks, dimensions and counts are validated against fixed
//     limits BEFORE any buffer is allocated, and element counts are checked
//     with overflow-safe multiplication (math/bits.Mul64, the same
//     discipline as tensor.Size). The limits: maxElems = 1<<30 float32s per
//     tensor (4 GiB of payload), maxCount = 1<<20 tensors per stream and
//     maxRank = 8 axes. A stream claiming a 1<<62-wide dimension is rejected
//     with an error, not serviced with a petabyte-sized make(). In v2 a
//     checksum mismatch is reported as an error after the stream is parsed,
//     so corrupted data is refused too.
//   - When the reader reports its remaining length (the Len() method of
//     bytes.Buffer, bytes.Reader, strings.Reader), every payload claim is
//     additionally checked against those bytes, so an oversized or truncated
//     claim is validated before its buffer is allocated and the full payload
//     is allocated in a single make.
//   - Readers without a length (io.Pipe, net.Conn, gzip.Reader) cannot be
//     checked up front; for them, payload buffers start small (at most one
//     16 KiB chunk) and grow only as bytes arrive, so a stream claiming
//     1<<30 elements but stopping after its 18-byte header peaks at a few
//     chunks of memory and fails with io.ErrUnexpectedEOF instead of
//     front-loading a 4 GiB make.
//
// The write path, by contrast, handles in-memory tensors the caller owns;
// still, it returns errors rather than panicking, so a Save loop can report
// I/O failures uniformly.
//
// # Format versioning
//
// Two layouts exist. Version 1 — the layout documented above minus the
// checksum — is frozen, and the reader still understands it, so legacy
// checkpoints load unchanged. Version 2 adds the trailing CRC-32C checksum
// and is the only layout this build writes: every new stream carries
// integrity protection. The meaning, position and encoding of every byte
// this build writes will not change: same magic, same header fields, same
// tensor order inside the nn model blobs, same little-endian float32
// payloads. Two sanctioned paths exist for future formats:
//
//   - Append-only extension within v2: new data may be added only as extra
//     tensors at the tail of a stream — counted in the header like any other
//     tensor (bytes after the counted payloads remain corruption), never
//     interleaved with existing ones, and only where the consuming model
//     loader explicitly accepts both the old and the extended count. A
//     moved, retyped or re-encoded field is not an extension. The
//     model-level kind registry (nn/save.go: 0 = LTC, 1 = CfC, 2 = Linear)
//     grows the same way: new kinds are appended with fresh tags; existing
//     tags are never reused or redefined.
//   - A whole-format upgrade: bump Version (to 3) and teach ReadTensors the
//     v2 layout explicitly. One version byte governs the entire stream, so
//     mixed-version streams cannot exist.
//
// Unknown versions are rejected, never guessed: a version byte above Version
// fails with an error stating the stream was written by a newer version of
// the library — update this build to read it — and a byte below v1 as
// corrupt, since no layout older than v1 was ever released. Guessing a
// layout would silently mis-decode checkpoints, the exact failure mode this
// package's error contract exists to prevent.
//
// The freeze is regression-pinned by golden vectors: testdata/ holds the
// committed Save* byte streams of a documented LTC, CfC and Linear cell
// (fixed seeds and construction parameters, listed in golden_test.go)
// together with the exact Step outputs each loaded cell must reproduce.
// Two families coexist: golden_v1_* freezes the historical v1 layout byte
// for byte (proof the reader still decodes legacy checkpoints exactly), and
// golden_v2_* freezes the v2 layout this build's writer emits (rebuilt and
// compared by TestGoldenWriterStability). Neither family feeds the other.
// Four tests stand guard, graded by platform because the Go specification
// permits an implementation to fuse multiple floating-point operations into
// a single rounded operation (FMA contraction), so the same source can
// legitimately round differently across architectures. On the architecture
// the vectors were generated on (arm64) the guard is absolute: writer
// stability requires a same-seed rebuild to re-emit the golden bytes byte
// for byte, and load exactness requires the loaded cells to reproduce the
// recorded outputs bit for bit. On any other architecture the format
// skeleton — magic, version, tensor count, ranks and shapes — is still
// pinned byte for byte, while float32 payloads are pinned within a 16 ULP
// window (measured cross-architecture drift: 1 ULP on a Linear forward
// output, up to 6 ULPs on CfC Box-Muller construction parameters). On
// every platform,
// reader-class agreement requires bit-identical loads on known- and
// unknown-length readers — a same-binary self-check, so it stays strict
// everywhere. Any unintended change to the wire format or to cell semantics
// fails at least one of them; regenerating the files is a deliberate,
// reviewed act (go test ./serialize -write-golden), never a side effect.
package serialize

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash"
	"hash/crc32"
	"io"
	"math"
	"math/bits"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/tensor"
)

// Version is the format version this build writes and reads: 2, the v2
// layout with the trailing CRC-32C checksum. Version 1 (no checksum) is
// still read for legacy checkpoints; the reader learns old layouts
// explicitly, and a whole-format upgrade means bumping Version and teaching
// ReadTensors the previous layout, exactly as this build was taught v1.
const Version uint8 = 2

// magicV1 is the version byte of the historical v1 layout (no checksum).
const magicV1 uint8 = 1

var magic = [4]byte{'L', 'N', 'N', 'S'}

// crc32cTable is the Castagnoli polynomial table (CRC-32C) used for the v2
// stream checksum: the same algorithm safetensors uses, hardware-accelerated
// on modern CPUs. It is an integrity (damage-detection) checksum, not a
// cryptographic MAC — see "Integrity checksum (v2)" in the package doc.
var crc32cTable = crc32.MakeTable(crc32.Castagnoli)

// newCRC32C returns a fresh incremental CRC-32C hasher.
func newCRC32C() hash.Hash32 { return crc32.New(crc32cTable) }

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
	sum hash.Hash32 // non-nil for v2: every byte written through write() feeds it
}

func (bw *writer) write(b []byte) {
	if bw.err != nil {
		return
	}
	_, bw.err = bw.w.Write(b)
	if bw.err == nil && bw.sum != nil {
		bw.sum.Write(b)
	}
}

// checksum appends the 4-byte CRC-32C of everything written so far and
// reports any write failure of that final field. It must be called once, at
// the end of a v2 stream, after the last tensor payload and only after
// WriteTensors has confirmed the body wrote without error (bw.err is nil);
// the checksum bytes themselves are not part of the checksum.
func (bw *writer) checksum() error {
	var cb [4]byte
	binary.LittleEndian.PutUint32(cb[:], bw.sum.Sum32())
	bw.sum = nil // the checksum field is not covered by the checksum
	bw.write(cb[:])
	if bw.err != nil {
		return fmt.Errorf("serialize: writing stream checksum: %w", bw.err)
	}
	return nil
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
	sum      hash.Hash32 // non-nil for v2: every byte read through full() feeds it
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
		if rd.sum != nil {
			rd.sum.Write(b)
		}
		return nil
	}
	if err == io.EOF || err == io.ErrUnexpectedEOF {
		return fmt.Errorf("truncated stream: %w", io.ErrUnexpectedEOF)
	}
	return err
}

// floats reads a payload of n float32s, choosing its allocation strategy by
// what the underlying reader can promise (red team F1):
//
//   - Known remaining length: a claim that does not fit the bytes actually
//     left is truncation, reported BEFORE any buffer is allocated; a claim
//     that fits is serviced by a single full-size make, filled in
//     chunk-sized I/O passes. This is the fast path bytes.Buffer,
//     bytes.Reader and strings.Reader take.
//   - Unknown length (io.Pipe, net.Conn, gzip.Reader — readers without a
//     Len method): nothing can be proven up front, so the buffer starts
//     small (at most chunk elements, 16 KiB) and grows only as bytes
//     actually arrive. A stream that claims 1<<30 elements but stops after
//     its 18-byte header then peaks at a few chunks of memory and fails
//     with io.ErrUnexpectedEOF, instead of front-loading a 4 GiB make —
//     peak allocation stays proportional to the delivered bytes. A complete
//     stream still ends with all n elements in a single slice.
//
// n must have passed the limit and overflow checks already; n*4 is then at
// most 4 GiB and cannot overflow uint64.
func (rd *reader) floats(n uint64) ([]float32, error) {
	var enc [chunk * 4]byte
	if rem := rd.remaining(); rem >= 0 {
		if n*4 > uint64(rem) {
			return nil, fmt.Errorf("truncated stream: claims %d data bytes but only %d remain: %w",
				n*4, rem, io.ErrUnexpectedEOF)
		}
		data := make([]float32, n)
		for done := uint64(0); done < n; {
			k := n - done
			if k > chunk {
				k = chunk
			}
			buf := enc[:4*k]
			if err := rd.full(buf); err != nil {
				return nil, fmt.Errorf("reading data: %w", err)
			}
			for i := uint64(0); i < k; i++ {
				data[done+i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[4*i:]))
			}
			done += k
		}
		return data, nil
	}
	init := n
	if init > chunk {
		init = chunk
	}
	data := make([]float32, 0, init)
	var dec [chunk]float32
	for done := uint64(0); done < n; {
		k := n - done
		if k > chunk {
			k = chunk
		}
		buf := enc[:4*k]
		if err := rd.full(buf); err != nil {
			return nil, fmt.Errorf("reading data: %w", err)
		}
		for i := uint64(0); i < k; i++ {
			dec[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[4*i:]))
		}
		data = append(data, dec[:k]...)
		done += k
	}
	return data, nil
}

// WriteTensors writes ts to w in the package's wire format: magic
// "LNNS", version, count, then each tensor's rank, shape and
// little-endian float32 payload (see the package doc for the byte
// layout).
//
// Errors (never panics): more than maxCount (2^20) tensors, a nil
// tensor, a rank above maxRank (8), a negative dimension, an element
// count that overflows int64 (checked with the same overflow-safe
// multiplication as tensor.Size, but returning an error instead of
// panicking), a tensor whose Shape and Data disagree, and any I/O
// failure (reported once, wrapped).
func WriteTensors(w io.Writer, ts []*tensor.Tensor) error {
	if uint64(len(ts)) > maxCount {
		return fmt.Errorf("serialize: %d tensors exceed the stream count limit %d", len(ts), maxCount)
	}
	bw := &writer{w: w, sum: newCRC32C()}
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
	return bw.checksum()
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

// ReadTensors reads a stream written by WriteTensors (v2, checksummed) or by
// an older build (v1, no checksum) and returns its tensors, each in a freshly
// allocated buffer that aliases nothing of the reader's bytes.
//
// Errors (never panics — the stream is untrusted input): a bad magic,
// an unknown version byte (directional: "newer version… update this
// build" above Version, "corrupt or forged" below), a tensor count
// above maxCount (2^20), a rank above maxRank (8), negative dimensions,
// a payload above maxElems (2^30 float32s) or overflowing int64, any
// truncation (surfaced as io.ErrUnexpectedEOF), trailing bytes after
// the last counted tensor (v1) or after the checksum (v2), a v2
// checksum mismatch (the stream is corrupt or has been tampered with),
// and underlying I/O errors. Memory use is bounded as described in the
// package doc: fixed limits validated first, a remaining-bytes check on
// known-length readers, and growth proportional to the bytes actually
// delivered on unknown-length readers. The checksum is computed
// incrementally as the stream is parsed and verified after the last
// tensor, so it never requires buffering the stream.
func ReadTensors(r io.Reader) ([]*tensor.Tensor, error) {
	rd := newReader(r)
	var m [4]byte
	if err := rd.full(m[:]); err != nil {
		return nil, fmt.Errorf("serialize: reading stream magic: %w", err)
	}
	if m != magic {
		return nil, fmt.Errorf("serialize: bad magic % x, want % x (\"LNNS\"): not an LNN tensor stream", m[:], magic[:])
	}
	var vb [1]byte
	if err := rd.full(vb[:]); err != nil {
		return nil, fmt.Errorf("serialize: reading format version: %w", err)
	}
	switch vb[0] {
	case magicV1:
		// Legacy layout: no checksum. Parsed exactly as v1 always was, so a
		// checkpoint saved by an older build loads byte-for-byte unchanged.
		return readV1(rd)
	case Version:
		// v2: every byte from the magic on is fed to the CRC as it is read,
		// and the stored checksum is verified after the last tensor.
		rd.sum = newCRC32C()
		rd.sum.Write(m[:])
		rd.sum.Write(vb[:])
		return readV2(rd)
	default:
		// Rejected, never guessed (see "Format versioning" in the package
		// doc): the message tells the caller which way the skew goes, so a
		// checkpoint from the future reads as an actionable "update this
		// build" and anything below v1 as the corruption it must be.
		if vb[0] > Version {
			return nil, fmt.Errorf("serialize: unsupported format version %d (this build reads version %d): the stream was written by a newer version of the library; update this build to read it", vb[0], Version)
		}
		return nil, fmt.Errorf("serialize: unsupported format version %d (this build reads version %d): no earlier layout exists, the stream is corrupt or forged", vb[0], Version)
	}
}

// readTensorsBody parses the count field and the count tensors, shared by
// both layouts. For v2 the reader's sum is already live, so every byte read
// here is fed to the checksum.
func readTensorsBody(rd *reader) ([]*tensor.Tensor, error) {
	var cb [4]byte
	if err := rd.full(cb[:]); err != nil {
		return nil, fmt.Errorf("serialize: reading tensor count: %w", err)
	}
	count := uint64(binary.LittleEndian.Uint32(cb[:]))
	if count > maxCount {
		return nil, fmt.Errorf("serialize: stream claims %d tensors, exceeding the count limit %d", count, maxCount)
	}
	// Grow the slice as tensors arrive instead of allocating count slots up
	// front: a 9-byte stream claiming nearly maxCount tensors must not force
	// an ~8 MiB pointer slice just to fail on the first missing rank byte
	// (red team F3). Legitimate streams stay small — model blobs carry 17
	// tensors — so a small starting capacity grows at most a couple of times.
	const countStart = 16
	start := count
	if start > countStart {
		start = countStart
	}
	ts := make([]*tensor.Tensor, 0, start)
	for i := uint64(0); i < count; i++ {
		t, err := readTensor(rd, int(i))
		if err != nil {
			return nil, err
		}
		ts = append(ts, t)
	}
	return ts, nil
}

// readV1 finishes a version-1 stream: no checksum, and anything after the
// last tensor's payload is corruption rather than "another message".
func readV1(rd *reader) ([]*tensor.Tensor, error) {
	ts, err := readTensorsBody(rd)
	if err != nil {
		return nil, err
	}
	var probe [1]byte
	if err := rd.full(probe[:]); err == nil {
		return nil, fmt.Errorf("serialize: unexpected trailing byte(s) after tensor %d", len(ts)-1)
	} else if !errors.Is(err, io.ErrUnexpectedEOF) {
		return nil, fmt.Errorf("serialize: checking for trailing data: %w", err)
	}
	return ts, nil
}

// readV2 finishes a version-2 stream: it reads the stored CRC-32C (which is
// itself excluded from the checksum), compares it against the value computed
// over magic, version, count and every tensor, and rejects a mismatch as
// corruption. Trailing bytes after the checksum are corruption too.
func readV2(rd *reader) ([]*tensor.Tensor, error) {
	ts, err := readTensorsBody(rd)
	if err != nil {
		return nil, err
	}
	want := rd.sum.Sum32()
	rd.sum = nil // the stored checksum bytes are not part of the checksum
	var cb [4]byte
	if err := rd.full(cb[:]); err != nil {
		return nil, fmt.Errorf("serialize: reading stream checksum: %w", err)
	}
	if got := binary.LittleEndian.Uint32(cb[:]); got != want {
		return nil, fmt.Errorf("serialize: checksum mismatch: stored %08x, computed %08x: the stream is corrupt or has been tampered with", got, want)
	}
	var probe [1]byte
	if err := rd.full(probe[:]); err == nil {
		return nil, fmt.Errorf("serialize: unexpected trailing byte(s) after the stream checksum")
	} else if !errors.Is(err, io.ErrUnexpectedEOF) {
		return nil, fmt.Errorf("serialize: checking for trailing data: %w", err)
	}
	return ts, nil
}

// readTensor decodes tensor idx, validating every claim before allocating:
// rank against maxRank, dimensions against negativity, and the total element
// count against both overflow (bits.Mul64) and maxElems. The payload itself
// is then read through reader.floats, which picks an allocation strategy by
// what the underlying reader can promise (see its doc).
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
	data, err := rd.floats(n)
	if err != nil {
		return nil, fmt.Errorf("serialize: tensor %d: %w", idx, err)
	}
	return &tensor.Tensor{Shape: shape, Data: data}, nil
}

// WriteParameters writes the Data tensors of params (in order) to w. It
// is WriteTensors over p.Data with nil guards, named for the checkpoint
// use case: the order of params fixes the order of the tensors in the
// stream, and LoadParameters restores by position.
//
// Errors (never panics): a nil parameter or one without Data, plus
// every WriteTensors error (I/O failure, shape/Data disagreement). It
// writes nothing but values: gradients and graph structure are not part
// of the stream.
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

// LoadParameters reads a stream written by WriteParameters and copies
// the values back into params IN PLACE, by position: the i-th streamed
// tensor overwrites params[i].Data. The *autograd.Variable pointers
// keep their identity, so every graph edge that references them stays
// valid and graph ownership is preserved.
//
// Errors (never panics — the stream is untrusted input): a nil
// parameter or one without Data, a count mismatch, any shape mismatch,
// and every ReadTensors failure (bad magic, unknown version,
// truncation as io.ErrUnexpectedEOF, hostile size claims). All shapes
// are validated before anything is copied, so a failing load leaves
// every parameter exactly as it was.
//
// The load overwrites each parameter's Data and deliberately leaves its
// Grad field untouched: gradients accumulated on an earlier graph
// survive the load as stale values. A caller that reuses the variables
// in a new graph should call ZeroGrad first, exactly as before any
// training step (see doc/persistence.md, "stale Grad").
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
