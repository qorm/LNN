// Command ltc-resume is an end-to-end checkpoint-and-resume demonstration:
// it trains a small LTC plus a Linear readout with Adam on the same toy
// bounded-accumulator task as examples/ltc-sequence (at every step the
// network sees a sign u_t in {-1, +1} and must output the clipped running
// sum s_t = clip(s_{t-1} + 0.25*u_t, -1, 1)), checkpoints model + readout +
// optimizer state to a temporary directory, then resumes training into
// brand-new objects and proves the resumed trajectory is bit-identical to
// uninterrupted training.
//
// The cell is the LTC rather than the CfC so the example pairs directly
// with examples/ltc-sequence, and its ODE unrolls exercise the resume path
// through a heavier graph. The checkpoint streams themselves are
// cell-agnostic: nn.SaveCfC/nn.LoadCfC drop in unchanged, because the
// optimizer state stream ("LNO1") sees only the parameter slice.
//
// Run from the repository root:
//
//	go run ./examples/ltc-resume
//
// The demo makes two points:
//
//   - Adam's state must be checkpointed alongside the model. Its moment
//     buffers (m, v), update count t and bias-correction powers (Beta1^t,
//     Beta2^t) live only inside the optimizer; a model-only resume restarts
//     the bias correction as if t were 0 and silently discards every prior
//     step's learning-rate adaptation. optimizer.SaveState/LoadState
//     persist those buffers bit for bit, so the resumed run continues with
//     exactly the adaptation the interrupted run had accumulated.
//   - Seeds do not survive a checkpoint, and do not need to: Load runs the
//     constructor with a throwaway RNG and overwrites every RNG-derived
//     field from the stream, so a model rebuilt under a different seed and
//     loaded steps bit-identically (act 2 below).
//
// Two more contracts appear explicitly: serialize.LoadParameters restores
// values in place and deliberately leaves stale Grad untouched, so the
// resumed parameters are ZeroGrad'd before the first step (exactly as
// before any training step); and SaveState/LoadState key state by index
// into the params slice, so the same parameter order — here
// nn.ParametersOf(cell, readout) on both sides — must be given to both.
//
// The program is a self-checking smoke test: fixed seeds, deterministic
// output, and a non-zero exit if any assertion fails.
package main

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/nn"
	"github.com/qorm/LNN/optimizer"
	"github.com/qorm/LNN/serialize"
	"github.com/qorm/LNN/tensor"
)

const (
	inDim   = 1
	units   = 8
	unfolds = 4
	seqLen  = 12
	batch   = 16
	trainN  = 60 // act 1 iterations before the checkpoint
	resumeM = 60 // act 3 iterations after the reload
	lr      = 0.01
	maxNorm = 1.0  // global gradient-norm clip
	ts      = 1.0  // ODE time span per step
	step    = 0.25 // accumulator increment per input
	clip    = 1.0  // target saturation bound
)

func main() {
	// One shared data stream for the checkpointed run: act 1 consumes the
	// first trainN draws, act 3 consumes the next resumeM draws. The
	// uninterrupted control starts from the same seed, so it sees exactly
	// the same sequences at every iteration.
	data := rand.New(rand.NewSource(42))

	fmt.Printf("LTC resume task: inDim=%d units=%d unfolds=%d seqLen=%d batch=%d optimizer=Adam(lr=%v)\n\n",
		inDim, units, unfolds, seqLen, batch, lr)

	fmt.Printf("== act 1: train %d iterations, checkpoint model + readout + Adam state ==\n", trainN)
	modelRng := rand.New(rand.NewSource(7))
	cell := nn.NewLTC(inDim, units, nil, unfolds, modelRng)
	readout := nn.NewLinear(units, 1, modelRng)
	params := nn.ParametersOf(cell, readout)
	opt := optimizer.NewAdamDefault(lr)
	losses1 := train(data, cell, readout, params, opt, trainN, 0, 20)

	dir, err := os.MkdirTemp("", "ltc-resume-")
	must(err)
	defer os.RemoveAll(dir)
	modelPath := filepath.Join(dir, "ltc.model")
	paramPath := filepath.Join(dir, "readout.params")
	statePath := filepath.Join(dir, "adam.state")
	saveFile(modelPath, func(w io.Writer) error { return nn.SaveLTC(w, cell) })
	saveFile(paramPath, func(w io.Writer) error { return serialize.WriteParameters(w, readout.Parameters()) })
	saveFile(statePath, func(w io.Writer) error { return optimizer.SaveState(w, opt, params) })
	fi1, _ := os.Stat(modelPath)
	fi2, _ := os.Stat(paramPath)
	fi3, _ := os.Stat(statePath)
	fmt.Printf("saved ltc.model (%d bytes) + readout.params (%d bytes) + adam.state (%d bytes)\n",
		fi1.Size(), fi2.Size(), fi3.Size())
	fmt.Println("checkpoint files live in a temp directory, removed on exit")

	fmt.Printf("\n== act 2: rebuild with a different seed, load all three streams ==\n")
	freshRng := rand.New(rand.NewSource(123)) // seed is irrelevant: Load overwrites every RNG-derived field
	loaded, err := nn.LoadLTC(openStream(modelPath))
	must(err)
	readout2 := nn.NewLinear(units, 1, freshRng)
	must(serialize.LoadParameters(openStream(paramPath), readout2.Parameters()))
	params2 := nn.ParametersOf(loaded, readout2)
	// Stale-Grad contract (serialize.LoadParameters): the load overwrites
	// each parameter's Data and leaves its Grad untouched, so variables
	// about to enter a new graph are zeroed first — exactly as before any
	// training step. The same nn.ParametersOf(cell, readout) order on both
	// sides satisfies SaveState/LoadState's index-keying contract.
	for _, p := range params2 {
		p.ZeroGrad()
	}
	opt2 := optimizer.NewAdamDefault(lr) // hyperparameters come from the destination optimizer
	must(optimizer.LoadState(openStream(statePath), opt2, params2))
	fmt.Println("LoadLTC + LoadParameters + LoadState: ok")

	// The loaded model is bit-for-bit the saved one, whatever the seed.
	probe := autograd.Var(tensor.Uniform(rand.New(rand.NewSource(99)), -1, 1, 4, inDim))
	out1, _ := cell.Step(probe, nil, ts)
	out2, _ := loaded.Step(probe, nil, ts)
	sameStep := true
	for i := range out1.Data.Data {
		if math.Float32bits(out1.Data.Data[i]) != math.Float32bits(out2.Data.Data[i]) {
			sameStep = false
		}
	}
	fmt.Printf("bit-identical: %v\n", sameStep)

	fmt.Printf("\n== act 3: resume %d iterations, compare with %d uninterrupted ==\n", resumeM, trainN+resumeM)
	losses2 := train(data, loaded, readout2, params2, opt2, resumeM, trainN, 20)

	// Control: the same construction seed, a fresh data stream from the
	// same seed, a fresh Adam — trained straight through with no
	// checkpoint in between (and therefore no output, by design).
	ctrlRng := rand.New(rand.NewSource(7))
	ctrlCell := nn.NewLTC(inDim, units, nil, unfolds, ctrlRng)
	ctrlReadout := nn.NewLinear(units, 1, ctrlRng)
	ctrlParams := nn.ParametersOf(ctrlCell, ctrlReadout)
	ctrlLosses := train(rand.New(rand.NewSource(42)), ctrlCell, ctrlReadout, ctrlParams,
		optimizer.NewAdamDefault(lr), trainN+resumeM, 0, 0)

	resumed := append(append([]uint32(nil), losses1...), losses2...)
	sameLoss := len(resumed) == len(ctrlLosses)
	for i := range ctrlLosses {
		if sameLoss && resumed[i] != ctrlLosses[i] {
			sameLoss = false
		}
	}
	sameParams := true
	for i := range ctrlParams {
		a, b := ctrlParams[i].Data.Data, params2[i].Data.Data
		for j := range a {
			if math.Float32bits(a[j]) != math.Float32bits(b[j]) {
				sameParams = false
			}
		}
	}
	fmt.Printf("loss bits identical for all %d steps: %v\n", trainN+resumeM, sameLoss)
	fmt.Printf("final parameters bit-identical: %v\n", sameParams)
	fmt.Printf("resume trajectory bit-identical to uninterrupted: %v\n", sameLoss && sameParams)

	first := float64(math.Float32frombits(losses1[0]))
	last := float64(math.Float32frombits(losses2[len(losses2)-1]))
	fmt.Printf("\nfirst loss %.6f -> final loss %.6f\n", first, last)
	switch {
	case !sameStep:
		fmt.Println("FAIL: loaded model does not step bit-identically to the saved one")
		os.Exit(1)
	case !(sameLoss && sameParams):
		fmt.Println("FAIL: resumed trajectory diverged from uninterrupted training")
		os.Exit(1)
	case last >= first:
		fmt.Println("FAIL: loss did not decrease")
		os.Exit(1)
	}
	fmt.Println("OK: resume is bit-exact and loss decreased")
}

// train runs iters iterations of the bounded-accumulator task with Adam and
// caller-owned global gradient-norm clipping (the recommended production
// form from doc/training.md), returning the loss measured before each
// iteration's update as a float32 bit pattern. The clip is a deterministic
// function of the gradients alone, so it preserves bit-exactness: two runs
// over the same gradients take the same clipped step. Loss lines print with
// a global iteration offset whenever printEvery > 0.
func train(data *rand.Rand, cell nn.Cell, readout *nn.Linear, params []*autograd.Variable,
	opt optimizer.Optimizer, iters, offset, printEvery int) []uint32 {
	losses := make([]uint32, 0, iters)
	for it := 0; it < iters; it++ {
		// Fresh random sequences every iteration (online SGD).
		xs := make([]*autograd.Variable, seqLen)
		targets := make([]*autograd.Variable, seqLen)
		state := make([]float32, batch)
		for t := 0; t < seqLen; t++ {
			xb := make([]float32, batch)
			yb := make([]float32, batch)
			for b := 0; b < batch; b++ {
				u := float32(1)
				if data.Intn(2) == 0 {
					u = -1
				}
				xb[b] = u
				s := state[b] + step*u
				if s > clip {
					s = clip
				} else if s < -clip {
					s = -clip
				}
				state[b] = s
				yb[b] = s
			}
			xs[t] = autograd.Var(tensor.FromData(xb, batch, inDim))
			targets[t] = autograd.Var(tensor.FromData(yb, batch, 1))
		}

		// Forward through the unrolled sequence and a linear readout.
		ys, _ := nn.Unroll(cell, xs, nil, ts)
		var acc *autograd.Variable
		for t, y := range ys {
			diff := autograd.Sub(readout.Forward(y), targets[t]) // [batch, 1]
			sq := autograd.Hadamard(diff, diff)
			if t == 0 {
				acc = sq
			} else {
				acc = autograd.Add(acc, sq)
			}
		}
		loss := autograd.Scale(autograd.MeanAll(acc), 1/float32(seqLen))
		losses = append(losses, math.Float32bits(loss.Value()))

		// ZeroGrad is the caller's contract: Step never does it.
		for _, p := range params {
			p.ZeroGrad()
		}
		loss.Backward()

		// Global gradient-norm clipping (caller-owned): accumulate the norm
		// in float64, rescale the gradients in place, then let Adam step.
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

		if printEvery > 0 && (it%printEvery == 0 || it == iters-1) {
			fmt.Printf("iter %3d  loss=%.6f\n", it+offset, float64(loss.Value()))
		}
	}
	return losses
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

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "FATAL:", err)
		os.Exit(1)
	}
}
