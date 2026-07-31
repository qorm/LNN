// Command hello-train is the smallest possible training program: fit the
// line y = 2x + 1 with ONE CfC neuron, then save the model for
// examples/hello-use. Run from the repository root:
//
//	go run ./examples/hello-train
package main

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/nn"
	"github.com/qorm/LNN/tensor"
)

func main() {
	// 1. Create the model: a single neuron. With units=1 the cell's Step
	// output is already [batch,1], so no readout layer is needed. One neuron
	// suffices here because the target is a line; a fixed seed (42) makes the
	// run repeatable, and keeps this neuron in an initialization that learns
	// (others can stall on this toy task).
	cell := nn.NewCfC(1, 1, nil, rand.New(rand.NewSource(42)))

	// 2. Make some data: y = 2x + 1. The x values stay small so the cell's
	// sigmoid synapses work in their near-linear region (a single neuron
	// cannot bend a wide input range into a line).
	xs := []float32{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}
	ys := []float32{1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4}

	// 3. Training loop: forward → zero grads → backward → update. The update
	// is plain hand-written gradient descent, the one new concept here (the
	// optimizer package wraps this loop — see doc/training.md; production
	// training adds gradient clipping — see examples/ltc-sequence).
	params := cell.Parameters()
	lr := float32(0.05)
	var first, last float32
	for it := 0; it < 500; it++ {
		out, _ := cell.Step(autograd.Var(tensor.FromData(xs, 8, 1)), nil, 1.0)
		diff := autograd.Sub(out, autograd.Var(tensor.FromData(ys, 8, 1)))
		loss := autograd.MeanAll(autograd.Hadamard(diff, diff))
		for _, p := range params {
			p.ZeroGrad()
		}
		loss.Backward()
		for _, p := range params {
			if p.Grad == nil {
				continue
			}
			for i := range p.Data.Data {
				p.Data.Data[i] -= lr * p.Grad.Data[i]
			}
		}
		if it == 0 {
			first = loss.Value()
		}
		last = loss.Value()
		if it%100 == 0 || it == 499 {
			fmt.Printf("iter %3d  loss=%.6f\n", it, loss.Value())
		}
	}

	// 4. Save the model for hello-use to load.
	path := filepath.Join(os.TempDir(), "hello-LNN-model.lnns")
	f, err := os.Create(path)
	must(err)
	must(nn.SaveCfC(f, cell))
	must(f.Close())
	fmt.Printf("\nmodel saved to %s\nnow run: go run ./examples/hello-use\n\n", path)

	// 5. Print what we learned: the model's answers next to the true line.
	pred, _ := cell.Step(autograd.Var(tensor.FromData(xs, 8, 1)), nil, 1.0)
	for i := range xs {
		fmt.Printf("x=%.1f  model says %.3f  (true %.1f)\n", xs[i], pred.Data.Data[i], ys[i])
	}
	if last >= first {
		fmt.Println("FAIL: loss did not decrease")
		os.Exit(1)
	}
	fmt.Println("OK: loss decreased")
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "FATAL:", err)
		os.Exit(1)
	}
}
