// Command hello-use is the smallest possible inference program: it loads the
// model that examples/hello-train saved and runs it. No training here — just
// load and one forward step. Run hello-train first, then from the repository
// root:
//
//	go run ./examples/hello-use
package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/nn"
	"github.com/qorm/LNN/tensor"
)

func main() {
	path := filepath.Join(os.TempDir(), "hello-LNN-model.lnns")

	// 1. Open the saved model.
	f, err := os.Open(path)
	if err != nil {
		fmt.Println("no saved model — run hello-train first: go run ./examples/hello-train")
		os.Exit(1)
	}
	defer f.Close()

	// 2. Load it: the same single-neuron cell, with its trained weights.
	cell, err := nn.LoadCfC(f)
	must(err)

	// 3. Make an input: a few x values; the last one lies beyond the trained
	// range [0, 0.7], where the model's extrapolation drifts off the line.
	xs := []float32{0.15, 0.35, 0.55, 0.90}

	// 4. Run the model: one forward step over the batch.
	out, _ := cell.Step(autograd.Var(tensor.FromData(xs, 4, 1)), nil, 1.0)

	// 5. Print the prediction against the true line y = 2x + 1.
	for i, x := range xs {
		fmt.Printf("x=%.2f  model says %.3f  (true %.2f)\n", x, out.Data.Data[i], 2*x+1)
	}
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "FATAL:", err)
		os.Exit(1)
	}
}
