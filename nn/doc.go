// Package nn provides neural-network building blocks for the LNN library:
// the Linear layer, Wiring synapse topologies, the LTC liquid cell, and the
// Cell/Unroll abstractions for driving recurrent cells over sequences.
//
// # Modules and parameters
//
// A Module is anything that owns trainable parameters:
//
//	type Module interface {
//		Parameters() []*autograd.Variable
//	}
//
// ParametersOf flattens several modules into one slice, ready to be handed
// to a training loop:
//
//	params := nn.ParametersOf(cell, readout)
//
// Linear is the fully connected layer y = x @ W + b, with Xavier-uniform W
// and zero bias.
//
// # Cells and sequences
//
// A Cell advances one RNN step at a time:
//
//	type Cell interface {
//		Step(x, h *autograd.Variable, ts float64) (out, hNew *autograd.Variable)
//		StateSize() int
//	}
//
// Step takes x of shape [batch, inDim] and h of shape [batch, StateSize()]
// (nil means a zero initial state) and returns the step output plus the new
// raw state. ts is the time span the step covers and must be positive and
// finite: NaN, +/-Inf, zero and negative values panic. Unroll threads a
// cell over a whole input sequence at a fixed ts; the entire sequence stays
// in one graph, so a loss built on the outputs differentiates through time
// with a single Backward. Unroll over an empty sequence returns an empty
// output slice and h0 unchanged (possibly nil).
//
// # Liquid Time-Constant cells
//
// LTC implements the Liquid Time-Constant cell of Hasani et al., "Liquid
// Time-constant Networks" (AAAI 2021), following the reference
// implementation mlech26l/ncps. Each neuron is a leaky membrane whose time
// constant itself depends on synaptic input, which gives the network
// input-adaptive dynamics with far fewer units than a classical RNN. The
// membrane ODE is integrated with the reference's semi-implicit Euler
// scheme over `unfolds` solver substeps per Step, and positivity of
// capacitance, leak conductance and synaptic weights is enforced through
// softplus (the reference's implicit parameter constraints) rather than
// optimizer-side clipping. Reversal potentials are fixed random +/-1
// constants, deliberately not trainable and absent from Parameters().
// NewLTC takes an optional Wiring (nil means fully connected) whose binary
// masks gate individual synapses: sensory [inDim, units] and recurrent
// [units, units], entry [i, j] being the synapse from input/neuron i to
// neuron j. See doc/ltc.md for the equation-to-code correspondence and
// doc/pitfalls.md for the numeric fine print.
//
// # Closed-form Continuous-time cells
//
// CfC implements the Closed-form Continuous-time cell of Hasani et al.
// (2022), "Closed-form continuous-time neural networks" (Nature Machine
// Intelligence 4, 992-1003): the very membrane ODE the LTC integrates
// numerically, advanced over the step's time span by its Lemma 1
// closed-form solution instead of the semi-implicit Euler loop. NewCfC
// shares the LTC's synapse parameterization (the same 13 trainable
// tensors, fixed +/-1 reversal potentials, wiring masks, ts contract) and
// satisfies the same Cell interface, so Unroll drives it unchanged. See
// doc/cfc.md for the paper-to-code correspondence.
//
// # Training
//
// Training loops are written over Parameters() — ZeroGrad, forward,
// Backward, then the parameter update. The hand-rolled update loop (plain
// Go over p.Data/p.Grad) remains the basis for understanding the engine;
// the optimizer package (SGD, Momentum, Adam) packages exactly that loop
// and is the recommended form for production training. Global
// gradient-norm clipping stays caller-owned in both forms. See
// doc/training.md and examples/ltc-sequence for complete loops.
//
// # Concurrency
//
// lnn is single-threaded by design. Backward mutates the Grad buffers of a
// graph's leaf variables without synchronization, so running it
// concurrently on variables that share parameters is a data race that loses
// or corrupts gradients (verified under go test -race). Never share a
// Variable, Tensor, or computation graph across goroutines — give each
// goroutine its own cell, tensors and RNG.
//
// A minimal LTC forward/backward session:
//
//	rng := rand.New(rand.NewSource(1))
//	cell := nn.NewLTC(4, 8, nil, 6, rng) // inDim=4, units=8, fully connected, 6 ODE unfolds
//	readout := nn.NewLinear(8, 2, rng)
//
//	x := autograd.Var(tensor.Uniform(rng, -1, 1, 3, 4)) // batch of 3
//	ys, _ := nn.Unroll(cell, []*autograd.Variable{x, x}, nil, 0.1)
//	out := readout.Forward(ys[len(ys)-1])
//	loss := autograd.MeanAll(autograd.Hadamard(out, out))
//
//	params := nn.ParametersOf(cell, readout) // 13 LTC parameters plus W and B
//	for _, p := range params {
//		p.ZeroGrad()
//	}
//	loss.Backward() // gradients are now in p.Grad for every p
package nn
