# Makefile for the lnn library (pure-Go tensor / autograd / liquid NN).
# POSIX shell syntax; every recipe line is indented with a TAB.

GO ?= go
# Regex filter for `make bench` (e.g. `make bench BENCH=MatMul`).
BENCH ?= .
# Per-target fuzzing window for `make fuzz` (exploratory, local).
FUZZTIME ?= 30s
# Per-target window for `make fuzz-smoke` (CI gate; override for longer runs).
SMOKETIME ?= 10s

.PHONY: all fmt vet test cover build bench fuzz fuzz-smoke

# Default: format, vet, and run the whole test suite with the race detector.
all: fmt vet test

# Format every Go file in the repository in place (-l lists what changed).
fmt:
	gofmt -l -w .

vet:
	$(GO) vet ./...

test:
	$(GO) test ./... -count=1 -race

# Full coverage profile plus a per-function summary; the test run itself
# prints one "coverage: ..." line per package.
cover:
	$(GO) test ./... -count=1 -race -coverprofile=coverage.txt -covermode=atomic
	$(GO) tool cover -func=coverage.txt

build:
	$(GO) build ./...

# Run the benchmark suite with allocation stats. Filter with the BENCH
# variable, e.g. `make bench BENCH=MatMul`. Deliberately not part of `all`:
# a full benchmark run is slow.
bench:
	$(GO) test ./... -run '^$$' -bench $(BENCH) -benchmem

# The native Go fuzz targets (func FuzzXxx), as package/Name pairs. These
# crystallize the red team's ad-hoc mutation discipline into sustainable
# `go test -fuzz` targets over the untrusted-stream and constructor surfaces.
FUZZ_TARGETS = \
	serialize/FuzzReadTensors \
	serialize/FuzzLoadParameters \
	nn/FuzzLoadLTC \
	nn/FuzzLoadCfC \
	nn/FuzzLoadLinear \
	optimizer/FuzzLoadState \
	tensor/FuzzTensorConstructors \
	autograd/FuzzOpGraphs

# Fuzz every target for FUZZTIME each (exploratory; -fuzz accepts one target
# per invocation, so they run sequentially). Any crash fails the run and
# leaves the minimizing input under <pkg>/testdata/fuzz/.
fuzz:
	@for t in $(FUZZ_TARGETS); do \
		pkg=$${t%/*}; name=$${t#*/}; \
		echo "==> fuzzing $$pkg/$$name for $(FUZZTIME)"; \
		$(GO) test ./$$pkg -run '^$$' -fuzz '^'"$$name"'$$' -fuzztime=$(FUZZTIME) || exit 1; \
	done

# Short-window gate for CI: SMOKETIME per target (10s default, 30s in CI).
fuzz-smoke:
	@for t in $(FUZZ_TARGETS); do \
		pkg=$${t%/*}; name=$${t#*/}; \
		echo "==> smoke $$pkg/$$name for $(SMOKETIME)"; \
		$(GO) test ./$$pkg -run '^$$' -fuzz '^'"$$name"'$$' -fuzztime=$(SMOKETIME) || exit 1; \
	done
