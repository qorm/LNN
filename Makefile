# Makefile for the lnn library (pure-Go tensor / autograd / liquid NN).
# POSIX shell syntax; every recipe line is indented with a TAB.

GO ?= go

.PHONY: all fmt vet test cover build

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
