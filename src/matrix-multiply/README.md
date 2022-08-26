---
title: The 'matrix-multiply' binary
layout: default
permalink: resources/matrix-multiply
shortdoc: >
    Source for 'matrix-multiply'. A matrix-multiply binary runs a multiplication on two 100x100 matrixes. The sum of the multiplied matrix is printed upon completion.
author: ["Bobby R. Bruce"]
---

The 'matrix-multiply' resource runs a multiplication on two 100x100 matrixes.
The sum of the multiplied matrix is printed upon completion.

## Building Instructions

Run `make`.

This will only compile the binary to the X86 ISA.

## Cleaning Instructions

Run `make clean` in the Makefile directory.

## Usage

As this binary does not contain any special `m5` library code it can be run outside of a gem5 simulation:

```sh
./matrix-multiply
```

It can also be run in a gem5 simulation in SE mode.
Below is a snippet which utilizes the gem5 standard library to do so:

```py
board.set_se_workload(Resource("x86-matrix-multiply"))

simulator = Simulator(board = board)
```

## Pre-built binaries

Compiled to the X86 ISA: http://dist.gem5.org/dist/develop/test-progs/matrix-multiply/x86-matrix-multiply-20220825

## License

This code is covered by By the [03-Clause BSD License (BSD-3-Clause)](https://opensource.org/licenses/BSD-3-Clause).
