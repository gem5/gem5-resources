---
title: The 'matrix-multiply-omp' binary
layout: default
permalink: resources/matrix-multiply-omp
shortdoc: >
    Source for 'matrix-multiply-omp' resources. The matrix-multiple-omp binary runs a multiplication on two 300x300 matrixes followed by a multiplication of two 150x150 matrices. These two matrix opperations are iterated over by an amount specified by the user via the first parameter. This binary utilizes OpenMP to parallelize the matrix operation. The number of threads used is specified by the user via the second parameter. This was provided to the gem5 project by the National University of Singapore.
author: ["Bobby R. Bruce"]
---

The 'matrix-multiply-omp' resource runs a multiplication on two 300x300 matrixes followed by a multiplication of two 150x150 matrices.
These two matrix opperations are iterated over by an amount specified by the user via the first parameter.
This binary utilizes OpenMP to parallelize the matrix operation.
The number of threads used is specified by the user via the second parameter.

## Building Instructions

Run `make`.

## Cleaning Instructions

Run `make clean` in the Makefile directory.

## Usage

As this binary does not contain any special `m5` library code it can be run outside of a gem5 simulation:

```sh
./matrix-multiply-omp <number of iterations> <number of threads>
```

The first parameter is the number of iterations to make for the two matrix multiplication operations.
The second parameter is the number of threads to run the matrix multiplications on.

It can also be run in a gem5 simulation in SE mode.
Below is a snippet which utilizes the gem5 standard library to do so:

```py
board.set_se_workload(Resource("x86-matrix-multiply-omp"))

simulator = Simulator(board = board)
```

## Pre-built binaries

Compiled to the X86 ISA: http://dist.gem5.org/dist/develop/test-progs/matrix-multiply-omp/x86-matrix-multiply-omp-20230127.

## Source and License

This code was taken from http://blog.speedgocomputing.com/2010/08/parallelizing-matrix-multiplication.html.

It is licenced under [Creative Commons Attribution-Noncommercial-ShareAlike 3.0 Unported](https://creativecommons.org/licenses/by-nc-sa/3.0/).
