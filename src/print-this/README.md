---
title: The 'print-this' binary
layout: default
permalink: resources/print-this
shortdoc: >
    Sources for 'print-this' binary.
author: ["Zhantong Qiu"]
---

This 'print-this' resource is a simple binary which takes in two parameters: a string to print and an integer specifying the number of times it is to be printed.

This resource is deliberately kept simple and is used primarily for testing purposes.

## Building Instructions

Run `make`.

## Cleaning Instructions

Run `make clean` on the Makefile directory.

## Usage

It takes two parameters, a string to print and an integer stating how many times it should be printed.

### Example

`./print_this "a string to print" 10`

Will output:

```
1 a string to print
2 a string to print
3 a string to print
4 a string to print
5 a string to print
6 a string to print
7 a string to print
8 a string to print
9 a string to print
10 a string to print
```

## Pre-built binaries

Compiled to the RISC-V ISA: http://dist.gem5.org/dist/develop/test-progs/print-this/riscv-print-this

Compiled to the X86 ISA: http://dist.gem5.org/dist/develop/test-progs/print-this/x86-print-this

## License

This code is covered by By the [03-Clause BSD License (BSD-3-Clause)](https://opensource.org/licenses/BSD-3-Clause).
