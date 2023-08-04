---
title: gem5 Specific RISC-V tests
tags:
    - testing
    - riscv
layout: default
permalink: resources/asmtests
license: BSD-3-Clause
---

About
-----

This work provides assembly testing infrastructure including single-threaded
and multi-threaded tests for the RISC-V ISA in gem5. Each test targets an
individual RISC-V instruction or a Linux system call. It uses system call
emulation (SE) mode in gem5.

This work is based on the "riscv-tests" project.

Link to the orignal riscv-tests projects can be found here:
  <https://github.com/riscv/riscv-tests>

Link to the original riscv-tests project's LICENSE and README can be found
here:
  <https://github.com/riscv/riscv-tests/blob/master/LICENSE>

  <https://github.com/riscv/riscv-tests/blob/master/README.md>

Specific commit ID that this work is based off: <https://github.com/riscv/riscv-tests/tree/68cad7baf3ed0a4553fffd14726d24519ee1296a>.

Changes from the orignal riscv-tests project
--------------------------------------------

1. New testing environment for gem5

Since the original riscv-tests project is designed for bare-metal systems (i.e.,
without OS support), it offers several environments to control how a test
interacts with a host machine (to-host communication). However, in gem5 SE
mode, gem5 emulates an OS, and there is no host machine. Therefore, we
developed a new testing environment called `ps` for gem5.

This testing environment uses system call `exit` to return test results as an
exit code of a particular test instead of writing them to a host machine. This
environment requires the testing platform to implement/emulate at least `exit`
system call.

2. Minimal threading library written in assembly (`isa/macros/mt`)

To simplify debugging multi-threading systems, we developed a minimal threading
library that supports very basic threading functionality including creating a
thread, exiting a thread, waiting for some thread(s) on a condition, and waking
up some thread(s).

Multi-threaded tests can rely on this library to manage multiple threads.

3. RISC-V AMO, LR, and SC instruction tests (`isa/rv32uamt`, `isa/rv64uamt`)

This is a set of assembly tests that target multi-core systems and test AMO
instructions. This test set uses a minimal number of system calls (i.e., clone,
mmap, munmap and exit) to create and manage threads.  It does not use any
complex sleep/wakeup mechanism to manage and synchronize threads to avoid
adding extra unnecessary complexity. The major goal of this test set is to
stress AMO instructions. Threads only synchronize at the end of their
execution. The master thread does a spin-wait to wait for all threads to
complete before it checks final results.

4. Thread-related system call tests (`isa/rv32samt`, `isa/rv64samt`)

This is a set of assembly tests that target thread-related system calls and
thread wait/wakeup behaviors. This set reuses some of the tests in
`isa/rv64uamt` but uses more advanced futex system call operations to make
threads wait and wake up in certain cases. This test set also checks functional
behaviors of threads after a wait/wakeup operation.

5. Bit-manipulation ISA tests (`isa/rv32ub`, `isa/rv64ub`)

This is a instructions test sets of Zba, Zbb, Zbc and Zbs extensions. They are
bit-manipulations of registers.

6. Makefile for `benchmarks` directory

The `compile_template` in the Makefile has been changed to not use 
the default `gcc` options and the `riscv-tests` linkers. 
Instead, the new compile template only uses the `common` directory in `benchmarks` and the `-static` and `-O2` flags. 
To facilitate gem5 compatible ROIs, the `Makefile` links with the `libm5.a` present in the `gem5/include` directory 
(NOTE: the `gem5` directory must be in the `common` directory for compiling the benchmarks).
As part of this change, all the source code of the benchmarks use `m5_work_begin` and `m5_work_end` to mark the beginning and end of the ROI.

7. `mm` benchmark source code

A minor change was made to the `mm` benchmark source code to make it 
compatible with the `Makefile` changes mentioned above. 
Since `mm_main.c` used `thread_entry` as the `main` function, 
the compiler was not able to find the `main` function. 
This was fixed by renaming `thread_entry` to `main`.

How to compile this test suite
------------------------------

1. Install RISC-V GNU toolchain. Source code and instruction on how to install
it can be found here: <https://github.com/riscv/riscv-gnu-toolchain>.

2. Put gem5 with compiled m5ops in the `common` directory. Documentation on how to compile m5ops can be found here: <http://www.gem5.org/documentation/general_docs/m5ops>.

3. Navigate to `gem5/include/gem5/m5ops.h` and change the `#include <gem5/asm/generic/m5ops.h>` statement to `#include <gem5/include/gem5/asm/generic/m5ops.h>`.

4. Run `make`.

5. Test binaries are in `bin`.
