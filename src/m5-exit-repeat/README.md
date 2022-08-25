---
title: The 'm5-exit-repeat' binary
layout: default
permalink: resources/m5-exit-repeat
shortdoc: >
    Source for 'm5-exit-repeat'. A 'm5-exit-repeat' binary runs the `m5_exit(0)` command in an infinite loop. This is useful for testing exit event handlers.
author: ["Bobby R. Bruce"]
---

This 'm5-exit-repeat' resource runs the `m5_exit` function in an infinite loop.
This resource is deliberately kept simple and is used primarily for testing purposes.

## Building Instructions

Run `make`.

**Note:** This will automatically clone the gem5 repository to the `src/m5-exit-repeat` directory.
If this is not desired, please ensure the gem5 repository is present in this directory before running make.

This will only compile the binary to the X86 ISA.

## Cleaning Instructions

Run `make clean` in the Makefile directory.

## Usage

As this binary utilizes the `m5_exit` function, it should be run within a gem5 simulation.
Its purpose is to test exit events handler and other situations where it's desirable for a simulation to continually run an `m5_exit` command.
It should be run in SE Mode.

### Example

The following code snippet utilizes the standard library:

```py

board.set_se_workload(Resource("x86-m5-exit-repeat"))

def unique_exit_event():
    print("Handling the first exit event.")
    yield False
    print("Handling the second exit event.")
    yield False
    print("Handling the third exit event. We'll exit now.")
    yield True

simulator = Simulator(
    board = board,
    on_exit_event = {
        ExitEvent.Exit : unique_exit_event(),
    },
)

```

This will handle the exit event three times.

## Pre-built binaries

Compiled to the X86 ISA: http://dist.gem5.org/dist/develop/test-progs/m5-exit-repeat/x86-m5-exit-repeat-20220825

## License

This code is covered by By the [03-Clause BSD License (BSD-3-Clause)](https://opensource.org/licenses/BSD-3-Clause).
