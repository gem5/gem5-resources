---
title: RISC-V Full System with OpenSBI bootloader
tags:
    - fullsystem
    - bootloader
    - riscv
layout: default
permalink: resources/riscv-opensbi
shortdoc: >
    Resources to build OpenSBI bootloader that works with gem5 full system simulations.
author: ["Hoa Nguyen"]
---

# RISCV Full System with Bootloader

This document provides instructions to create an (OpenSBI)[https://github.com/riscv-software-src/opensbi] bootloader binary and a Linux kernel binary that work with gem5 full system simulations.
The bootloader and the kernel binaries are completely independent; however, we'll provide instructions to build both binaries for the sake of completeness.

In gem5, this bootloader is supposed to work with the default RISC-V Linux kernel configuration from the Linux project, as well as with any disk RISC-V image provided by `gem5-resources`.
Different from the `riscv-fs` resource, the bootloader and the Linux kernel are separate inputs to gem5.

**Note:** The bootloader is compiled using a docker container as specified in `build-env.Dockerfile` derived from Ubuntu 22.04 LTS and uses the cross compilers provided by Ubuntu.

We assume the following directory structure while following the instructions in this README file:

```
riscv-opensbi/
  |___ opensbi/                                # The source of OpenSBI cloned from https://github.com/riscv-software-src/opensbi
  |
  |___ linux/                                  # Linux source code
  |
  |___ README.md                               # This README file
```

## Overview

`OpenSBI` is a reference implementation of RISC-V SBI (Supervisor Binary Interface), which provides an interface to the M-mode (machine mode) or HS-mode (hypervisor mode).
Despite the name, `OpenSBI` itself can act as a first-stage bootloader setting up the environment before jumping to the start of a payload, a Linux kernel in this case.
`OpenSBI` also offers handles for events requiring M-mode interventation, e.g., interrupts.

`OpenSBI` offers 3 different boot flows: `FW_JUMP`, `FW_DYNAMIC`, and `FW_PAYLOAD`.
The definition for each can be found in the official documentation.
We use `FW_JUMP` as a bootloader.

TODO: inputs to `FW_JUMP` and how it works.

## Building the OpenSBI FW_JUMP bootloader
