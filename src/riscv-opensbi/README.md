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

# RISCV Full System with OpenSBI Firmware as a Bootloader

This document provides instructions to create an (OpenSBI)[https://github.com/riscv-software-src/opensbi] bootloader binary and a Linux kernel binary that work with gem5 full system simulations.
The bootloader and the kernel binaries are completely independent; however, we'll provide instructions to build both binaries for completeness.

In gem5, this bootloader is supposed to work with the default RISC-V Linux kernel configuration from the Linux project, as well as with any disk RISC-V image provided by `gem5-resources`.
Different from the `riscv-fs` resource, the bootloader and the Linux kernel are separate inputs to gem5.

**Note:** The bootloader and the Linux kernel are compiled using a docker container as specified in `build-env.Dockerfile`.
The docker image is derived from Ubuntu 22.04 LTS and uses the cross compilers provided by Ubuntu.

We assume the following directory structure while following the instructions in this README file:

```
riscv-opensbi/
  |___ opensbi/                                # The source of OpenSBI cloned from https://github.com/riscv-software-src/opensbi
  |
  |___ linux/                                  # Linux source code
  |
  |___ build-env.dockerfile                    # A docker file to build cross compilation environment
  |
  |___ README.md                               # This README file
```

## Overview

`OpenSBI` is a reference implementation of RISC-V SBI (Supervisor Binary Interface), which provides an interface to the M-mode (machine mode) or HS-mode (hypervisor mode).
Despite the name, `OpenSBI` itself can act as a first-stage bootloader setting up the environment before jumping to the start of a payload, a Linux kernel in this case.
`OpenSBI` also offers handles for events requiring M-mode interventation, e.g., interrupts.

`OpenSBI` offers 3 different boot flows: `FW_JUMP`, `FW_DYNAMIC`, and `FW_PAYLOAD`.
The definition for each can be found in the official documentation.
In this document, we use `FW_JUMP` as a bootloader.

The `FW_JUMP` bootloader assumes that the device tree blob and the payload (in this case, the linux kernel) are written to memory before `FW_JUMP` bootloader is executed.
In gem5 simulation, the `FW_JUMP` bootloader itself, the device tree blob, and the linux kernel are written to memory before simulation.

During simulation, the `FW_JUMP` bootloader performs the first stage of booting then jumps to a specific address hardcoded in the bootloader.
Therefore, you need to make sure that the linux kernel is written to memory starting from that specific address.

## Building the Docker Image

The following command builds a Docker image as specified in the `build-env.dockerfile` file.
The image name is `gem5/riscv-gcc`.

In the `riscv-opensbi/` folder,

```sh
docker build -f build-env.dockerfile -t gem5/riscv-gcc .
```

## Downloading and Building the OpenSBI `FW_JUMP` Bootloader

The following commands download the official `OpenSBI` repository and checkout the `v1.3.1` release.

In the `riscv-opensbi/` folder,

```sh
git clone https://github.com/riscv-software-src/opensbi
cd opensbi
git checkout v1.3.1
```

The following commands use the `gem5/riscv-gcc` docker image to build the `FW_JUMP` firmware, which we will use as a bootloader later on.

In the `riscv-opensbi/` folder,

```sh
cd opensbi
docker run -u $UID:$GID --volume $PWD:/workdir -w /workdir --rm -it gem5/riscv-gcc
(docker) make PLATFORM=generic FW_JUMP=y FW_JUMP_ADDR=0x80200000 FW_JUMP_FDT_ADDR=0x87e00000 CROSS_COMPILE=riscv64-linux-gnu- -j`nproc`
(docker) exit
```

The above `make` command explicitly specifies that,

- `FW_JUMP_ADDR=0x80200000`: after the first stage booting is done by OpenSBI, it will jump to the instruction at the address `0x80200000`.
As we are interesting in booting the Linux kernel, we should write to memory the linux kernel (the vmlinux file) at the physical address of `0x80200000` before starting gem5 simulation.
- `FW_JUMP_FDT_ADDR=0x87e00000`: the device tree is located at `0x87e00000`.
This means, in gem5, we should write to memory the device tree blob at the physical address of `0x87e00000` before starting gem5 simulation.

The bootloader is located at `opensbi/build/platform/generic/firmware/fw_jump.elf`.

## Downloading and Building the Linux kernel

The following commands download the linux kernel version 6.5.5 from `kernel.org`, and build the linux kernel using the default configuration.

In the `riscv-opensbi/` folder,

```sh
wget https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.5.5.tar.xz
tar xf linux-6.5.5.tar.xz
mv linux-6.5.5 linux
cd linux
docker run -u $UID:$GID --volume $PWD:/workdir -w /workdir --rm -it gem5/riscv-gcc
(docker) yes "" | make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- oldconfig
(docker) make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- -j$(nproc)
(docker) exit
```

The `yes "" | make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- oldconfig` chooses all default options in the linux kernel configuration for RISC-V.
To change the configuration after that step, you can use `make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- menuconfig`.

The linux kernel is located at `linux/vmlinux`.
