---
title: MI355X GPUFS smoke-kernel checkpoint
tags:
    - x86
    - amdgpu
    - gfx950
layout: default
permalink: resources/x86-mi355x-gpu-fs-smoke-checkpoint
shortdoc: >
    Configuration for creating the MI355X GPUFS smoke-kernel checkpoint.
---

# Resource: `x86-mi355x-gpu-fs-smoke-checkpoint`

This directory contains the gem5 configuration and HIP loader used to create
the MI355X full-system CI checkpoint. Linux and the ROCm 7.0 AMDGPU/KFD driver
are initialized first. The loader then initializes HIP, allocates host-visible
memory, loads the `x86-mi355x-gpu-fs-smoke` `gfx950` code object, completes one
warm-up dispatch, resets the test value, and takes the final checkpoint. The
restored system launches and verifies the kernel again.

The final resource is specialized for this smoke kernel. Its saved state
already contains the loaded module, resolved `_Z9incrementPi` function,
kernel argument, and completed warm-up dispatch. It cannot accept another
kernel after restoration. The intermediate loader-stage checkpoint described
below can load another compatible `gfx950` code object, but it does not provide
the final checkpoint's fast restore path.

The configuration uses `ViperBoard`, an Atomic x86 CPU, `MI355X`, the Viper
cache hierarchy, 8 GiB of system memory, 16 GiB of HBM, and four compute
units. Four CUs retain one complete SQC and scalar-cache group. KVM is not
required. It obtains version `1.0.0` of both
`x86-ubuntu-24.04-gpu-img` and `x86-linux-kernel-6.8.0-gpu`.

Build the VEGA_X86 simulator in a gem5 checkout containing the corresponding
MI355X checkpoint support. Create the initialized loader checkpoint first:

```sh
gem5/build/VEGA_X86/gem5.opt \
    gem5-resources/src/x86-mi355x-gpu-fs-smoke-checkpoint/\
create-checkpoint.py \
    --gem5-root=gem5 \
    --mode=create-loader \
    --checkpoint-output=m5out/mi355x-hip-loader
```

Then restore the loader, supply the smoke code object, complete the one-time
module and dispatch warmup, and take the final checkpoint:

```sh
gem5/build/VEGA_X86/gem5.opt \
    gem5-resources/src/x86-mi355x-gpu-fs-smoke-checkpoint/\
create-checkpoint.py \
    --gem5-root=gem5 \
    --mode=create-kernel \
    --loader-checkpoint=m5out/mi355x-hip-loader \
    --kernel-binary=gem5-resources/src/x86-mi355x-gpu-fs-smoke/\
x86-mi355x-gpu-fs-smoke \
    --checkpoint-output=m5out/x86-mi355x-gpu-fs-smoke-checkpoint
```

The loader-stage checkpoint remains useful locally: restoring it with a
different `gfx950` code object loads that object after restoration. The final
distributed checkpoint is intentionally tied to the smoke kernel so pull
request CI pays only the kernel launch and synchronization cost.

Create the upload archive with the checkpoint files at the archive root:

```sh
COPYFILE_DISABLE=1 tar -C m5out/x86-mi355x-gpu-fs-smoke-checkpoint -cf \
    x86-mi355x-gpu-fs-smoke-checkpoint-1.0.0.tar .
```

`COPYFILE_DISABLE=1` prevents macOS `tar` from adding `._*` AppleDouble files.
Those extra files would change gem5's extracted-directory checksum.

The archive and its metadata are coupled to the exact `gfx950` smoke code
object, four-CU MI355X topology, disk resource, and Linux kernel version above.
Changing any of these inputs requires a new immutable checkpoint resource
version and blob path.
