---
title: VEGA HIP-Samples Tests
tags:
    - x86
    - amdgpu
layout: default
permalink: resources/hip-samples
shortdoc: >
    Resources to build a disk image with the VEGA HIP-Sample-Applications workloads.
---

# Resource: HIP Sample Applications

The [HIP sample apps](https://github.com/ROCm/HIP/tree/rocm-4.0.x/samples)
contain applications that introduce various GPU programming concepts that are
usable in HIP.

The samples cover topics such as using and accessing different parts of GPU
memory, running multiple GPU streams, and optimization techniques for GPU code.

Certain apps aren't included due to complexities with either ROCm or Docker
(hipEvent, profiler), or due to lack of feature support in gem5 (peer2peer)

Compiling the HIP samples, compiling the VEGA_X86 gem5, and running the HIP samples on gem5 is dependent on the gcn-gpu docker image, built from the `util/dockerfiles/gcn-gpu/Dockerfile` on the [gem5 stable branch](https://github.com/gem5/gem5).

## Compilation

```
cd src/gpu/hip-samples
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu:v24-0 make
```

Individual programs can be made by specifying the name of the program

By default, the apps are built for all supported GPU types (gfx900, gfx902).
This can be changed by editing the --amdgpu-target argument in the Makefile.

## Pre-built binary

<https://storage.googleapis.com/dist.gem5.org/dist/v24-0/test-progs/hip-samples/2dshfl>

<https://storage.googleapis.com/dist.gem5.org/dist/v24-0/test-progs/hip-samples/dynamic_shared>

<https://storage.googleapis.com/dist.gem5.org/dist/v24-0/test-progs/hip-samples/inline_asm>

<https://storage.googleapis.com/dist.gem5.org/dist/v24-0/test-progs/hip-samples/MatrixTranspose>

<https://storage.googleapis.com/dist.gem5.org/dist/v24-0/test-progs/hip-samples/sharedMemory>

<https://storage.googleapis.com/dist.gem5.org/dist/v24-0/test-progs/hip-samples/shfl>

<https://storage.googleapis.com/dist.gem5.org/dist/v24-0/test-progs/hip-samples/stream>

<https://storage.googleapis.com/dist.gem5.org/dist/v24-0/test-progs/hip-samples/unroll>
