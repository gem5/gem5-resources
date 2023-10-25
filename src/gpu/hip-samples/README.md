---
title: GCN3 HIP-Samples Tests
tags:
    - x86
    - amdgpu
layout: default
permalink: resources/hip-samples
shortdoc: >
    Resources to build a disk image with the GCN3 HIP-Sample-Applications workloads.
---

# Resource: HIP Sample Applications

The [HIP sample apps](
https://github.com/ROCm-Developer-Tools/HIP/tree/roc-1.6.0/samples) contain
applications that introduce various GPU programming concepts that are usable
in HIP.

The samples cover topics such as using and accessing different parts of GPU
memory, running multiple GPU streams, and optimization techniques for GPU code.

Certain apps aren't included due to complexities with either ROCm or Docker
(hipEvent, profiler), or due to lack of feature support in gem5 (peer2peer)

Compiling the HIP samples, compiling the GCN3_X86 gem5, and running the HIP samples on gem5 is dependent on the gcn-gpu docker image, built from the `util/dockerfiles/gcn-gpu/Dockerfile` on the [gem5 stable branch](https://gem5.googlesource.com/public/gem5/+/refs/heads/stable).

## Compilation

```
cd src/gpu/hip-samples
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu:v22-1 make
```

Individual programs can be made by specifying the name of the program

By default, the apps are built for all supported GPU types (gfx801, gfx803).
This can be changed by editing the --amdgpu-target argument in the Makefile.

## Pre-built binary

<http://dist.gem5.org/dist/v22-1/test-progs/hip-samples/2dshfl>

<http://dist.gem5.org/dist/v22-1/test-progs/hip-samples/dynamic_shared>

<http://dist.gem5.org/dist/v22-1/test-progs/hip-samples/inline_asm>

<http://dist.gem5.org/dist/v22-1/test-progs/hip-samples/MatrixTranspose>

<http://dist.gem5.org/dist/v22-1/test-progs/hip-samples/sharedMemory>

<http://dist.gem5.org/dist/v22-1/test-progs/hip-samples/shfl>

<http://dist.gem5.org/dist/v22-1/test-progs/hip-samples/stream>

<http://dist.gem5.org/dist/v22-1/test-progs/hip-samples/unroll>
