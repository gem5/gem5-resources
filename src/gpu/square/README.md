---
title: GCN3 Square Test
tags:
    - x86
    - amdgpu
layout: default
permalink: resources/square
shortdoc: >
    Resources to build a disk image with the GCN3 Square workload.
---

The square test is used to test the GCN3-GPU model.

Compiling square, compiling the GCN3_X86 gem5, and running square on gem5 is dependent on the gcn-gpu docker image, built from the `util/dockerfiles/gcn-gpu/Dockerfile` on the [gem5 stable branch](https://gem5.googlesource.com/public/gem5/+/refs/heads/stable).

## Compiling Square

By default, square will build for all supported GPU types (gfx801, gfx803)
```
cd src/gpu/square
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu:v22-1 make
```

The compiled binary can be found in the `bin` directory.

## Pre-built binary

A pre-built binary can be found at <http://dist.gem5.org/dist/v22-1/test-progs/square/square>.

## Compiling GCN3_X86/gem5.opt

The test is run with the GCN3_X86 gem5 variant, compiled using the gcn-gpu docker image:

```
git clone https://gem5.googlesource.com/public/gem5
cd gem5
docker run -u $UID:$GID --volume $(pwd):$(pwd) -w $(pwd) ghcr.io/gem5/gcn-gpu:v22-1 scons build/GCN3_X86/gem5.opt -j <num cores>
```

## Running Square on GCN3_X86/gem5.opt

```
docker run -u $UID:$GID --volume $(pwd):$(pwd) -w $(pwd) ghcr.io/gem5/gcn-gpu:v22-1 gem5/build/GCN3_X86/gem5.opt gem5/configs/example/apu_se.py -n 3 -c bin/square
```
