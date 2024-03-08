---
title: VEGA Square Test
tags:
    - x86
    - amdgpu
layout: default
permalink: resources/square
shortdoc: >
    Resources to build a disk image with the VEGA Square workload.
---

The square test is used to test the VEGA-GPU model.

Compiling square, compiling the VEGA_X86 gem5, and running square on gem5 is dependent on the gcn-gpu docker image, built from the `util/dockerfiles/gcn-gpu/Dockerfile` on the [gem5 stable branch](https://github.com/gem5/gem5).

## Compiling Square

By default, square will build for all supported GPU types (gfx900, gfx902)
```
cd src/gpu/square
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu:v24-0 make
```

The compiled binary can be found in the `bin` directory.

## Pre-built binary

A pre-built binary can be found at <https://storage.googleapis.com/dist.gem5.org/dist/v24-0/test-progs/square/square>

## Compiling VEGA_X86/gem5.opt

The test is run with the VEGA_X86 gem5 variant, compiled using the gcn-gpu docker image:

```
git clone https://github.com/gem5/gem5
cd gem5
docker run -u $UID:$GID --volume $(pwd):$(pwd) -w $(pwd) ghcr.io/gem5/gcn-gpu:v24-0 scons build/VEGA_X86/gem5.opt -j <num cores>
```

## Running Square on VEGA_X86/gem5.opt

```
docker run -u $UID:$GID --volume $(pwd):$(pwd) -w $(pwd) ghcr.io/gem5/gcn-gpu:v24-0 gem5/build/VEGA_X86/gem5.opt gem5/configs/example/apu_se.py -n 3 -c bin/square
```
