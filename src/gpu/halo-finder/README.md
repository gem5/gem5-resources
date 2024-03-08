---
title: VEGA HACC Test
tags:
    - x86
    - amdgpu
layout: default
permalink: resources/hacc
shortdoc: >
    Resources to build a disk image with the VEGA HACC (halo-finder) workload.
---

# Resource: halo-finder (HACC)

[HACC](https://asc.llnl.gov/coral-2-benchmarks) is a DoE application designed to simulate the
evolution of the universe by simulating the formation of structure in collisionless fluids
under the influence of gravity. The halo-finder code can be GPU accelerated by using
the code in RCBForceTree.cxx

`src/gpu/halo-finder/src` contains the code required to build and run ForceTreeTest from `src/halo_finder` in the main HACC codebase.
`src/gpu/halo-finder/src/dfft` contains the dfft code from `src/dfft` in the main HACC codebase.

HACC can be used to test the VEGA-GPU model.

Compiling HACC, compiling the VEGA_X86 gem5, and running HACC on gem5 is dependent on the gcn-gpu docker image, built from the `util/dockerfiles/gcn-gpu/Dockerfile` on the [gem5 stable branch](https://github.com/gem5/gem5).

## Compilation and Running

In order to test the GPU code in halo-finder, we compile and run ForceTreeTest.

Note: HACC requires a number of environment variables to be set to compile and run correctly.  Our Dockerfile sets these flags appropriately for you, including building for both gfx900 and gfx902.  If you would prefer not doing this, then you will need to pass in these environment variables using -e.

```
cd src/gpu/halo-finder
docker run --rm -v ${PWD}:${PWD} -w ${PWD}/src -u $UID:$GID ghcr.io/gem5/gcn-gpu:v24-0 make hip/ForceTreeTest
```

The binary is built for gfx900 and gfx902 by default and is placed at `src/gpu/halo-finder/src/hip/ForceTreeTest`

ForceTreeTest is a GPU application, which requires that gem5 is built with the VEGA_X86 architecture.
To build VEGA_X86:
```
# Working directory is your gem5 directory
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu:v24-0 scons -sQ -j$(nproc) build/VEGA_X86/gem5.opt
```

To run ForceTreeTest:
```
# Assuming gem5 and gem5-resources are in the working directory
docker run --rm -v $PWD:$PWD -w $PWD -u $UID:$GID ghcr.io/gem5/gcn-gpu:v24-0 gem5/build/VEGA_X86/gem5.opt gem5/configs/example/apu_se.py -n3 --benchmark-root=gem5-resources/src/gpu/halo-finder/src/hip -c ForceTreeTest --options="0.5 0.1 64 0.1 1 N 12 rcb"
```

## Pre-built binary

<https://storage.googleapis.com/dist.gem5.org/dist/v24-0/test-progs/halo-finder/ForceTreeTest>
