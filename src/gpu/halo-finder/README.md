---
title: GCN3 HACC Test
tags:
    - x86
    - amdgpu
layout: default
permalink: resources/hacc
shortdoc: >
    Resources to build a disk image with the GCN3 HACC (halo-finder) workload.
---

# Resource: halo-finder (HACC)

[HACC](https://asc.llnl.gov/coral-2-benchmarks) is a DoE application designed to simulate the
evolution of the universe by simulating the formation of structure in collisionless fluids
under the influence of gravity. The halo-finder code can be GPU accelerated by using
the code in RCBForceTree.cxx

`src/gpu/halo-finder/src` contains the code required to build and run ForceTreeTest from `src/halo_finder` in the main HACC codebase.
`src/gpu/halo-finder/src/dfft` contains the dfft code from `src/dfft` in the main HACC codebase.

HACC can be used to test the GCN3-GPU model.

Compiling HACC, compiling the GCN3_X86 gem5, and running HACC on gem5 is dependent on the gcn-gpu docker image, built from the `util/dockerfiles/gcn-gpu/Dockerfile` on the [gem5 stable branch](https://gem5.googlesource.com/public/gem5/+/refs/heads/stable).

## Compilation and Running

halo-finder requires that certain libraries that aren't installed by default in the
GCN3 docker container provided by gem5, and that the environment is configured properly
in order to build. We provide a Dockerfile that installs those libraries and
sets the environment.

In order to test the GPU code in halo-finder, we compile and run ForceTreeTest.

To build the Docker image and the benchmark:

Note: HACC requires a number of environment variables to be set to compile and run correctly.  Our Dockerfile sets these flags appropriately for you.  This Dockerfile automatically runs when a new docker image is created, including building for both gfx801 and gfx803, which is why our instructions below recommend doing this.  If you would prefer not doing this, then you will need to pass in these environment variables using -e.

```
cd src/gpu/halo-finder
docker build -t <image_name> .
docker run --rm -v ${PWD}:${PWD} -w ${PWD}/src -u $UID:$GID <image_name> make hip/ForceTreeTest
```

The binary is built for gfx801 by default and is placed at `src/gpu/halo-finder/src/hip/ForceTreeTest`

ForceTreeTest is a GPU application, which requires that gem5 is built with the GCN3_X86 architecture.
To build GCN3_X86:
```
# Working directory is your gem5 directory
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID <image_name> scons -sQ -j$(nproc) build/GCN3_X86/gem5.opt
```

To run ForceTreeTest:
```
# Assuming gem5 and gem5-resources are in the working directory
docker run --rm -v $PWD:$PWD -w $PWD -u $UID:$GID <image_name> gem5/build/GCN3_X86/gem5.opt gem5/configs/example/apu_se.py -n3 --benchmark-root=gem5-resources/src/gpu/halo-finder/src/hip -cForceTreeTest --options="0.5 0.1 64 0.1 1 N 12 rcb"
```

## Pre-built binary

<http://dist.gem5.org/dist/v22-1/test-progs/halo-finder/ForceTreeTest>
