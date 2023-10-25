---
title: GCN3 LULESH Test
tags:
    - x86
    - amdgpu
layout: default
permalink: resources/lulesh
shortdoc: >
    Resources to build a disk image with the GCN3 LULESH workload.
---

# Resource: lulesh

[lulesh](https://computing.llnl.gov/projects/co-design/lulesh) is a DOE proxy
application that is used as an example of hydrodynamics modeling. The version
provided is for use with the gpu-compute model of gem5.

Compiling LULESH, compiling the GCN3_X86 gem5, and running LULESH on gem5 is dependent on the gcn-gpu docker image, built from the `util/dockerfiles/gcn-gpu/Dockerfile` on the [gem5 stable branch](https://gem5.googlesource.com/public/gem5/+/refs/heads/stable).

## Compilation and Running
```
cd src/gpu/lulesh
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu:v22-1 make
```

By default, the makefile builds for gfx801, and is placed in the `src/gpu/lulesh/bin` folder.

lulesh is a GPU application, which requires that gem5 is built with the GCN3_X86 architecture.
To build GCN3_X86:

```
# Working directory is your gem5 directory
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu:v22-1 scons -sQ -j$(nproc) build/GCN3_X86/gem5.opt
```

The following command shows how to run lulesh

Note: lulesh has two optional command-line arguments, to specify the stop time and number
of iterations. To set the arguments, add `--options="<stop_time> <num_iters>`
to the run command. The default arguments are equivalent to `--options="1.0e-2 10"`


```
# Assuming gem5 and gem5-resources are in your working directory
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu:v22-1 gem5/build/GCN3_X86/gem5.opt gem5/configs/example/apu_se.py -n3 --mem-size=8GB --benchmark-root=gem5-resources/src/gpu/lulesh/bin -clulesh
```

## Pre-built binary

<http://dist.gem5.org/dist/v21-1/test-progs/lulesh/lulesh>
