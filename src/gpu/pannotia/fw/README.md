---
title: Pannotia FW Test
tags:
    - x86
    - amdgpu
layout: default
permalink: resources/pannotia/fw
shortdoc: >
    Resources to build a disk image with the GCN3 Pannotia FW workload.
---

Floyd-Warshall (FW) is a graph analytics application that is part of the Pannotia benchmark suite.  It is a classical dynamic-programming algorithm designed to solve the all-pairs shortest path (APSP) problem.  The provided version is for use with the gpu-compute model of gem5.  Thus, it has been ported from the prior CUDA and OpenCL variants to HIP, and validated on a Vega-class AMD GPU.

Compiling FW, compiling the GCN3_X86/Vega_X86 versions of gem5, and running FW on gem5 is dependent on the gcn-gpu docker image, `util/dockerfiles/gcn-gpu/Dockerfile` on the [gem5 stable branch](https://gem5.googlesource.com/public/gem5/+/refs/heads/stable).

## Compilation and Running

To compile FW:

```
cd src/gpu/pannotia/fw
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu make gem5-fusion; make default
```

If you use the Makefile.default file instead, the Makefile will generate code designed to run on the real GPU instead.  Moreover, note that Makefile.gem5-fusion requires you to set the GEM5_ROOT variable (either on the command line or by modifying the Makefile), because the Pannotia applications have been updated to use [m5ops](https://www.gem5.org/documentation/general_docs/m5ops/).  By default, the Makefile builds for gfx801 and gfx803, and is placed in the src/gpu/pannotia/fw/bin folder. FW can be run on a non-mmapped input file, used to generate an mmapped input file, or run on an mmapped input file. To run FW using an mmapped input file, you must generate it first. An input file can be reused until it is overwritten by another file generation.  

## Compiling GCN3_X86/gem5.opt

FW is a GPU application, which requires that gem5 is built with the GCN3_X86 (or Vega_X86, although this has been less heavily tested) architecture.   The test is run with the GCN3_X86 gem5 variant, compiled using the gcn-gpu docker image:

```
git clone https://gem5.googlesource.com/public/gem5
cd gem5
docker run -u $UID:$GID --volume $(pwd):$(pwd) -w $(pwd) ghcr.io/gem5/gcn-gpu:latest scons build/GCN3_X86/gem5.opt -j <num cores>
```

## Running FW on GCN3_X86/gem5.opt

# Assuming gem5 and gem5-resources are in your working directory

# Run FW without using a mmapped input file

```
wget http://dist.gem5.org/dist/develop/datasets/pannotia/bc/1k_128k.gr
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu gem5/build/GCN3_X86/gem5.opt gem5/configs/example/apu_se.py -n3 --mem-size=8GB --benchmark-root=gem5-resources/src/gpu/pannotia/fw/bin -c fw_hip.gem5 --options="-f 1k_128k.gr -m default"
```

# Generate a mmapped input file

# We recommend running mmap generation on the actual CPU instead of simulating it.

```
wget http://dist.gem5.org/dist/develop/datasets/pannotia/bc/1k_128k.gr
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu bash -c "./gem5-resources/src/gpu/pannotia/fw/bin/fw_hip -f ./gem5-resources/src/gpu/pannotia/fw/1k_128k.gr -m generate"
```

# Run FW using a mmapped input file

To run FW using an mmapped input file, you must generate it first. An input file can be reused until it is overwritten by another file generation.  

```
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu gem5/build/GCN3_X86/gem5.opt gem5/configs/example/apu_se.py -n3 --mem-size=8GB --benchmark-root=gem5-resources/src/gpu/pannotia/fw/bin -c fw_hip.gem5 --options="-f 1k_128k.gr -m usemmap"
```
                  
Note that the datasets from the original Pannotia suite have been uploaded to: <http://dist.gem5.org/dist/develop/datasets/pannotia>.  We recommend you start with the 1k_128k.gr input (<http://dist.gem5.org/dist/develop/datasets/pannotia/fw/1k_128k.gr>), as this is the smallest input that can be run with FW.  Note that 1k_128k is not designed for FW specifically though -- the above link has larger graphs designed to run with FW that you should consider using for larger experiments.

## Pre-built binary

A pre-built binary will be added soon.
