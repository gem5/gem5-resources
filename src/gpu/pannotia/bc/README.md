---
title: Pannotia BC Test
tags:
    - x86
    - amdgpu
layout: default
permalink: resources/pannotia/bc
shortdoc: >
    Resources to build a disk image with the GCN3 Pannotia BC workload.
---

Betweenness Centrality (BC) is a graph analytics application that is part of the Pannotia benchmark suite.
It is used to calculate betweenness centrality scores for all the vertices in a graph.
The provided version is for use with the gpu-compute model of gem5.
Thus, it has been ported from the prior CUDA and OpenCL variants to HIP, and validated on a Vega-class AMD GPU.

Compiling BC, compiling the GCN3_X86/Vega_X86 versions of gem5, and running BC on gem5 is dependent on the gcn-gpu docker image, `util/dockerfiles/gcn-gpu/Dockerfile` on the [gem5 stable branch](https://gem5.googlesource.com/public/gem5/+/refs/heads/stable).

## Compiling GCN3_X86/gem5.opt

BC is a GPU application, which requires that gem5 is built with the GCN3_X86 (or Vega_X86, although this has been less heavily tested) architecture.
The test is run with the GCN3_X86 gem5 variant, compiled using the gcn-gpu docker image:

```
git clone https://gem5.googlesource.com/public/gem5
cd gem5
docker run -u $UID:$GID --volume $(pwd):$(pwd) -w $(pwd) ghcr.io/gem5/gcn-gpu:latest scons build/GCN3_X86/gem5.opt -j <num cores>
```

## Compiling BC
The Pannotia applications have been updated to use [m5ops](https://www.gem5.org/documentation/general_docs/m5ops/).

The docker command needs visibility to the gem5 repository for usage of the m5ops.
Thus we run the docker command from a directory with visibility and cd into the folder before running the make command.  
  
To compile BC assuming the gem5 and gem5-resources repositories are in your working directory:

```
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu:latest bash -c "cd gem5-resources/src/gpu/pannotia/bc ; make gem5-fusion"
```

If you use the Makefile.default file instead, the Makefile will generate code designed to run on the real GPU instead.
By default, the Makefile builds for gfx801 and gfx803, and is placed in the src/gpu/pannotia/bc/bin folder.

# Running BC on GCN3_X86/gem5.opt

Assuming gem5 and gem5-resources are in your working directory:
```
wget http://dist.gem5.org/dist/develop/datasets/pannotia/bc/1k_128k.gr
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu:latest gem5/build/GCN3_X86/gem5.opt gem5/configs/example/apu_se.py -n3 --mem-size=8GB --benchmark-root=gem5-resources/src/gpu/pannotia/bc/bin -c bc.gem5 --options="1k_128k.gr"
```

Note that the datasets from the original Pannotia suite have been uploaded to: <http://dist.gem5.org/dist/develop/datasets/pannotia>.
We recommend you start with the 1k_128k.gr input (<http://dist.gem5.org/dist/develop/datasets/pannotia/bc/1k_128k.gr>), as this is the smallest input designed to run with BC.

## Pre-built binary

A pre-built binary will be added soon.
