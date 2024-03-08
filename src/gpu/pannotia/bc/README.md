---
title: Pannotia BC Test
tags:
    - x86
    - amdgpu
layout: default
permalink: resources/pannotia/bc
shortdoc: >
    Resources to build a disk image with the VEGA Pannotia BC workload.
---

Betweenness Centrality (BC) is a graph analytics application that is part of the Pannotia benchmark suite.  It is used to calculate betweenness centrality scores for all the vertices in a graph.  The provided version is for use with the gpu-compute model of gem5.  Thus, it has been ported from the prior CUDA and OpenCL variants to HIP, and validated on a Vega-class AMD GPU.

Compiling BC, compiling the VEGA_X86/Vega_X86 versions of gem5, and running BC on gem5 is dependent on the gcn-gpu docker image, `util/dockerfiles/gcn-gpu/Dockerfile` on the [gem5 stable branch](https://github.com/gem5/gem5).

## Building m5ops

Pannotia requires gem5 pseudo instructions to compile. This means the m5ops library must be built in the gem5 directory first. To build m5ops, follow the instructions on the [gem5 documentation](https://www.gem5.org/documentation/general_docs/m5ops/).

## Compilation and Running

BC requires m5ops and common graph parsing libraries located in the parent directory. Docker requires that the paths to both are located within the --volume (-v) parameter and docker will not follow symlinks. The below instructions assume that gem5-resources is checked out in the gem5 directory. If that is not the case, please adapt your docker command with the correct paths. To compile BC:

```
cd src/gpu/pannotia/bc
docker run --rm -v ${PWD}/../../../../../:${PWD}/../../../../../ -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu:v24-0 make gem5-fusion
```

Alternatively from the gem5 directory, still assuming gem5-resources is checked out in the gem5 directory:

```
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu:v24-0 bash -c 'cd gem5-resources/src/gpu/pannotia/bc; make gem5-fusion'
```

If you use the Makefile.default file instead, the Makefile will generate code designed to run on the real GPU instead.  Moreover, note that Makefile.gem5-fusion requires you to set the GEM5_ROOT variable (either on the command line or by modifying the Makefile), because the Pannotia applications have been updated to use [m5ops](https://www.gem5.org/documentation/general_docs/m5ops/).  By default, the Makefile builds for gfx900 and gfx902, and is placed in the src/gpu/pannotia/bc/bin folder.

## Compiling VEGA_X86/gem5.opt

BC is a GPU application, which requires that gem5 is built with the VEGA_X86 (or Vega_X86, although this has been less heavily tested) architecture.  The test is run with the VEGA_X86 gem5 variant, compiled using the gcn-gpu docker image:

```
git clone https://github.com/gem5/gem5
cd gem5
docker run -u $UID:$GID --volume $(pwd):$(pwd) -w $(pwd) ghcr.io/gem5/gcn-gpu:latest scons build/VEGA_X86/gem5.opt -j <num cores>
```

## Running BC on VEGA_X86/gem5.opt

# Assuming gem5 and gem5-resources are in your working directory
```
wget http://dist.gem5.org/dist/develop/datasets/pannotia/bc/1k_128k.gr
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu gem5/build/VEGA_X86/gem5.opt gem5/configs/example/apu_se.py -n3 --mem-size=8GB --benchmark-root=gem5-resources/src/gpu/pannotia/bc/bin -c bc.gem5 --options="1k_128k.gr"
```

Note that the datasets from the original Pannotia suite have been uploaded to: <http://dist.gem5.org/dist/develop/datasets/pannotia>.  We recommend you start with the 1k_128k.gr input (<http://dist.gem5.org/dist/develop/datasets/pannotia/bc/1k_128k.gr>), as this is the smallest input designed to run with BC.

## Pre-built binary

<https://storage.googleapis.com/dist.gem5.org/dist/v24-0/test-progs/pannotia/bc.gem5>
