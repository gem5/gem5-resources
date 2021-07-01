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

Compiling square, compiling the GCN3_X86 gem5, and runing square on gem5is dependent on the gcn-gpu docker image, built from the `util/dockerfiles/gcn-gpu/Dockerfile` on the [gem5 stable branch](https://gem5.googlesource.com/public/gem5/+/refs/heads/stable).

## Compiling Square

```
cd src/gpu/square
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu make gfx8-apu
```

The compiled binary can be found in the `bin` directory.

A pre-built binary can be found at <http://dist.gem5.org/dist/develop/test-progs/square/square.o>.

## Compiling GN3_X86/gem5.opt

The test is run with the GCN3_X86 gem5 variant, compiled using the gcn-gpu docker image:

```
git clone https://gem5.googlesource.com/public/gem5
cd gem5
docker run -u $UID:$GUID --volume $(pwd):$(pwd) -w $(pwd) gcr.io/gem5-test/gcn-gpu:latest scons build/GCN3_X86/gem5.opt -j <num cores>
```

## Running Square on GCN3_X86/gem5.opt

```
docker run -u $UID:$GUID --volume $(pwd):$(pwd) -w $(pwd) gcr.io/gem5-test/gcn-gpu:latest gem5/build/GCN3_X86/gem5.opt gem5/configs/example/apu_se.py -n <num cores> -c bin/square.o
```
