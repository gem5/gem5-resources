---
title: Pannotia Tests
tags:
    - x86
    - amdgpu
layout: default
permalink: resources/pannotia/
shortdoc: >
    Resources to build a disk image for each of the GCN3 Pannotia workloads.
---

This folder and its subfolders contain each of the 9 Pannotia benchmarks (there are 6 folders because Color, and PageRank, SSSP each have 2 versions).  All of these benchmarks have been ported from the prior CUDA and OpenCL variants to HIP, and validated on a Vega-class AMD GPU.  See each application's README for details on how to compile and run them in gem5 using the GCN3 GPU model.

## Compiling m5ops
To compile the m5ops:
```
cd gem5/util/m5
scons build/x86/out/m5
```