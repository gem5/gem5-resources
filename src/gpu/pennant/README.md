---
title: GCN3 PENNANT Test
tags:
    - x86
    - amdgpu
layout: default
permalink: resources/pennant
shortdoc: >
    Resources to build a disk image with the GCN3 PENNANT workload.
---

# Resource: PENNANT

PENNANT is an unstructured mesh physics mini-app designed for advanced
architecture research.  It contains mesh data structures and a few
physics algorithms adapted from the LANL rad-hydro code FLAG, and gives
a sample of the typical memory access patterns of FLAG.

## Compiling and Running

```
cd src/gpu/pennant
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu:v22-1 make
```

By default, the binary is built for gfx801 and is placed in `src/gpu/pennant/build`

pennant is a GPU application, which requires that gem5 is built with the GCN3_X86 architecture.

pennant has sample input files located at `src/gpu/pennant/test`. The following command shows how to run the sample `noh`

```
# Assuming gem5 and gem5-resources are in your working directory
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu:v22-1 gem5/build/GCN3_X86/gem5.opt gem5/configs/example/apu_se.py -n3 --benchmark-root=gem5-resources/src/gpu/pennant/build -cpennant --options="gem5-resources/src/gpu/pennant/test/noh/noh.pnt"
```

The output gets placed in `src/gpu/pennant/test/noh/`, and the file `noh.xy`
against the `noh.xy.std` file. Note: Only some tests have `.xy.std` files to
compare against, and there may be slight differences due to floating-point rounding

## Pre-built binary

<http://dist.gem5.org/dist/v22-1/test-progs/pennant/pennant>

The information from the original PENNANT README is included below.

PENNANT Description:

PENNANT is an unstructured mesh physics mini-app designed for advanced
architecture research.  It contains mesh data structures and a few
physics algorithms adapted from the LANL rad-hydro code FLAG, and gives
a sample of the typical memory access patterns of FLAG.

Further documentation can be found in the 'doc' directory of the
PENNANT distribution.


Version Log:

0.6, February 2014:
     Replaced GMV mesh reader with internal mesh generators.
     Added QCS velocity difference routine to reflect a recent
     bugfix in FLAG.  Increased size of big test problems.
     [ Master branch contained this change but CUDA branch does not:
     First MPI version.  MPI capability is working and mostly
     optimized; MPI+OpenMP is working but needs optimization. ]

0.5, May 2013:
     Further optimizations.

0.4, January 2013:
     First open-source release.  Fixed a bug in QCS and added some
     optimizations.  Added Sedov and Leblanc test problems, and some
     new input keywords to support them.

0.3, July 2012:
     Added OpenMP pragmas and point chunk processing.  Modified physics
     state arrays to be flat arrays instead of STL vectors.

0.2, June 2012:
     Added side chunk processing.  Miscellaneous minor cleanup.

0.1, March 2012:
     Initial release, internal LANL only.

