---
title: GAP Benchmark Suite (GAPBS) tests
tags:
    - x86
    - fullsystem
permalink: resources/gapbs
shortdoc: >
    This resource implementes the [GAP benchmark suite](http://gap.cs.berkeley.edu/benchmark.html).
author: ["Marjan Fariborz"]
license: BSD-3-Clause
---

This document provides instructions to create a GAP Benchmark Suite (GAPBS) disk image, which, along with an example script, may be used to run GAPBS within gem5 simulations. The example script uses a pre-built disk-image.

A pre-built disk image, for X86, can be found, gzipped, here: <http://dist.gem5.org/dist/v22-1/images/x86/ubuntu-18-04/gapbs.img.gz>.

## Building the Disk Image

Assuming that you are in the `src/gapbs/` directory, first create `m5` (which is needed to create the disk image):

```sh
git clone https://gem5.googlesource.com/public/gem5
cd gem5/util/m5
scons build/x86/out/m5
```

To create the disk image you need to add the packer binary in the disk-image directory:

```sh
cd disk-image/
./build.sh          # the script downloading packer binary and building the disk image
```

After this process succeeds, the disk image can be found on the `src/gapbs/disk-image/gapbs-image/gapbs`.

GAPBS disk image can support both real and synthetic graph inputs. The current pre-built disk image contains only one graph input which includes the New York city road map (with 733K nodes) it can be found: <http://users.diag.uniroma1.it/challenge9/download.shtml>.

To use other graphs simply copy the graph in the gapbs/ directory and add them to gapbs/gapbs.json.

## Simulating GAPBS using an example script

An example script with a pre-configured system is available in the following directory within the gem5 repository:

```
gem5/configs/example/gem5_library/x86-gapbs-benchmarks.py
```

The example script specifies a system with the following parameters:

* A `SimpleSwitchableProcessor` (`KVM` for startup and `TIMING` for ROI execution). There are 2 CPU cores, each clocked at 3 GHz.
* 2 Level `MESI_Two_Level` cache with 32 kB L1I and L1D size, and, 256 kB L2 size. The L1 cache(s) has associativity of 8, and, the L2 cache has associativity 16. There are 2 L2 cache banks.
* The system has 3 GB `SingleChannelDDR4_2400` memory.
* The script uses `x86-linux-kernel-4.19.83` and `x86-gapbs`, the disk image created from following the instructions in this `README.md`.

The example script must be run with the `X86` binary. To build:

```sh
git clone https://gem5.googlesource.com/public/gem5
cd gem5
scons build/X86/gem5.opt -j<proc>
```
Once compiled, you may use the example config file to run the GAPBS benchmark programs using the following command:

```sh
# In the gem5 directory
build/X86/gem5.opt \
configs/example/gem5_library/x86-gapbs-benchmarks.py \
--benchmark <benchmark_program> \
--synthetic <synthetic> \
--size <size_or_graph_name>
```

Description of the three arguments, provided in the above command are:
* **--benchmark**, which refers to one of 5 benchmark programs, provided in the GAP Benchmark Suite. These include `cc`, `bc`, `tc`, `pr` and `bfs`. For more information on the workloads can be found at <http://gap.cs.berkeley.edu/benchmark.html>.
* **--synthetic** refers whether to use a synthetic or a real graph. It accepts a boolean value.
* **--size**, which refers to either the size of a synthetic graph from 1 to 16 nodes, or, a real graph. The real graph included in the pre-built disk-image is `USA-road-d.NY.gr`. Note that `--synthetic True` and `--size USA-road-d.NY.gr` cannot be combined, and, vice versa for real graphs.
