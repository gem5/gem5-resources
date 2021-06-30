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

This document provides instructions to create a GAP Benchmark Suite (GAPBS) disk image, which, along with provided configuration scripts, may be used to run GAPBS within gem5 simulations.

A pre-build disk image, for X86, can be found, gzipped, here: <http://dist.gem5.org/dist/develop/images/x86/ubuntu-18-04/gapbs.img.gz>.

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
wget https://releases.hashicorp.com/packer/1.6.0/packer_1.6.0_linux_amd64.zip   # (if packer is not already installed)
unzip packer_1.6.0_linux_amd64.zip # (if packer is not already installed)
./packer validate gapbs/gapbs.json
./packer build gapbs/gapbs.json
```

After this process succeeds, the disk image can be found on the `src/gapbs/disk-image/gapbs-image/gapbs`.

GAPBS disk image can support both real and synthetic graph inputs. The current pre-build disk image contains only one graph input which includes the New York city road map (with 733K nodes) it can be found: <http://users.diag.uniroma1.it/challenge9/download.shtml>.

To use other graphs simply copy the graph in the gapbs/ directory and add them to gapbs/gapbs.json.

## gem5 Configuration Scripts

gem5 scripts which configure the system and run the simulation are available in `configs/`.
The main script `run_gapbs.py` expects following arguments:

* **kernel** : A manditory positional argument. The path to the Linux kernel. GAPBS has been tested with [vmlinux-5.2.3](http://dist.gem5.org/dist/develop/kernels/x86/static/vmlinux-5.2.3). See `src/linux-kernel` for information on building a linux kernel for gem5.

* **disk** : A manditory positional argument. The path to the disk image.

* **cpu\_type** : A manditory positional argument. The cpu model (`kvm`, `atomic`, `simple`, `o3`).

* **num\_cpus** : A manditory positional argument. The number of cpu cores.

* **mem\_sys** : A manditory positional argument. The memory model (`classic`, `MI_example`, or `MESI_Two_Level`).

* **benchmark** : A manditory positional argument. The graph workload (`cc`, `bc`, `bfs`, `tc`, `pr`, `sssp`).

* **synthetic** : A manditory positional argument. The graph type. If synthetic graph then `1`, otherwise `0` for a real world graph.

* **graph** : A manditory positional argument. If synthetic, then the size of the graph. Otherwise the name of graph to execute.

Example usage:

```sh
<gem5 X86 binary> configs/run_gapbs.py <kernel> <disk> <cpu_type> <num_cpus> <mem_sys> <benchmark> <synthetic> <graph>
```
## Working Status

Working status of these tests for gem5-20 can be found [here](https://www.gem5.org/documentation/benchmark_status/gem5-20#gapbs-tests).
