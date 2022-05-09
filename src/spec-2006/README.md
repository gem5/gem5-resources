---
title: SPEC 2006
tags:
    - x86
    - fullsystem
layout: default
permalink: resources/spec-2006
shortdoc: >
    Resources to build a disk image with the [SPEC 2006 workloads](https://www.spec.org/cpu2006/).
license: Proprietary SPEC License
---

This document aims to provide instructions to create a gem5-compatible disk
image containing the SPEC 2006 benchmark suite. It also demonstrates how to
simulate the SPEC CPU2006 benchmarks using an example configuration script.

## Building the Disk Image
Creating a disk-image for SPEC 2006 requires the benchmark suite ISO file.
More info about SPEC 2006 can be found <https://www.spec.org/cpu2006/>.

In this tutorial, we assume that the file `CPU2006v1.0.1.iso` contains the SPEC
benchmark suite, and we provide the scripts that are made specifically for
SPEC 2006 version 1.0.1.
Throughout the this document, the root folder is `src/spec-2006/`.
All commands should be run from this root folder.

The layout of the folder after the scripts are run is as follows,

```
spec-2006/
  |___ gem5/                                   # gem5 folder
  |
  |___ disk-image/
  |      |___ build.sh                         # the script downloading packer binary and building the disk image
  |      |___ shared/
  |      |___ spec-2006/
  |             |___ spec-2006-image/
  |             |      |___ spec-2006          # the disk image will be generated here
  |             |___ spec-2006.json            # the Packer script
  |             |___ CPU2006v1.0.1.iso         # SPEC 2006 ISO (add here)
  |
  |___ vmlinux-4.19.83                         # download link below
  |
  |___ README.md
```

First, to build `m5` (required for interactions between gem5 and the guest):

```sh
git clone https://gem5.googlesource.com/public/gem5
cd gem5
cd util/m5
scons build/x86/out/m5
```

We use [Packer](https://www.packer.io/), an open-source automated disk image
creation tool, to build the disk image.
In the root folder,

```sh
cd disk-image
./build.sh          # the script downloading packer binary and building the disk image
```

## Simulating SPEC CPU2006 using an example script

An example script with a pre-configured system is available in the following directory within the gem5 repository:

```
gem5/configs/example/gem5_library/x86-spec-cpu2006-benchmarks.py
```

The example script specifies a system with the following parameters:

* A `SimpleSwitchableProcessor` (`KVM` for startup and `TIMING` for ROI execution). There are 2 CPU cores, each clocked at 3 GHz.
* 2 Level `MESI_Two_Level` cache with 32 kB L1I and L1D size, and, 256 kB L2 size. The L1 cache(s) has associativity of 8, and, the L2 cache has associativity 16. There are 2 L2 cache banks.
* The system has 3 GB `SingleChannelDDR4_2400` memory.
* The script uses `x86-linux-kernel-4.19.83` and the disk image created from following the instructions in this `README.md`.
* The user inputs the path to the built disk image, along with the root partition.
* The script then uses `CustomResource` class to use the `spec-2006` disk-image.

The example script must be run with the `X86_MESI_Two_Level` binary. To build:

```sh
git clone https://gem5.googlesource.com/public/gem5
cd gem5
scons build/X86/gem5.opt -j<proc>
```
Once compiled, you may use the example configuration file to run the SPEC CPU2006 benchmark programs using the following command:

```sh
# In the gem5 directory
build/X86/gem5.opt \
configs/example/gem5_library/x86-spec-cpu2006-benchmarks.py \
--image <path_to_built_spec-2006_disk_image> \
--partition <root_partition_to_mount> \
--benchmark <benchmark_program> \
--size <workload_size>
```

Description of the four arguments, provided in the above command are:
* **--image** refers to the full path of the the SPEC CPU2006 disk-image, built using the instructions specified above.
* **--partition** refers to the root partition of the disk-image to mount. If the disk has no partitions, then pass `--partition ""`. Otherwise, pass an integer specifying the partition number. Set `--partition 1` if the above instructions to build the disk-image are followed.
* **--benchmark**, which refers to one of 26 benchmark programs, provided in the SPEC CPU2006 Benchmark Suite. For more information on the workloads can be found at <https://www.spec.org/cpu2006/>. The list of benchmark programs include:
  * 401.bzip2
  * 403.gcc
  * 410.bwaves
  * 416.gamess
  * 429.mcf
  * 433.milc
  * 434.zeusmp
  * 435.gromacs
  * 436.cactusADM
  * 437.leslie3d
  * 444.namd
  * 445.gobmk
  * 453.povray
  * 454.calculix
  * 456.hmmer
  * 458.sjeng
  * 459.GemsFDTD
  * 462.libquantum
  * 464.h264ref
  * 465.tonto
  * 470.lbm
  * 471.omnetpp
  * 473.astar
  * 481.wrf
  * 482.sphinx3
  * 998.specrand
  * 999.specrand
* **--size**, which refers to the workload size to simulate. Valid choices for `--size` are `test`, `train` and `ref`.

The output directory, where the simulation statistics will be redirected to, will have a new folder named `speclogs_<Day><Month><Date><Hour><Minute><Second>`. The time is of execution is appended to avoid conflicts while coping the files. The output files, generated on the disk-image in the folder `speclogs` will be copied to this aforementioned directory.
