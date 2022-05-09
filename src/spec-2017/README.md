---
title: SPEC 2017
tags:
    - x86
    - fullsystem
layout: default
permalink: resources/spec-2017
shortdoc: >
    Resources to build a disk image with the [SPEC 2017 workloads](https://www.spec.org/cpu2017/).
license: Proprietary SPEC License
---

This document aims to provide instructions to create a gem5-compatible disk
image containing the SPEC 2017 benchmark suite. It also demonstrates how to
simulate the SPEC CPU2017 benchmarks using an example configuration script.

## Building the Disk Image
Creating a disk-image for SPEC 2017 requires the benchmark suite ISO file.
More info about SPEC 2017 can be found at <https://www.spec.org/cpu2017/>.

In this tutorial, we assume that the file `cpu2017-1.1.0.iso` contains the SPEC
benchmark suite, and we provide the scripts that are made specifically for
SPEC 2017 version 1.1.0.
Throughout the this document, the root folder is `src/spec-2017/`.
All commands should be run from the assumed root folder.

The layout of the folder after the scripts are run is as follows,

```
spec-2017/
  |___ gem5/                                   # gem5 folder
  |
  |___ disk-image/
  |      |___ build.sh                         # the script downloading packer binary and building the disk image
  |      |___ shared/
  |      |___ spec-2017/
  |             |___ spec-2017-image/
  |             |      |___ spec-2017          # the disk image will be generated here
  |             |___ spec-2017.json            # the Packer script
  |             |___ cpu2017-1.1.0.iso         # SPEC 2017 ISO (add here)
  |
  |___ vmlinux-4.19.83                         # Linux kernel, link to download provided below
  |
  |___ README.md

```

First, to build `m5` (required for interactions between gem5 and the system under simuations):

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

## Simulating SPEC CPU2017 using an example script

An example script with a pre-configured system is available in the following directory within the gem5 repository:

```
gem5/configs/example/gem5_library/x86-spec-cpu2017-benchmarks.py
```

The example script specifies a system with the following parameters:

* A `SimpleSwitchableProcessor` (`KVM` for startup and `TIMING` for ROI execution). There are 2 CPU cores, each clocked at 3 GHz.
* 2 Level `MESI_Two_Level` cache with 32 kB L1I and L1D size, and, 256 kB L2 size. The L1 cache(s) has associativity of 8, and, the L2 cache has associativity 16. There are 2 L2 cache banks.
* The system has 3 GB `SingleChannelDDR4_2400` memory.
* The script uses `x86-linux-kernel-4.19.83` and the disk image created from following the instructions in this `README.md`.
* The user inputs the path to the built disk image, along with the root partition.
* The script then uses `CustomResource` class to use the `spec-2017` disk-image.

The example script must be run with the `X86_MESI_Two_Level` binary. To build:

```sh
git clone https://gem5.googlesource.com/public/gem5
cd gem5
scons build/X86/gem5.opt -j<proc>
```
Once compiled, you may use the example configuration file to run the SPEC CPU2017 benchmark programs using the following command:

```sh
# In the gem5 directory
build/X86/gem5.opt \
configs/example/gem5_library/x86-spec-cpu2017-benchmarks.py \
--image <path_to_built_spec-2017_disk_image> \
--partition <root_partition_to_mount> \
--benchmark <benchmark_program> \
--size <workload_size>
```

Description of the four arguments, provided in the above command are:
* **--image** refers to the full path of the the SPEC CPU2017 disk-image, built using the instructions specified above.
* **--partition** refers to the root partition of the disk-image to mount. If the disk has no partitions, then pass `--partition ""`. Otherwise, pass an integer specifying the partition number. Set `--partition 1` if the above instructions to build the disk-image are followed.
* **--benchmark**, which refers to one of 47 benchmark programs, provided in the SPEC CPU2017 Benchmark Suite. For more information on the workloads can be found at <https://www.spec.org/cpu2017/>. The list of benchmark programs include:
  * 500.perlbench_r
  * 502.gcc_r
  * 503.bwaves_r
  * 505.mcf_r
  * 507.cactuBSSN_r
  * 508.namd_r
  * 510.parest_r
  * 511.povray_r
  * 519.lbm_r
  * 520.omnetpp_r
  * 521.wrf_r
  * 523.xalancbmk_r
  * 525.x264_r
  * 526.blender_r
  * 527.cam4_r
  * 531.deepsjeng_r
  * 538.imagick_r
  * 541.leela_r
  * 544.nab_r
  * 548.exchange2_r
  * 549.fotonik3d_r
  * 554.roms_r
  * 557.xz_r
  * 600.perlbench_s
  * 602.gcc_s
  * 603.bwaves_s
  * 605.mcf_s
  * 607.cactuBSSN_s
  * 619.lbm_s
  * 620.omnetpp_s
  * 621.wrf_s
  * 623.xalancbmk_s
  * 625.x264_s
  * 627.cam4_s
  * 628.pop2_s
  * 631.deepsjeng_s
  * 638.imagick_s
  * 641.leela_s
  * 644.nab_s
  * 648.exchange2_s
  * 649.fotonik3d_s
  * 654.roms_s
  * 657.xz_s
  * 996.specrand_fs
  * 997.specrand_fr
  * 998.specrand_is
  * 999.specrand_ir
* **--size**, which refers to the workload size to simulate. Valid choices for `--size` are `test`, `train` and `ref`.

The output directory, where the simulation statistics will be redirected to, will have a new folder named `speclogs_<Day><Month><Date><Hour><Minute><Second>`. The time is of execution is appended to avoid conflicts while coping the files. The output files, generated on the disk-image in the folder `speclogs` will be copied to this aforementioned directory.
