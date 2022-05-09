---
title: PARSEC
tags:
    - x86
    - fullsystem
layout: default
permalink: resources/parsec
shortdoc: >
    Resources to build a disk image with the [parsec workloads](https://parsec.cs.princeton.edu/).
author: ["Mahyar Samani"]
license: BSD-3-Clause
---

This document includes instructions on how to create an Ubuntu 18.04 disk-image with PARSEC benchmark installed. The disk-image will be compatible with the gem5 simulator. It also demostrates how tosimulate the same using an example gem5 script with a pre-configured system. The script uses a pre-built disk-image.

This is how the `src/parsec-tests/` directory will look like if all the artifacts are created correctly.

```
parsec/
  |___ gem5/                                   # gem5 folder
  |
  |___ disk-image/
  |      |___ build.sh                         # the script downloading packer binary and building the disk image
  |      |___ shared/
  |      |___ parsec/
  |             |___ parsec-image/
  |             |      |___ parsec             # the disk image will be here
  |             |___ parsec.json               # the Packer script
  |             |___ parsec-install.sh         # the script to install PARSEC
  |             |___ post-installation.sh      # the script to install m5
  |             |___ runscript.sh              # script to run each workload
  |             |___ parsec-benchmark          # the parsec benchmark suite
  |
  |___ README.md
```
## Building the disk image

In order to build the disk-image for PARSEC tests with gem5, build the m5 utility in `src/parsec-tests/` using the following:

```sh
git clone https://gem5.googlesource.com/public/gem5
cd gem5/util/m5
scons build/x86/out/m5
```

We use packer to create our disk-image. The instructions on how to install packer is shown below:

```sh
cd disk-image
./build.sh          # the script downloading packer binary and building the disk image
```

In order to build the disk-image first the script needs to be validated. Run the following command to validate `disk-image/parsec/parsec.json`.

```sh
./packer validate parsec/parsec.json
```

After the script has been successfuly validated you can create the disk-image by runnning:

```sh
./packer build parsec/parsec.json
```

You can find the disk-image in `parsec/parsec-image/parsec`.

## Simulating PARSEC using an example script

An example script with a pre-configured system is available in the following directory within the gem5 repository:

```
gem5/configs/example/gem5_library/x86-parsec-benchmarks.py
```

The example script specifies a system with the following parameters:

* A `SimpleSwitchableProcessor` (`KVM` for startup and `TIMING` for ROI execution). There are 2 CPU cores, each clocked at 3 GHz.
* 2 Level `MESI_Two_Level` cache with 32 kB L1I and L1D size, and, 256 kB L2 size. The L1 cache(s) has associativity of 8, and, the L2 cache has associativity 16. There are 2 L2 cache banks.
* The system has 3 GB `SingleChannelDDR4_2400` memory.
* The script uses `x86-linux-kernel-4.19.83` and `x86-parsec`, the disk image created from following the instructions in this `README.md`.

The example script must be run with the `X86_MESI_Two_Level` binary. To build:

```sh
git clone https://gem5.googlesource.com/public/gem5
cd gem5
scons build/X86/gem5.opt -j<proc>
```
Once compiled, you may use the example config file to run the PARSEC benchmark programs using the following command:

```sh
# In the gem5 directory
build/X86/gem5.opt \
configs/example/gem5_library/x86-parsec-benchmarks.py \
--benchmark <benchmark_program> \
--size <size> \
```

Description of the two arguments, provided in the above command are:
* **--benchmark**, which refers to one of 13 benchmark programs, provided in the PARSEC benchmark suite. These include `blackscholes`, `bodytrack`, `canneal`, `dedup`, `facesim`, `ferret`, `fluidanimate`, `freqmine`, `raytrace`, `streamcluster`, `swaptions`, `vips`, `x264`. For more information on the workloads can be found at <https://parsec.cs.princeton.edu/>.
* **--size**, which refers to the size of the workload to simulate. There are three valid choices for the same: `simsmall`, `simmedium` and `simlarge`.
