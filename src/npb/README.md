---
title: NAS Parallel Benchmarks (NPB) Tests
tags:
    - x86
    - fullsystem
permalink: resources/npb
shortdoc: >
    Disk image and a gem5 configuration script to run the [NAS parallel benchmarks](https://www.nas.nasa.gov/).
author: ["Ayaz Akram"]
license: BSD-3-Clause
---

This document provides instructions to create a disk image needed to run the NPB tests with gem5 and points to an example gem5 configuration script needed to run these tests. The example script uses a pre-built disk-image.

The NAS parallel benchmarks ([NPB](https://www.nas.nasa.gov/)) are high performance computing (HPC) workloads consisting of different kernels and pseudo applications:

Kernels:
- **IS:** Integer Sort, random memory access
- **EP:** Embarrassingly Parallel
- **CG:** Conjugate Gradient, irregular memory access and communication
- **MG:** Multi-Grid on a sequence of meshes, long- and short-distance communication, memory intensive
- **FT:** discrete 3D fast Fourier Transform, all-to-all communication

Pseudo Applications:
- **BT:** Block Tri-diagonal solver
- **SP:** Scalar Penta-diagonal solver
- **LU:** Lower-Upper Gauss-Seidel solver

There are different classes (A,B,C,D,E and F) of each workload based on the input data size. Detailed discussion of the data sizes is available [here](https://www.nas.nasa.gov/publications/npb_problem_sizes.html).

We make use of a modified source of the NPB suite for these tests, which can be found in `disk-images/npb/npb-hooks`.
We have added ROI (region of interest) annotations for each benchmark which is used by gem5 to separate simulation statistics between different regions of each benchmark. gem5 magic instructions are used before and after each ROI to exit the guest and transfer control to gem5 the gem5 configuration script. This can then dump and reset stats, or switch to cpus of interest.

We assume the following directory structure while following the instructions in this README file:

```
npb/
  |___ gem5/                               # gem5 source code
  |
  |___ disk-image/
  |      |___ build.sh                     # The script downloading packer binary and building the disk image
  |      |___ shared/                      # Auxiliary files needed for disk creation
  |      |___ npb/
  |            |___ npb-image/             # Will be created once the disk is generated
  |            |      |___ npb             # The generated disk image
  |            |___ npb.json               # The Packer script to build the disk image
  |            |___ runscript.sh           # Executes a user provided script in simulated guest
  |            |___ post-installation.sh   # Moves runscript.sh to guest's .bashrc
  |            |___ npb-install.sh         # Compiles NPB inside the generated disk image
  |            |___ npb-hooks              # The NPB source (modified to function better with gem5).
  |
  |___ linux                               # Linux source and binary will live here
  |
  |___ README.md                           # This README file
```

## Disk Image

Assuming that you are in the `src/npb/` directory (the directory containing this README), first build `m5` (which is needed to create the disk image):

```sh
git clone https://gem5.googlesource.com/public/gem5
cd gem5/util/m5
scons build/x86/out/m5
```

Next,

```sh
cd disk-image
./build.sh          # the script downloading packer binary and building the disk image
```

Once this process succeeds, the created disk image can be found on `npb/npb-image/npb`.
A disk image already created following the above instructions can be found, gzipped, [here](http://dist.gem5.org/dist/v22-1/images/x86/ubuntu-18-04/npb.img.gz).

## Simulating NPB using an example script

An example script with a pre-configured system is available in the following directory within the gem5 repository:

```
gem5/configs/example/gem5_library/x86-npb-benchmarks.py
```

The example script specifies a system with the following parameters:

* A `SimpleSwitchableProcessor` (`KVM` for startup and `TIMING` for ROI execution). There are 2 CPU cores, each clocked at 3 GHz.
* 2 Level `MESI_Two_Level` cache with 32 kB L1I and L1D size, and, 256 kB L2 size. The L1 cache(s) has associativity of 8, and, the L2 cache has associativity 16. There are 2 L2 cache banks.
* The system has 3 GB `SingleChannelDDR4_2400` memory.
* The script uses `x86-linux-kernel-4.19.83` and `x86-npb`, the disk image created from following the instructions in this `README.md`.

The example script must be run with the `X86_MESI_Two_Level` binary. To build:

```sh
git clone https://gem5.googlesource.com/public/gem5
cd gem5
scons build/X86/gem5.opt -j<proc>
```
Once compiled, you may use the example config file to run the NPB benchmark programs. You would need to specify the benchmark program (`bt`, `cg`, `ep`, `ft`, `is`, `lu`, `mg`, `sp`) and the class (`A`, `B`, `C`) separately, using the following command:

```sh
# In the gem5 directory
build/X86/gem5.opt \
configs/example/gem5_library/x86-npb-benchmarks.py \
--benchmark <benchmark_program> \
--size <class_of_the_benchmark>
```

Description of the two arguments, provided in the above command are:
* **--benchmark**, which refers to one of 8 benchmark programs, provided in the NAS parallel benchmark suite. These include `bt`, `cg`, `ep`, `ft`, `is`, `lu`, `mg` and `sp`. For more information on the workloads can be found at <https://www.nas.nasa.gov/>.
* **--size**, which refers to the workload class to simulate. The classes present in the pre-built disk-image are `A`, `B`, `C` and `D`. More information regarding these classes are written in the following paragraphs.

A few important notes to keep in mind while simulating NPB using the disk-image from gem5 resources:

* The pre-built disk image has NPB executables for classes `A`, `B`, `C` and `D`.
* Classes `D` and `F` requires main memory sizes of more than 3 GB. Therefore, most of the benchmark programs for class `D` will fail to be executed properly, as our system only has 3 GB of main memory. The `X86Board` from `gem5 stdlib` is currently limited to 3 GB of memory.
* Only benchmark `ep` with class `D` works in the aforemented configuration.
* The configuration file `x86-npb-benchmarks.py` takes class input of `A`, `B` or `C`.
* More information on memory footprint for NPB is available in the paper by [Akram et al.](https://arxiv.org/abs/2010.13216)
