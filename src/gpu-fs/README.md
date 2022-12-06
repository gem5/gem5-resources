---
title: ROCm 4.2
tags:
    - x86
    - fullsystem
layout: default
permalink: resources/rocm42
shortdoc: >
    Resources to build a disk image with [AMD ROCm](https://rocmdocs.amd.com/).
author: ["Matthew Poremba"]
license: BSD-3-Clause
---

This document includes instructions on how to create an Ubuntu 18.04 disk-image with ROCm 4.2 installed. The disk-image will be compatible with the gem5 simulator. It also demonstrates how to simulate the same using an example gem5 script with a pre-configured system.

```
## Building the disk image

In order to build the disk-image for ROCm 4.2 with gem5, build the m5 utility in `src/gpu-fs/` using the following:

```sh
git clone https://gem5.googlesource.com/public/gem5
cd gem5/util/m5
scons build/x86/out/m5
cp build/x86/out/m5 /path/to/gem5-resources/src/gpu-fs/
```

We use packer to create our disk-image. The instructions on how to install packer is shown below:

```sh
cd disk-image
./build.sh          # the script downloading packer binary and building the disk image
```

You can find the disk-image in `disk-image/rocm42/rocm42-image/rocm42`.

## Simulating GPU full system using an example script

An example script with a pre-configured system is available in the following directory within the gem5 repository:

```
gem5/configs/example/gpufs/vega10_kvm.py
```

The example script specifies a system with the following parameters:

* A single 'KVM' CPU with the `MOESI_AMD_Base` protocol. The CPU and CPU cache configurations are largely irrelevant for GPU simulation.
* 2 Level `GPU_VIPER` cache with 32 kB L1I (SQC), 16 kB per-CU L1D (TCP), and 256 kB L2 (TCC).
* The system has 3 GB of --mem-type memory for CPU and 16 GB of --mem-type memory for GPU.

The example script must be run with the `VEGA_X86` binary. To build:

```sh
git clone https://gem5.googlesource.com/public/gem5
cd gem5
scons build/VEGA_X86/gem5.opt -j<proc>
```

Once compiled, you may use one of the example config scripts to run a GPU application on the simulated machine:

```sh
gem5/configs/example/gpufs/hip_samples.py
gem5/configs/example/gpufs/hip_cookbook.py
gem5/configs/example/gpufs/hip_rodinia.py
```

These scripts can be run as follows pointing to the disk image created above and the provided kernel and GPU trace in gem5-resources. For example:

```
build/VEGA_X86/gem5.opt configs/example/gpufs/hip_samples.py --disk-image /path/to/gem5-resources/src/gpu-fs/disk-image/rocm42/rocm42-image/rocm42 --kernel /path/to/gem5-resources/src/gpu-fs/vmlinux-5.4.0-105-generic --gpu-mmio-trace /path/to/gem5-resources/src/gpu-fs/vega_mmio.log --app PrefixSum
build/VEGA_X86/gem5.opt configs/example/gpufs/hip_cookbook.py --disk-image /path/to/gem5-resources/src/gpu-fs/disk-image/rocm42/rocm42-image/rocm42 --kernel /path/to/gem5-resources/src/gpu-fs/vmlinux-5.4.0-105-generic --gpu-mmio-trace /path/to/gem5-resources/src/gpu-fs/vega_mmio.log --app 4_shfl
build/VEGA_X86/gem5.opt configs/example/gpufs/hip_rodinia.py --disk-image /path/to/gem5-resources/src/gpu-fs/disk-image/rocm42/rocm42-image/rocm42 --kernel /path/to/gem5-resources/src/gpu-fs/vmlinux-5.4.0-105-generic --gpu-mmio-trace /path/to/gem5-resources/src/gpu-fs/vega_mmio.log --app nn
```

You can obtain the `vmlinux-5.4.0-105-generic` kernel using the following path from gem5-resources: `wget --no-check-certificate https://dist.gem5.org/dist/v22-1/kernels/x86/static/vmlinux-5.4.0-105-generic`

It is sometimes useful to build your own application and run in gem5. A docker is provided to allow users to build applications without needing to install ROCm locally. A pre-built docker image is available on gcr.io. This image can be pulled then used to build as follows:

```sh
docker pull gcr.io/gem5-test/gpu-fs:latest
cd /path/to/gem5-resources/src/gpu/square
docker run --rm -v ${PWD}:${PWD} -w ${PWD} gcr.io/gem5-test/gpu-fs:latest bash -c 'make clean; HCC_AMDGPU_TARGET=gfx900 make'
```

Currently only Vega (gfx900) is supported for full system GPU simulation in gem5. It is therefore required to tell the compiler to build for this ISA using the HCC_AMDGPU_TARGET environment variable. Otherwise, the command to build the application is the same as if you were building locally.

The build docker can also be built from the gem5 directory:

```sh
cd gem5/util/dockerfiles/gpu-fs/
docker build -t rocm42-build .
cd /path/to/gem5-resources/src/gpu/square
docker run --rm -v ${PWD}:${PWD} -w ${PWD} rocm42-build bash -c 'make clean; HCC_AMDGPU_TARGET=gfx900 make'
```

The application can then be run using the vega10_kvm.py example script. There are two arguments available in the example script:
* **--app**, which copies the pre-built application from the host into the simulated gem5 environment and runs the command with the options given by **--opts**.
* **--opts**, which passes options to the application being run

Below is an example using the square application which was built in above using the docker image:

```sh
build/VEGA_X86/gem5.opt configs/example/gpufs/vega10_kvm.py --disk-image /path/to/gem5-resources/src/gpu-fs/disk-image/rocm42/rocm42-image/rocm42 --kernel /path/to/gem5-resources/src/gpu-fs/vmlinux-5.4.0-105-generic --gpu-mmio-trace /path/to/gem5-resources/src/gpu-fs/vega_mmio.log --app /path/to/gem5-resources/src/gpu/square/bin/square
```

## Working Status

The known working ROCm 4.2 applications for gem5-22 is below for each of the example config scripts and other gem5-resources. Missing applications either do not work or have not been fully tested:
* **hip_samples.py**: BinomialOption, BitonicSort, FastWalshTransform, FloydWarshall, Histogram, PrefixSum, RecursiveGaussian, SimpleConvolution, dct, dwtHaar1D
* **hip_cookbook.py**: 0_MatrixTranspose, 3_shared_memory, 4_shfl, 5_2dshfl, 6_dynamic_shared, 7_streams, 9_unroll, 10_inline_asm, 11_texture_driver, 13_occupancy, 14_gpu_arch, 15_static_library
* **hip_rodinia.py**: bfs, nn
* **gem5-resources**: heterosync (lfTreeBarrUniq 10 16 4), pagerank, fw

The following features are known not to work:
* Dynamic scratch space allocation (gem5 will fatal)
* HIP events (simulation will hang/never finish).

## Troubleshooting

- `perf_event_paranoid` error when running a FS simulation:

```tx
This error may be caused by a too restrictive setting   in the file
'/proc/sys/kernel/perf_event_paranoid'  The default value was changed to 2
in kernel 4.6  A value greater than 1 prevents gem5 from making  the
syscall to perf_event_open.

You need to run something like the following (as root).

# echo -1 > /proc/sys/kernel/perf_event_paranoid
```

- If you encounter `Qemu stderr: qemu-system-x86_64: failed to initialize KVM: Permission denied`, the issue is likely related to permission on /dev/kvm