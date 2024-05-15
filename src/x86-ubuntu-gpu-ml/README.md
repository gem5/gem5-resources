---
title: Linux x86-ubuntu image with ROCm GPU stack and ML frameworks
shortdoc: >
    Resources to build an x86 Ubuntu disk image with GPU and ML stacks
authors: ["Matthew Poremba"]
---

This disk image is designed to work with the example GPU full system (GPUFS) configurations located in the gem5 repository in configs/example/gpufs/.
The disk installs Ubuntu 22.04, the officially supported Ubuntu 22.04 version of ROCm 6.1, and popular machine learning (ML) frameworks.
Some Ubuntu configuration files are modified to automatically login as root user and load an application from the host into gem5.

## Major Contents

The disk image starts with a minimal Ubuntu server plus essential packages to build basic applications.
For GPU applications, the ROCm 6.1 version of the amdgpu DKMS driver and the `rocm` package are installed.
The DKMS driver builds against the kernel that is running at the time of install.
Therefore, the kernel extracted from this disk image *must* be paired with this disk image when running gem5.

Details of the disk contents are:
- [ROCm](https://rocm.docs.amd.com/) 6.1: The singular `rocm` package in this install includes:
    - [HIP](https://github.com/ROCm/HIP): hipcc LLVM compiler and HIP versions of roc libraries.
    - roc Libraries: rocBLAS, rocSPARSE, rocgdb, etc.
    - MI libraries: MIOpen, MIGraphX, etc.
- [PyTorch](https://pytorch.org/) 2.3.0: PyTorch is a machine learning library based on the Torch library
- [TensorFlow](https://tensorflow.org/) 2.14: a free and open-source software library for machine learning and artificial intelligence.
    - The python package tensorflow_datasets is also installed for convenience as these are used in the first TensorFlow tutorials.

## Disk Image with QEMU

The disk image is setup to automatically log in by default in gem5.
If you are using an emulator such as QEMU to work with the disk, the login information is:

- username: gem5
- password: 12345

## Example gem5 commands

The disk image is intended to be used with the GPUFS configuration for [MI200](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi250.html).
ROCm 6.1 no longer supports the Vega 10 device used in previous GPUFS generations.

The following commands assume gem5-resources is clone inside your gem5 directory.
Modify the paths as needed is that is not true:

```sh
scons build/VEGA_X86/gem5.opt -j`nproc`
./build/VEGA_X86/gem5.opt configs/example/gpufs/mi200.py --disk-image gem5-resources/src/x86-ubuntu-gpu-ml/disk-image/x86-ubuntu-gpu-ml --kernel gem5-resources/src/x86-ubuntu-gpu-ml/vmlinux-gpu-ml --app ./pytorch_test.py
```

The contents of `pytorch_test.py` are:

```python
#!/usr/bin/env python3

import torch
print("GPU available!") if torch.cuda.is_available() else print("No GPU available.")
x = torch.rand(5, 3)
print(x)
```

The simple command above does not specify a gem5 output directory, so the default output can be found in `m5out/system.pc.com_1.device`.
Recall that full system output is *not* included in the gem5 output!
Rather, it is output to a file or connected terminal:

```
GPU available!
tensor([[0.5262, 0.3074, 0.1449],
        [0.7719, 0.7705, 0.0318],
        [0.9069, 0.9333, 0.7441],
        [0.9383, 0.0746, 0.3980],
        [0.4793, 0.3785, 0.6773]])
```

**Note:** The mi200.py script work best on a host machine with KVM. The atomic CPU could also be used, however GPUFS has been optimized for KVM.

## Building, extending, or pruning the Disk Image

Instructions for these are located in the companion document [BUILDING.md](BUILDING.md).
