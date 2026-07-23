---
title: Linux x86-ubuntu images with ROCm GPU stack and ML frameworks
shortdoc: >
    Resources to build x86 Ubuntu disk images with GPU and ML stacks
authors: ["Matthew Poremba"]
---

This directory builds x86 Ubuntu disk images designed to work with the example GPU full system (GPUFS) configurations located in the gem5 repository in `configs/example/gpufs/`.
Two disk images are provided: a smaller ROCm-only image and a larger image with PyTorch.

## Disk Images

### ROCm image (`disk-image/x86-ubuntu-rocm714`)

Installs Ubuntu, ROCm 7.14, and all gem5 infrastructure.
Disk size: 12 GB.
Use this image for HIP/ROCm workloads that do not require PyTorch.

Contents:
- [ROCm](https://rocm.docs.amd.com/) 7.14: The `amdrocm7.14-gfx950` package includes:
    - [HIP](https://github.com/ROCm/HIP): hipcc LLVM compiler and HIP versions of roc libraries.
    - roc Libraries: rocBLAS, rocSPARSE, rocgdb, etc.
    - MI libraries: MIOpen, MIGraphX, etc.

### PyTorch image (`disk-image/x86-ubuntu-pytorch-r72`)

Installs Ubuntu, PyTorch with its bundled ROCm runtime, and all gem5 infrastructure.
Disk size: 24 GB.
Use this image for PyTorch machine learning workloads.

Contents:
- [PyTorch](https://pytorch.org/) 2.12.0 with bundled ROCm 7.2 runtime

## Common contents (both images)

Both images include:
- Ubuntu 26.04 server base
- gem5-bridge (`m5` binary and kernel module)
- Auto-login and workload-loading infrastructure (`run_gem5_app.sh`, `serial-getty`)
- GPU BIOS ROMs and hardware topology discovery files for MI200/MI300/MI350

The extracted `vmlinux-*` kernel **must** be paired with the disk image it was built from.

## Example gem5 commands

The disk images are intended for use with the GPUFS configurations for [MI300X](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300.html) or [MI200](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi250.html).

The following commands assume gem5-resources is cloned inside your gem5 directory.
Modify paths as needed.

**ROCm image:**
```sh
scons build/VEGA_X86/gem5.opt -j`nproc`
./build/VEGA_X86/gem5.opt configs/example/gpufs/mi300.py \
    --disk-image gem5-resources/src/x86-ubuntu-gpu-ml/disk-image/x86-ubuntu-rocm714 \
    --kernel     gem5-resources/src/x86-ubuntu-gpu-ml/vmlinux-rocm714 \
    --app $GEM5_RESOURCES/src/gpu/square/bin.default/square.default
```

**PyTorch image:**
```sh
scons build/VEGA_X86/gem5.opt -j`nproc`
./build/VEGA_X86/gem5.opt configs/example/gpufs/mi300.py \
    --disk-image gem5-resources/src/x86-ubuntu-gpu-ml/disk-image/x86-ubuntu-pytorch-r72 \
    --kernel     gem5-resources/src/x86-ubuntu-gpu-ml/vmlinux-pytorch-r72 \
    --app ./pytorch_test.py
```

The contents of `pytorch_test.py` are:

```python
#!/usr/bin/env python3

import torch
print("GPU available!") if torch.cuda.is_available() else print("No GPU available.")
x = torch.rand(5, 3)
print(x)
```

Full system stdout goes to `m5out/board.pc.com_1.device`, not the gem5 output directory.

```
GPU available!
tensor([[0.5262, 0.3074, 0.1449],
        [0.7719, 0.7705, 0.0318],
        [0.9069, 0.9333, 0.7441],
        [0.9383, 0.0746, 0.3980],
        [0.4793, 0.3785, 0.6773]])
```

**Note:** The GPU config scripts work best on a host machine with KVM, but Atomic CPU may also be used.

## Building, extending, or pruning the Disk Images

Instructions are in the companion document [BUILDING.md](BUILDING.md).
