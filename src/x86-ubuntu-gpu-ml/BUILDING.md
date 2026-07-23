---
title: Building the x86-ubuntu-gpu-ml disk images
authors:
    - Matthew Poremba
---

This document provides instructions to create the `x86-ubuntu-gpu-ml` disk images.
Two images are produced from a single Packer template, both written to the common `disk-image/` directory:

- **ROCm image** (`disk-image/x86-ubuntu-rocm714`, 12 GB) — Ubuntu + ROCm 7.14
- **PyTorch image** (`disk-image/x86-ubuntu-pytorch-r72`, 24 GB) — Ubuntu + PyTorch with bundled ROCm 7.2.

Documentation and files here are adapted from the x86-ubuntu image by Harshil Patel and Jason Lowe-Power.

## Requirements

The following packages must be installed to use the `build.sh` script: `unzip`, `qemu-system-x86_64`

## Build scripts

The provisioner scripts are layered:

| Script | Purpose |
|--------|---------|
| `scripts/base-install.sh` | Common to both images: apt packages, gem5-bridge, kernel install and extraction, amdgpu discovery file setup, gem5 auto-login |
| `scripts/rocm-install.sh` | ROCm-only: registers ROCm apt repos and installs `amdrocm7.14-gfx950` |
| `scripts/pytorch-install.sh` | PyTorch-only: pip-installs `torch` and `torchvision` with bundled ROCm |

Each image build runs `base-install.sh` followed by its variant script.

## Creating the Disk Images

Run `./build.sh` from this directory.
This downloads the packer tool, initializes it, and builds the disk images.

```sh
./build.sh           # build both images sequentially (~30 min each)
./build.sh rocm      # build only the ROCm image
./build.sh pytorch   # build only the PyTorch image
```

Additional arguments are passed through to `packer build`:

```sh
./build.sh rocm -var qemu_path=/path/to/qemu-system-x86_64
PACKER_LOG=INFO ./build.sh pytorch
```

Building each image takes approximately 30 minutes depending on CPU speed and internet bandwidth.
You will see `Waiting for SSH to become available...` while the installer runs.
See [Troubleshooting](#troubleshooting) for VNC monitoring.

## Disk build output

| Image | Disk image path | Kernel path |
|-------|----------------|-------------|
| ROCm | `disk-image/x86-ubuntu-rocm714` | `vmlinux-rocm714` |
| PyTorch | `disk-image/x86-ubuntu-pytorch-r72` | `vmlinux-pytorch-r72` |

You **must** pair each disk image with its matching extracted kernel when running gem5.
The amdgpu DKMS driver is compiled against the pinned kernel at build time; mixing kernels and images will fail.

## Extending the Disk Images

Mount a disk image to inspect or test changes:

```sh
mkdir mount
sudo mount -o loop,offset=1048576 disk-image/x86-ubuntu-rocm714 mount
# or
sudo mount -o loop,offset=1048576 disk-image/x86-ubuntu-pytorch-r72 mount
```

Mount a disk image to install packages (e.g., using apt):

```sh
sudo systemd-nspawn -i disk-image/x86-ubuntu-rocm714
or
sudo systemd-nspawn -i disk-image/x86-ubuntu-pytorch-r72
```

Once you have tested your changes, add them to the appropriate script:
- Common changes go in `scripts/base-install.sh`
- ROCm-specific changes go in `scripts/rocm-install.sh`
- PyTorch or other ML framework changes go in `scripts/pytorch-install.sh`

## Changing versions

Three version knobs must stay mutually consistent:

1. The `repo.amd.com/rocm/packages-multi-arch/...` apt URL in `scripts/rocm-install.sh`
2. The `amdrocm<ver>-gfx950` package name in `scripts/rocm-install.sh`
3. The `KERNEL` variable in `scripts/base-install.sh`

When producing a new variant, also update `rocm_image_name` / `pytorch_image_name` and the
corresponding `vmlinux-*` download destination in `x86-ubuntu-gpu-ml.pkr.hcl` so new
artifacts do not overwrite existing ones.

## Troubleshooting

To see verbose packer output use `PACKER_LOG=INFO ./build.sh`.

To watch the installer live, connect a VNC viewer to the port printed in the terminal while packer is running.

Useful documentation: <https://ubuntu.com/server/docs/install/autoinstall>
