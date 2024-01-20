---
title: Building the base x86-ubuntu-gpu-ml image
authors:
    - Matthew Poremba
---

This document provides instructions to create the `x86-ubuntu-gpu-ml` disk image.
This image is an Ubuntu 22.04 image with AMD's ROCm stack, PyTorch, and TensorFlow installed.
Documentation and files here are adapted from the x86-ubuntu image by Harshil Patel and Jason Lowe-Power.

## Creating the Disk Image

The disk image is created in one step.
Run `./build.sh` in this directory to build the disk image.
This will download the packer tool, initialize packer, and build the disk image.

Building the disk image takes approximately 30 minutes depending on CPU speed and internet bandwidth.
You will see `qemu.initialize: Waiting for SSH to become available...` while the installation is running.
You can watch the installation with a VNC viewer.
See [Troubleshooting](#troubleshooting) for more information.

## Disk build output

The disk-image output directory contains the disk image `x86-ubuntu-gpu-ml`.
The Linux kernel that is running at install time is downloaded from the disk as `vmlinux-gpu-ml` in this directory.

You *must* pair this disk image with the extracted kernel when running GPU applications.

## Extending the Disk Image

You can mount the disk image to see what is inside or to test adding additional files.
Use the following command to mount the disk image:

```sh
mkdir mount
sudo mount -o loop,offset=1048576 disk-image/x86-ubuntu-gpu-ml mount
```

Once you have tested your changes, you can add the changes to `scripts/rocm-install.sh` to preserve the changes when rebuilding the disk image.

If you want to add another ML framework, see the file `scripts/rocm-install.sh`.
The methods in that file for installing PyTorch and TensorFlow are taken directly from their corresponding websites.
You could use these installation commands as examples for any other GPU related package.

## Pruning the disk image

This disk image requires approximately 43GB of space within the disk.
Some additional space is provided for users to copy data files if desired.

If you want to save space and do not need all of the packages, you may remove files from `scripts/rocm-install.sh`.
For example, if you do not require the ML frameworks, you could prune the image size down to about 16GB.

To remove PyTorch, delete or comment out the PyTorch `pip3 install` line.
To remove TensorFlow or the datasets, delete or comment out the PyTorch `pip install` line(s) for TensorFlow.

## Troubleshooting

To see what `packer` is doing, you can use the environment variable `PACKER_LOG=INFO` when running `./build.sh`.

To see what is happening while packer is running, you can connect with a VNC viewer.
The port for the VNC viewer is shown in the terminal while packer is running.

Useful documentation: <https://ubuntu.com/server/docs/install/autoinstall>
