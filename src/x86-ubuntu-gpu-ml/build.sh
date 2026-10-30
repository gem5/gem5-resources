#!/bin/bash

# Copyright (c) 2024 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

PACKER_VERSION="1.10.0"

if [ ! -f ./packer ]; then
    wget https://releases.hashicorp.com/packer/${PACKER_VERSION}/packer_${PACKER_VERSION}_linux_amd64.zip;
    unzip packer_${PACKER_VERSION}_linux_amd64.zip;
    rm packer_${PACKER_VERSION}_linux_amd64.zip;
fi

# Install the needed plugins
./packer init x86-ubuntu-gpu-ml.pkr.hcl

# Optional first argument selects which image to build:
#   ./build.sh rocm      -- build only the ROCm image  (disk-image-rocm/)
#   ./build.sh pytorch   -- build only the PyTorch image (disk-image-pytorch/)
#   ./build.sh           -- build both images sequentially
# Any remaining arguments are passed through to packer build (e.g. -var qemu_path=...).
ONLY_ARG=""
if [ "$1" = "rocm" ]; then
    ONLY_ARG="-only=qemu.rocm"
    shift
elif [ "$1" = "pytorch" ]; then
    ONLY_ARG="-only=qemu.pytorch"
    shift
fi

./packer build -parallel-builds=1 $ONLY_ARG "$@" x86-ubuntu-gpu-ml.pkr.hcl
