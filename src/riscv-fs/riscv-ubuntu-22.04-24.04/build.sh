#!/bin/bash

# Copyright (c) 2024 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

PACKER_VERSION="1.10.0"

if [ ! -f ./packer ]; then
    wget https://releases.hashicorp.com/packer/${PACKER_VERSION}/packer_${PACKER_VERSION}_linux_amd64.zip;
    unzip packer_${PACKER_VERSION}_linux_amd64.zip;
    rm packer_${PACKER_VERSION}_linux_amd64.zip;
fi

# Check if the Ubuntu version variable is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <ubuntu_version>"
    echo "Example: $0 22.04 or $0 24.04"
    exit 1
fi

ubuntu_version="$1"

if [[ "$ubuntu_version" == "22.04" ]]; then
    wget https://old-releases.ubuntu.com/releases/jammy/ubuntu-22.04.3-preinstalled-server-riscv64+unmatched.img.xz
    unxz ubuntu-22.04.3-preinstalled-server-riscv64+unmatched.img.xz
fi

if [[ "$ubuntu_version" == "24.04" ]]; then
    wget https://old-releases.ubuntu.com/releases/noble/ubuntu-24.04-preinstalled-server-riscv64.img.xz
    unxz ubuntu-24.04-preinstalled-server-riscv64.img.xz
fi


# Install the needed plugins
./packer init riscv-ubuntu.pkr.hcl

# Build the image
./packer build -var "ubuntu_version=${ubuntu_version}" riscv-ubuntu.pkr.hcl
