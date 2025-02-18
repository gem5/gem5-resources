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

# Store the Ubuntu version from the command line argument
ubuntu_version="$1"

# Check if the specified Ubuntu version is valid
if [[ "$ubuntu_version" != "22.04" && "$ubuntu_version" != "24.04" ]]; then
    echo "Error: Invalid Ubuntu version '$ubuntu_version'. Must be '22.04' or '24.04'."
    exit 1
fi

# Install the needed plugins
./packer init ./packer-scripts/x86-ubuntu.pkr.hcl

# Build the image with the specified Ubuntu version
./packer build -var "ubuntu_version=${ubuntu_version}" ./packer-scripts/x86-ubuntu.pkr.hcl
