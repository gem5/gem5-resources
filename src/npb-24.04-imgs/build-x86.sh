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
    echo "Example: $0 24.04 or $0 26.04"
    exit 1
fi

# Store the Ubuntu version from the command line argument
ubuntu_version="$1"

# Check if the specified Ubuntu version is valid
if [[ "$ubuntu_version" != "24.04" && "$ubuntu_version" != "26.04" ]]; then
    echo "Error: Invalid Ubuntu version '$ubuntu_version'. Must be '24.04' or '26.04'."
    exit 1
fi

# Set the base disk image file name and download URL depending on the
# Ubuntu version.
if [ "$ubuntu_version" = "24.04" ]; then
    disk_image_file="x86-ubuntu-24.04-20250515"
    disk_image_url="https://gem5resources.blob.core.windows.net/dist-gem5-org/dist/develop/images/x86/ubuntu-24-04/x86-ubuntu-24.04-20250515.gz"
elif [ "$ubuntu_version" = "26.04" ]; then
    # TODO: Fill in the actual base disk image file name and download URL
    # for Ubuntu 26.04 once it is uploaded to the gem5 resources.
    disk_image_file="x86-ubuntu-26.04-TBD"
    disk_image_url="TBD"
fi

if [ ! -f "./${disk_image_file}" ]; then
  wget "${disk_image_url}"
  gunzip "${disk_image_file}.gz"
fi

# Install the needed plugins
./packer init x86-npb.pkr.hcl

# Build the image with the specified Ubuntu version
./packer build -var "ubuntu_version=${ubuntu_version}" x86-npb.pkr.hcl
