#!/bin/bash

# Copyright (c) 2024 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

PACKER_VERSION="1.10.0"

if [ ! -f ./packer ]; then
    wget https://releases.hashicorp.com/packer/${PACKER_VERSION}/packer_${PACKER_VERSION}_linux_amd64.zip;
    unzip packer_${PACKER_VERSION}_linux_amd64.zip;
    rm packer_${PACKER_VERSION}_linux_amd64.zip;
fi

if [! -f ./x86-ubuntu-24.04-20250515 ]; then
  wget https://gem5resources.blob.core.windows.net/dist-gem5-org/dist/develop/images/x86/ubuntu-24-04/x86-ubuntu-24.04-20250515.gz
  gunzip x86-ubuntu-24.04-20250515.gz
fi

# Install the needed plugins
./packer init x86-npb.pkr.hcl

# Build the image
./packer build x86-npb.pkr.hcl
