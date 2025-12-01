#!/bin/bash

# Copyright (c) 2024 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

PACKER_VERSION="1.10.0"

if [ ! -f ./packer ]; then
    wget https://releases.hashicorp.com/packer/${PACKER_VERSION}/packer_${PACKER_VERSION}_linux_amd64.zip;
    unzip packer_${PACKER_VERSION}_linux_amd64.zip;
    rm packer_${PACKER_VERSION}_linux_amd64.zip;
fi

if [ ! -f riscv-ubuntu-24.04-20250515 ]; then
wget https://dist.gem5.org/dist/develop/images/riscv/ubuntu-24-04/riscv-ubuntu-24.04-20250515.gz
gunzip riscv-ubuntu-24.04-20250515.gz;
fi

./packer init riscv-ubuntu-24-04-gapbs.pkr.hcl
./packer build riscv-ubuntu-24-04-gapbs.pkr.hcl