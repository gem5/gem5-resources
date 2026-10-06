#!/bin/bash

# Copyright (c) 2025 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

# Make sure the headers are installed to extract the kernel that DKMS
# packages will be built against.

if [ -z "$UBUNTU_VERSION" ]; then
  echo "Error: UBUNTU_VERSION environment variable is not set."
  exit 1
fi


if [ "${UBUNTU_VERSION}" = "22.04" || "${UBUNTU_VERSION}" = "24.04"]; then
    sudo apt -y install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
elif [ "${UBUNTU_VERSION}" = "26.04" ]; then
    sudo apt -y install "linux-headers-$(uname -r)"
else
    echo "Error: Unsupported UBUNTU_VERSION '${UBUNTU_VERSION}'."
    exit 1
fi

echo "Extracting linux kernel $(uname -r) to /home/gem5/vmlinux-x86-ubuntu"
sudo bash -c "/usr/src/linux-headers-$(uname -r)/scripts/extract-vmlinux /boot/vmlinuz-$(uname -r) > /home/gem5/vmlinux-x86-ubuntu"