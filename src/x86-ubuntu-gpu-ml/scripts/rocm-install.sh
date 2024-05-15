#!/bin/bash

# Copyright (c) 2024 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD 3-Clause

# Installing the packages in this script instead of the user-data
# file dueing ubuntu autoinstall. The reason is that sometimes
# the package install failes. This method is more reliable.
echo 'installing packages'
apt-get update
apt-get install -y scons
apt-get install -y git
apt-get install -y vim
apt-get install -y build-essential

# Remove the motd
rm /etc/update-motd.d/*

# Build the m5 util
git clone https://github.com/gem5/gem5.git --depth=1 --filter=blob:none --no-checkout --sparse --single-branch --branch=stable
pushd gem5
# Checkout just the files we need
git sparse-checkout add util/m5
git sparse-checkout add include
git checkout
# Build the library and binary
pushd util/m5
scons build/x86/out/m5
cp build/x86/out/m5 /sbin/m5
popd
popd
rm -rf gem5


# Make sure the headers are installed to extract the kernel that DKMS
# packages will be built against.
sudo apt -y install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"

echo "Extracting linux kernel"
sudo bash -c "/usr/src/linux-headers-$(uname -r)/scripts/extract-vmlinux /boot/vmlinuz-$(uname -r) > /home/gem5/vmlinux-gpu-ml"


# The following instructions were obtained from the ROCm installation guide:
# https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/
#     native-install/ubuntu.html

# Make the directory if it doesn't exist yet.
# This location is recommended by the distribution maintainers.
sudo mkdir --parents --mode=0755 /etc/apt/keyrings

# Download the key, convert the signing-key to a full
# keyring required by apt and store in the keyring directory
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
        gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/6.1/ubuntu jammy main" \
        | sudo tee /etc/apt/sources.list.d/amdgpu.list

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.1 jammy main" \
        | sudo tee --append /etc/apt/sources.list.d/rocm.list
echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \
        | sudo tee /etc/apt/preferences.d/rocm-pin-600
sudo apt update

sudo apt -y install amdgpu-dkms
sudo apt -y install rocm
sudo apt -y install cmake

# Make directory for GPU BIOS. These are placed in /root for compatibility with
# the legacy GPUFS configs.
sudo mkdir -p /root/roms
sudo chmod 777 /root
sudo chmod 777 /root/roms


# See https://pytorch.org/ . At the time of writing the selector was:
# Build: 2.3.0
# OS: Linux
# Package: Pip
# Language: Python
# Compute Platfrom: ROCm 6.0 (Note: Latest ROCm when this file was last modified)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# See https://pypi.org/project/tensorflow-rocm/#description
# Datasets are also installed as gem5 has no internet connection.
pip install tensorflow-rocm==2.14.0.600
pip install tensorflow-datasets==4.9.4


# Setup gem5 auto login.
mv /home/gem5/serial-getty@.service /lib/systemd/system/

echo -e "\n/home/gem5/run_gem5_app.sh\n" >> /root/.bashrc
