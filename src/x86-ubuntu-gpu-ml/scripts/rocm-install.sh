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

# Occasionally connecting to github fails. Bail now instead of making a disk
# image that is not usable.
if [ ! -f /sbin/m5 ]; then
    echo "m5 util did not appear to build correctly. This disk will not be usable."
    echo "Try to build the disk again if there was a temporary error (e.g., not able to connect to github)."
    echo "For other problems create an issue at https://github.com/gem5/gem5/issues."
    exit 1
fi

# Make the script to load amdgpu module with the above module executable
chmod a+x /home/gem5/load_amdgpu.sh

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

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/6.4/ubuntu noble main" \
        | sudo tee /etc/apt/sources.list.d/amdgpu.list

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.4 noble main" \
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

# File describing the hardware topology of the GPU. Used starting with MI300X.
# Change permission so packer can copy here after disk build.
sudo touch /usr/lib/firmware/amdgpu/ip_discovery.bin
sudo chmod 777 /usr/lib/firmware/amdgpu/ip_discovery.bin


# Install a known-working version of Linux as this might change after stable
# release. Installl this after DKMS so they are rebuilt.
KERNEL=6.8.0-60-generic

sudo apt -y install "linux-image-${KERNEL}"
sudo apt -y install "linux-headers-${KERNEL}" "linux-modules-extra-${KERNEL}"

echo "Extracting linux kernel"
sudo bash -c "/usr/src/linux-headers-${KERNEL}/scripts/extract-vmlinux /boot/vmlinuz-${KERNEL} > /home/gem5/vmlinux-gpu-ml"

# Build the gem5 Linux module containing symbols missing due to outdated ACPI support in gem5
pushd /home/gem5
make -C /lib/modules/${KERNEL}/build M=${PWD}
if [ ! -f ./gem5_wmi.ko ]; then
    echo "gem5_wmi module did not appear to build correctly. This disk will not be usable."
    echo "Please post an issue at https://github.com/gem5/gem5/issues with this output log."
    exit 1
fi
popd

# Note about pip: This disk is created for the express purpose of being run in
# gem5 and is therefore effectively sandboxed enough that we can use the pip
# option --break-system-packages. If you plan to modify this disk image with
# pip packages that might conflict, it is up to you to resolve the conflicts.

# See https://pytorch.org/.
# Build: 2.3.0
# OS: Linux
# Package: Pip
# Language: Python
# Compute Platfrom: ROCm 6.2.4
sudo apt -y install pip3
pip3 install --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# For a newer version uncomment one below and remove the above install:
# Warning: Absurdly slow compared to ROCm 6.0 *in simulation*:
#pip3 install --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4
# Warning: Missing python module torch.sparse.......:
#pip3 install --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
# Warning: nightly build, may not work depending on day. Use at your own risk:
#pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4/ --break-system-packages

# Setup gem5 auto login.
mv /home/gem5/serial-getty@.service /lib/systemd/system/

echo -e "\n/home/gem5/run_gem5_app.sh\n" >> /root/.bashrc
