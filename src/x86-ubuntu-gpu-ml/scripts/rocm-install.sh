#!/bin/bash

# Copyright (c) 2024 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD 3-Clause

# Install a known-working version of Linux as this might change after stable
# release.
KERNEL=6.17.0-35-generic

# Installing the packages in this script instead of the user-data
# file dueing ubuntu autoinstall. The reason is that sometimes
# the package install failes. This method is more reliable.
echo 'installing packages'
apt-get update
apt-get install -y scons
apt-get install -y git
apt-get install -y vim
apt-get install -y build-essential

# Make sure the init script is executable
chmod a+x /home/gem5/gem5_init.sh
mv /home/gem5/gem5_init.sh /sbin/
mv /sbin/init /sbin/init.gem5.bak
ln -s /sbin/gem5_init.sh /sbin/init

# Remove the motd
rm /etc/update-motd.d/*

# The following instructions were obtained from the ROCm installation guide:
# https://instinct.docs.amd.com/projects/amdgpu-docs/en/docs-30.30.4/install/
#           detailed-install/package-manager/package-manager-ubuntu.html

# Make the directory if it doesn't exist yet.
# This location is recommended by the distribution maintainers.
sudo mkdir --parents --mode=0755 /etc/apt/keyrings

# Download the key, convert the signing-key to a full
# keyring required by apt and store in the keyring directory
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
        gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/7.2.4/ubuntu noble main" \
        | sudo tee /etc/apt/sources.list.d/amdgpu.list

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2.4 noble main" \
        | sudo tee --append /etc/apt/sources.list.d/rocm.list
echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \
        | sudo tee /etc/apt/preferences.d/rocm-pin-600
sudo apt update

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


sudo apt -y install "linux-image-${KERNEL}"
sudo apt -y install "linux-headers-${KERNEL}" "linux-modules-extra-${KERNEL}"

echo "Extracting linux kernel"
sudo bash -c "/usr/src/linux-headers-${KERNEL}/scripts/extract-vmlinux /boot/vmlinuz-${KERNEL} > /home/gem5/vmlinux-gpu-ml"

# Make the discovery files writeable by packer
touch /usr/lib/firmware/amdgpu/mi300_discovery
touch /usr/lib/firmware/amdgpu/mi350_discovery

chmod 777 /usr/lib/firmware/amdgpu/mi300_discovery
chmod 777 /usr/lib/firmware/amdgpu/mi350_discovery

# Allow internet access when running this disk e.g., in systemd-nspawn
rm /etc/resolv.conf
echo "nameserver 127.0.0.53" > /etc/resolv.conf

# Note about pip: This disk is created for the express purpose of being run in
# gem5 and is therefore effectively sandboxed enough that we can use the pip
# option --break-system-packages. If you plan to modify this disk image with
# pip packages that might conflict, it is up to you to resolve the conflicts.

# See https://pytorch.org/.
# Build: 2.12.0
# OS: Linux
# Package: Pip
# Language: Python
# Compute Platfrom: ROCm 7.2
sudo apt -y install pip3
pip3 install --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2

# Try and import pytorch, exit failure if it does not exist
python3 -c "import torch"
if [ "$?" != "0" ]; then
    echo "Pytorch installation failed. See log."
    exit 1
fi

# Build the m5 util
git clone https://github.com/gem5/gem5.git --depth=1 --filter=blob:none --no-checkout --sparse --single-branch --branch=stable
pushd gem5
# Checkout just the files we need
git sparse-checkout add util/m5
git sparse-checkout add util/gem5_bridge
git sparse-checkout add include
git checkout
# Build the library and binary
pushd util/m5
scons build/x86/out/m5
cp build/x86/out/m5 /usr/local/bin/
cp build/x86/out/libm5.a /usr/local/lib/
popd # util/m5
pushd util/gem5_bridge
make build install KVERSION=${KERNEL}
depmod --quick
popd
popd

if [ ! -f /usr/local/bin/m5 ]; then
    echo "m5 util did not appear to build correctly. This disk will not be usable."
    echo "Try to build the disk again if there was a temporary error (e.g., not able to connect to github)."
    echo "For other problems create an issue at https://github.com/gem5/gem5/issues."
    exit 1
fi

mv /usr/local/bin/m5 /usr/local/bin/gem5-bridge

chmod 4755 /usr/local/bin/gem5-bridge
chmod u+s /usr/local/bin/gem5-bridge

ln -s /usr/local/bin/gem5-bridge /usr/local/bin/m5

rm -rf gem5

echo "gem5-bridge complete!"

# Setup gem5 auto login.
mv /home/gem5/serial-getty@.service /lib/systemd/system/

echo -e "\n/home/gem5/run_gem5_app.sh\n" >> /root/.bashrc
