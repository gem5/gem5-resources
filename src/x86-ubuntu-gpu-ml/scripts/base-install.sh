#!/bin/bash

# Copyright (c) 2024 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD 3-Clause

# Install a known-working version of Linux as this might change after stable
# release.
KERNEL=7.0.0-14-generic

# Installing the packages in this script instead of the user-data
# file during ubuntu autoinstall. The reason is that sometimes
# the package install fails. This method is more reliable.
echo 'installing packages'
sudo apt update
sudo apt install -y scons git vim build-essential

# Make sure the init script is executable
chmod a+x /home/gem5/gem5_init.sh
mv /home/gem5/gem5_init.sh /sbin/
mv /sbin/init /sbin/init.gem5.bak
ln -s /sbin/gem5_init.sh /sbin/init

# Remove the motd
rm /etc/update-motd.d/*

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
sudo apt -y install "linux-headers-${KERNEL}"

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

# Build the m5 util
git clone https://github.com/abmerop/gem5 --depth=1 --filter=blob:none --no-checkout --sparse --single-branch --branch=temp-26.04-fix
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
