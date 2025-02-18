#!/bin/bash

# Copyright (c) 2024 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

echo 'Post Installation Started'

# Installing the packages in this script instead of the user-data
# file dueing ubuntu autoinstall. The reason is that sometimes
# the package install failes. This method is more reliable.
echo 'installing packages'
apt-get update
apt-get install -y scons
apt-get install -y git
apt-get install -y vim
apt-get install -y build-essential

echo "Installing serial service for autologin after systemd"
mv /home/gem5/serial-getty@.service /lib/systemd/system/

# Make sure the headers are installed to extract the kernel that DKMS
# packages will be built against.
sudo apt -y install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"

echo "Extracting linux kernel $(uname -r) to /home/gem5/vmlinux-x86-ubuntu"
sudo bash -c "/usr/src/linux-headers-$(uname -r)/scripts/extract-vmlinux /boot/vmlinuz-$(uname -r) > /home/gem5/vmlinux-x86-ubuntu"

echo "Installing the gem5 init script in /sbin"
mv /home/gem5/gem5_init.sh /sbin
mv /sbin/init /sbin/init.old
ln -s /sbin/gem5_init.sh /sbin/init

# Add after_boot.sh to bashrc in the gem5 user account
# This will run the script after the user automatically logs in
echo -e "\nif [ -z \"\$AFTER_BOOT_EXECUTED\" ]; then\n   export AFTER_BOOT_EXECUTED=1\n    /home/gem5/after_boot.sh\nfi\n" >> /home/gem5/.bashrc

# Remove the motd
rm /etc/update-motd.d/*

# Build and install the gem5-bridge (m5) binary, library, and headers
echo "Building and installing gem5-bridge (m5) and libm5"

# Ensure the ISA environment variable is set
if [ -z "$ISA" ]; then
  echo "Error: ISA environment variable is not set."
  exit 1
fi

# Just get the files we need
git clone https://github.com/gem5/gem5.git --depth=1 --filter=blob:none --no-checkout --sparse --single-branch --branch=stable
pushd gem5
# Checkout just the files we need
git sparse-checkout add util/m5
git sparse-checkout add include
git checkout
# Install the headers globally so that other benchmarks can use them
cp -r include/gem5 /usr/local/include/\

# Build the library and binary
pushd util/m5
scons build/${ISA}/out/m5
cp build/${ISA}/out/m5 /usr/local/bin/
cp build/${ISA}/out/libm5.a /usr/local/lib/
popd   # util/m5
popd   # gem5

# rename the m5 binary to gem5-bridge
mv /usr/local/bin/m5 /usr/local/bin/gem5-bridge
# Set the setuid bit on the m5 binary
chmod 4755 /usr/local/bin/gem5-bridge
chmod u+s /usr/local/bin/gem5-bridge

#create a symbolic link to the gem5 binary for backward compatibility
ln -s /usr/local/bin/gem5-bridge /usr/local/bin/m5

# delete the git repo for gem5
rm -rf gem5
echo "Done building and installing gem5-bridge (m5) and libm5"

# You can extend this script to install your own packages here or by modifying the `x86-ubuntu.pkr.hcl`
# or `arm-ubuntu.pkr.hcl` file depending on the disk you are building.

# Disable network by default
echo "Disabling network by default"
echo "See README.md for instructions on how to enable network"
if [ -f /etc/netplan/50-cloud-init.yaml ]; then
    mv /etc/netplan/50-cloud-init.yaml /etc/netplan/50-cloud-init.yaml.bak
elif [ -f /etc/netplan/00-installer-config.yaml ]; then
    mv /etc/netplan/00-installer-config.yaml /etc/netplan/00-installer-config.yaml.bak
    netplan apply
fi
# Disable systemd service that waits for network to be online
systemctl disable systemd-networkd-wait-online.service
systemctl mask systemd-networkd-wait-online.service

if [ "${ISA}" = "x86" ]; then
    echo "Disabling systemd services for x86 architecture..."

    # Disable multipathd service
    systemctl disable multipathd.service

    # Disable thermald service
    systemctl disable thermald.service

    # Disable snapd services and socket
    systemctl disable snapd.service snapd.socket

    # Disable unnecessary timers
    systemctl disable apt-daily.timer apt-daily-upgrade.timer fstrim.timer

    # Disable accounts-daemon
    systemctl disable accounts-daemon.service

    # Disable LVM monitoring service
    systemctl disable lvm2-monitor.service

    # Switch default target to multi-user (no GUI)
    systemctl set-default multi-user.target

    # Optionally disable AppArmor if not required
    systemctl disable apparmor.service snapd.apparmor.service

    echo "completed disabling systemd services for x86."
fi

echo "Post Installation Done"
