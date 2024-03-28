#!/bin/bash

# Copyright (c) 2020 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

echo 'Post Installation Started'

echo "Installing serial service for autologin after systemd"
mv /home/gem5/serial-getty@.service /lib/systemd/system/

echo "Installing init script"
mv /home/gem5/gem5_init.sh /sbin
mv /sbin/init /sbin/init.old
ln -s /sbin/gem5_init.sh /sbin/init

# mv /home/gem5/exit.sh /root/
# mv /home/gem5/after_boot.sh /root/

# Add init script to bashrc
echo -e "\nif [ -z \"\$AFTER_BOOT_EXECUTED\" ]; then\n   export AFTER_BOOT_EXECUTED=1\n    /home/gem5/after_boot.sh\nfi\n" >> /home/gem5/.bashrc

# Remove the motd
rm /etc/update-motd.d/*

# Build and install the m5 binary, library, and headers
echo "Building and installing m5 and libm5"
# Just get the files we need
git clone https://github.com/gem5/gem5.git --depth=1 --filter=blob:none --no-checkout --sparse --single-branch --branch=stable
pushd gem5
git sparse-checkout add util/m5
git sparse-checkout add include
git checkout
# Install the headers
cp -r include/gem5 /usr/local/include/
pushd util/m5
# Build the library and binary
scons build/x86/out/m5
cp build/x86/out/m5 /usr/local/bin/
cp build/x86/out/libm5.a /usr/local/lib/
popd
popd
mv /usr/local/bin/m5 /usr/local/bin/gem5-bridge
# Set the setuid bit on the m5 binary
chmod 4755 /usr/local/bin/gem5-bridge
chmod u+s /usr/local/bin/gem5-bridge

#create a symbolic link to the gem5 binary
ln -s /usr/local/bin/gem5-bridge /usr/local/bin/m5

rm -rf gem5
echo "Done building and installing m5 and libm5"

# You can extend this script to install your own packages here or by modifying the `x86-ubuntu.pkr.hcl` file.

# Disable network by default
echo "Disabling network by default"
mv /etc/netplan/00-installer-config.yaml /etc/netplan/00-installer-config.yaml.bak
netplan apply

echo 'Post Installation Done'
