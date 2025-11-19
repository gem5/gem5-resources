#!/bin/bash

# Copyright (c) 2025 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

echo "Installing common packages."


echo "Installing serial service override for autologin after systemd."
mkdir /etc/systemd/system/serial-getty@.service.d/
mv serial-getty@.service-override.conf /etc/systemd/system/serial-getty@.service.d/override.conf

# Installing the packages in this script instead of the user-data
# file during ubuntu autoinstall. The reason is that sometimes
# the package install fails. This method is more reliable.

echo "Installing packages required for gem5-bridge (m5) and libm5."
apt-get update
apt-get install -y scons
apt-get install -y git
apt-get install -y vim
apt-get install -y build-essential

# Add after_boot.sh to bashrc in the gem5 user account
# This will run the script after the user automatically logs in
echo "Adding after_boot.sh to the gem5 user's .bashrc."
echo -e "\nif [ -z \"\$AFTER_BOOT_EXECUTED\" ]; then\n   export AFTER_BOOT_EXECUTED=1\n    /home/gem5/after_boot.sh\nfi\n" >> /home/gem5/.bashrc

# Remove the motd
rm /etc/update-motd.d/*

echo "Installation of common packages done."