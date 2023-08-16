#!/bin/bash

# Installing the dependencies
sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y install build-essential git python3-pip gfortran cmake ninja-build git-lfs
pip install scons --user
pip install meson --user

# Removing snapd
sudo snap remove $(snap list | awk '!/^Name|^core/ {print $1}') # not removing the core package as others depend on it
sudo snap remove $(snap list | awk '!/^Name/ {print $1}')
sudo systemctl disable snapd.service
sudo systemctl disable snapd.socket
sudo systemctl disable snapd.seeded.service
sudo apt remove --purge -y snapd
sudo apt -y autoremove
sudo apt -y autopurge

# Removing mounting /boot/efi
sudo sed -i '/\/boot\/efi/d' /etc/fstab
sudo systemctl stop boot-efi.mount
sudo systemctl disable boot-efi.mount

# Removing cloud-init
sudo touch /etc/cloud/cloud-init.disabled

# Use `systemctl mask <service/target/socket/timer>` to stop a systemd object from starting during the booting time
# Use `systemd-analyze blame` and `systemd-analyze critical-chain` to figure out the computing intensive services
# Use `grep` in /usr/lib/systemd to find which service to stop
