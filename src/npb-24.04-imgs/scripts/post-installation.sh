#!/bin/sh

# Copyright (c) 2020 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

# install build-essential (gcc and g++ included) and gfortran

#Compile NPB

apt-get install -y gfortran

cd /home/gem5/NPB3.4-OMP/

mkdir bin
make clean
make suite M5_ANNOTATION=1

echo "Disabling network by default"
echo "See README.md for instructions on how to enable network"

# Ensure the UBUNTU_VERSION environment variable is set
if [ -z "$UBUNTU_VERSION" ]; then
  echo "Error: UBUNTU_VERSION environment variable is not set."
  exit 1
fi

# The netplan config file name depends on the Ubuntu version.
if [ "${UBUNTU_VERSION}" = "24.04" ]; then
    mv /etc/netplan/50-cloud-init.yaml /etc/netplan/50-cloud-init.yaml.bak
elif [ "${UBUNTU_VERSION}" = "26.04" ]; then
    mv /etc/netplan/00-installer-config.yaml /etc/netplan/00-installer-config.yaml.bak
else
    echo "Error: Unsupported UBUNTU_VERSION '${UBUNTU_VERSION}'."
    exit 1
fi

# Disable systemd service that waits for network to be online
systemctl disable systemd-networkd-wait-online.service
systemctl mask systemd-networkd-wait-online.service
