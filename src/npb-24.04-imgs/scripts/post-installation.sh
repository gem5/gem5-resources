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
mv /etc/netplan/50-cloud-init.yaml /etc/netplan/50-cloud-init.yaml.bak
# Disable systemd service that waits for network to be online
systemctl disable systemd-networkd-wait-online.service
systemctl mask systemd-networkd-wait-online.service

netplan apply