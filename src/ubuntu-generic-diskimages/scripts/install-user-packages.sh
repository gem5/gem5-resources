#!/bin/bash

# Copyright (c) 2025 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

echo "Installing user packages..."

# Put your package installation commands here
# NOTE: Since we use packer we can not interact with the process of apt-get
# upgrade/install, I recommend that you add `Dpkg::Options::="--force-confnew"`
# to your commands here. See example below on how to update, upgrade, and
# install packages for cmake, python3-pip, gfortran, and openmpi, ...
# sudo apt-get -y update
# sudo apt-get -y -o Dpkg::Options::="--force-confnew" upgrade
# sudo apt-get -y -o Dpkg::Options::="--force-confnew" install \
# python3-pip gfortran cmake openmpi-bin libopenmpi-dev

echo "Done installing user packages."