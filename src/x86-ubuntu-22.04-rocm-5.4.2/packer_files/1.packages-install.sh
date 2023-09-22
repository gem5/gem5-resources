#!/bin/bash

# Copyright (c) 2023 Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2023 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause


# Installing dependencies for gem5
sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y install build-essential git python3-pip cmake
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

# Disable iSCSI
# This causes booting in gem5 to wait ~2 minutes for network services to come
# up. Since we don't care about this in simulation and simulating two minutes
# of waiting is not valuable, turn it off.
sudo systemctl mask open-iscsi.service iscsid.service iscsid.socket