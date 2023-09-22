#!/bin/bash

# Copyright (c) 2023 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

cd $HOME
git clone https://github.com/gem5/gem5
cd $HOME/gem5/util/m5
/home/ubuntu/.local/bin/scons -j`nproc` build/x86/out/m5

sudo mv /home/ubuntu/serial-getty@.service /lib/systemd/system/
sudo mv /home/ubuntu/gem5/util/m5/build/x86/out/m5 /sbin
sudo ln -s /sbin/m5 /sbin/gem5
sudo mv /home/ubuntu/gem5-init.sh /root/
sudo chmod +x /root/gem5-init.sh
sudo sh -c "echo \"/root/gem5-init.sh\" >> /root/.bashrc"
