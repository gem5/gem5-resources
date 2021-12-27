#!/bin/bash

# Copyright (c) 2020 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

echo 'Post Installation Started'

mv /home/gem5/serial-getty@.service /lib/systemd/system/

mv /home/gem5/m5 /sbin
ln -s /sbin/m5 /sbin/gem5

mv /home/gem5/exit.sh /root/
mv /home/gem5/gem5_init.sh /root/

# Add init script to bashrc
echo "/root/gem5_init.sh" >> /root/.bashrc

echo 'Post Installation Done'
