#!/bin/bash

# Copyright (c) 2022 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

mv /home/ubuntu/serial-getty@.service /lib/systemd/system/

mv /home/ubuntu/m5 /sbin
ln -s /sbin/m5 /sbin/gem5

mv /home/ubuntu/gem5_init.sh /root/
chmod +x /root/gem5_init.sh
echo "/root/gem5_init.sh" >> /root/.bashrc

