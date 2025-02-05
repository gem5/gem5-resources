#!/bin/bash

# Copyright (c) 2025 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

echo "Installing the gem5 init script in /sbin."

mv /home/gem5/gem5_init.sh /sbin/
mv /sbin/init /sbin/init.gem5.bak
ln -s /sbin/gem5_init.sh /sbin/init

echo "Done installing the gem5 init script as the init process."