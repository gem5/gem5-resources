#!/bin/bash

# Copyright (c) 2025 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

echo "Updating modules."

# moving modules to the correct location
mv /home/gem5/5.15.168 /lib/modules/5.15.168
depmod --quick -a 5.15.168
update-initramfs -u -k 5.15.168

echo "Modules updated."