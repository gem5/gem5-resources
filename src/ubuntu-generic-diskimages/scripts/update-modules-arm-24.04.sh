#!/bin/bash

# Copyright (c) 2025 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

echo "Updating modules."

# moving modules to the correct location
mv /home/gem5/6.8.12 /lib/modules/6.8.12
depmod --quick -a 6.8.12
update-initramfs -u -k 6.8.12

echo "Modules updated."