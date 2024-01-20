#!/bin/bash

# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD 3-Clause

# Read script from host and run it
/sbin/m5 readfile > script.sh
if [ -s script.sh ]; then
    # if the file is not empty, execute it and exit
    chmod +x script.sh
    ./script.sh
    /sbin/m5 exit
fi
