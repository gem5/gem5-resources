#!/bin/sh

# Copyright (c) 2020 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause


# This file is the script that runs on the gem5 guest. This reads a file from the host via m5 readfile
# which contains the workload if it's synthetic or real graph and the size to run.

cd /home/gem5/gapbs

# Read workload file
m5 readfile > workloadfile
echo "Done reading workloads"


# Read the name of the workload, the size of the workload
read -r workload arg size < workloadfile
./$workload $arg $size;
m5 exit