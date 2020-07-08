#!/bin/sh

# Copyright (c) 2020 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

# This file is the script that runs on the gem5 guest. This reads a file from the host via m5 readfile
# which contains the workload and the size to run. Then it resets the stats before running the workload.
# Finally, it exits the simulation after running the workload, then it copies out the result file to be checked.

cd /home/gem5/spec2006
source shrc
# Read workload file
m5 readfile > workloadfile
echo "Done reading workloads"

# The workload file should always exists
echo "Workload detected"
echo "Reset stats"
# Exit gem5 to reset the stats and switch to the detailed CPU
m5 exit

# Read the name of the workload, the size of the workload and the output folder
read -r workload size m5filespath < workloadfile
# Run the workload of the desired size once
runspec --size $size --iterations 1 --config myconfig.cfg --noreportable --nobuild $workload
# Exit gem5 to switch back to KVM
m5 exit

# Copy the SPEC result files to host
for filepath in /home/gem5/spec2006/result/*; do
    filename=$(basename $filepath)
    m5 writefile $filepath $m5filespath/$filename
done
# Exit gem5
m5 exit

