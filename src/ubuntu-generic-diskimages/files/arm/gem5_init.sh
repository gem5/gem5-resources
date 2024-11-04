#!/bin/bash

# Copyright (c) 2024 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

# This file is a custom init script for gem5.
# It will either boot with or without systemd depending on the kernel
# command line options. If the kernel command line contains "no_systemd",
# then systemd will not be started and the script will drop to a shell.

# mount /proc and /sys so that we can read the command line
mount -t proc /proc /proc
mount -t sysfs /sys /sys
# Read /proc/cmdline and parse options
cmdline=$(cat /proc/cmdline)
no_systemd=false

# Load gem5_bridge driver
## Default parameters (ARM64)
gem5_bridge_baseaddr=0x10010000
gem5_bridge_rangesize=0x10000
## Try to read overloads from kernel arguments
if [[ $cmdline =~ gem5_bridge_baseaddr=([[:alnum:]]+) ]]; then
    gem5_bridge_baseaddr=${BASH_REMATCH[1]}
fi
if [[ $cmdline =~ gem5_bridge_rangesize=([[:alnum:]]+) ]]; then
    gem5_bridge_rangesize=${BASH_REMATCH[1]}
fi
## Insert driver
modprobe gem5_bridge \
    gem5_bridge_baseaddr=$gem5_bridge_baseaddr \
    gem5_bridge_rangesize=$gem5_bridge_rangesize

# see if this modprode fails or not
# print warning if it fails, gem5-bridge module is not going to work, you will need sudo for running exit events

# gem5-bridge exit signifying that kernel is booted
# This will cause the simulation to exit. Note that this will
# cause qemu to fail.
printf "Kernel booted, In gem5 init...\n"
gem5-bridge --addr=0x10010000 exit # TODO: Make this a specialized event.

if [[ $cmdline == *"no_systemd"* ]]; then
    no_systemd=true
fi

# Run systemd via exec if not disabled
if [[ $no_systemd == false ]]; then
    printf "Starting systemd...\n"
    exec /lib/systemd/systemd
else
    # Directly log in as the gem5 user
    printf "Dropping to shell as gem5 user...\n"
    exec su - gem5
fi
