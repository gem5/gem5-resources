#!/bin/bash

# Copyright (c) 2022,2024 The University of California.
# Copyright (c) 2021 The University of Texas at Austin.
# SPDX-License-Identifier: BSD 3-Clause

# This file is executed at the end of the bashrc for the gem5 user.
# The script checks to see if we should run in interactive mode or not.
# If we are in interactive mode, the script will drop to a shell.
# If we are not in interactive mode, the script will check if we should
# run a script from the gem5-bridge. If so, it will run the script and
# exit. If there is no script and we are not in interactive mode, it will
# exit. This last option is used for testing purposes.

# gem5-bridge exit signifying that after_boot.sh is running
printf "In after_boot.sh...\n"
gem5-bridge --addr=0x10010000 exit # TODO: Make this a specialized event.

# Read /proc/cmdline and parse options

cmdline=$(cat /proc/cmdline)
interactive=false
IGNORE_M5=0
if [[ $cmdline == *"interactive"* ]]; then
    interactive=true
fi

printf "Interactive mode: $interactive\n"

if [[ $interactive == true ]]; then
    printf "Interactive mode enabled, dropping to shell."
    /bin/bash
else
    # Try to read the file from the host when running with gem5
    if ! [ -z $IGNORE_M5 ]; then
        printf "Starting gem5 init... trying to read run script file via readfile.\n"
        if ! gem5-bridge --addr=0x10010000 readfile > /tmp/script; then
            printf "Failed to run gem5-bridge readfile, exiting!\n"
            rm -f /tmp/script
            # If we can't read the script exit the simulation. If we cannot exit the
            # simulation, this probably means that we are running in QEMU. So, ignore
            # future calls to gem5-bridge.
            if ! gem5-bridge --addr=0x10010000 exit; then
                # Useful for booting the disk image in (e.g.,) qemu for debugging
                printf "gem5-bridge exit failed, dropping to shell.\n"
                IGNORE_M5=1 /bin/bash
            fi
        else
            printf "Running script from gem5-bridge stored in /tmp/script\n"
            chmod 755 /tmp/script
            /tmp/script
            printf "Done running script from gem5-bridge, exiting.\n"
            rm -f /tmp/script
            gem5-bridge --addr=0x10010000 exit
        fi
    fi
fi
