#!/bin/bash

# Copyright (c) 2022 The University of California.
# Copyright (c) 2021 The University of Texas at Austin.
# SPDX-License-Identifier: BSD 3-Clause

# gem5-bridge exit signifying that after_boot.sh is running
printf "Starting after_boot.sh..."
gem5-bridge exit

# Read /proc/cmdline and parse options

cmdline=$(cat /proc/cmdline)
interactive=false
IGNORE_M5=0
if [[ $cmdline == *"interactive"* ]]; then
    interactive=true
fi

if [[ $interactive == true ]]; then
    printf "Interactive mode enabled, dropping to shell."
    /bin/bash
else
    if ! [ -z $IGNORE_M5 ]; then
        printf "Starting gem5 init... reading run script file."
        if ! gem5-bridge readfile > /tmp/script; then
            printf "Failed to run gem5-bridge readfile, exiting!"
            rm -f /tmp/script
            if ! gem5-bridge exit; then
                # Useful for booting the disk image in (e.g.,) qemu for debugging
                printf "gem5-bridge exit failed, dropping to shell."
                IGNORE_M5=1 /bin/bash
            fi
        else
            printf "Running gem5-bridge script from /tmp/script"
            chmod 755 /tmp/script
            /tmp/script
            printf "Done running script, exiting."
            rm -f /tmp/script
            gem5-bridge exit
        fi
    fi
fi
