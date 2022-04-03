#!/bin/bash
# Copyright (c) 2022 The University of California
# Copyright (c) 2021 The University of Texas at Austin.
# SPDX-License-Identifier: BSD 3-Clause

echo "Starting gem5 init... reading run script file."
# Calling m5 with --addr as m5ops_base address.
if ! m5 --addr 0x10010000 readfile > /tmp/script; then
    echo "Failed to run m5 readfile, exiting!"
    rm -f /tmp/script
    if ! m5 exit; then
        # Useful for booting the disk image in (e.g.,) qemu for debugging.
        echo "m5 exit failed, dropping to shell."
        /bin/sh
    fi
else
    echo "Running m5 script from /tmp/script"
    chmod 755 /tmp/script
    /tmp/script || true
    echo "Done running script, exiting."
    rm -f /tmp/script
    m5 --addr 0x10010000 exit
fi
