#!/bin/bash

# Read /proc/cmdline and parse options
cmdline=$(cat /proc/cmdline)
no_systemd=false
no_m5_exit_on_boot=false

if [[ $cmdline == *"no_systemd"* ]]; then
    no_systemd=true
fi

if [[ $cmdline == *"no_m5_exit_on_boot"* ]]; then
    no_m5_exit_on_boot=true
fi

# Run m5 exit if not disabled
if [[ $no_m5_exit_on_boot == false ]]; then
    m5 exit
fi

# Run systemd via exec if not disabled
if [[ $no_systemd == false ]]; then
    exec /lib/systemd/systemd
else
    exec /bin/bash
fi
