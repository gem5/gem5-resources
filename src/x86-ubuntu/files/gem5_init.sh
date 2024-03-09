#!/bin/bash

# mount /proc and /sys
mount -t proc /proc /proc
mount -t sysfs /sys /sys
# Read /proc/cmdline and parse options
cmdline=$(cat /proc/cmdline)
no_systemd=false

# gem5-bridge exit signifying that kernel is booted
printf "Kernel booted, In gem5 init...\n"
gem5-bridge exit

if [[ $cmdline == *"no_systemd"* ]]; then
    no_systemd=true
fi

# Run systemd via exec if not disabled
if [[ $no_systemd == false ]]; then
    # gem5-bridge exit signifying that systemd will be booted
    printf "Starting systemd...\n"
    exec /lib/systemd/systemd
else
    exec su - gem5
fi
