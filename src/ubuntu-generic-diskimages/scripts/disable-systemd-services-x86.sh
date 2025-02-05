#!/bin/bash

# Copyright (c) 2024 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

echo "Disabling systemd services for x86 architecture..."

# Disable multipathd service
systemctl disable multipathd.service

# Disable thermald service
systemctl disable thermald.service

# Disable snapd services and socket
systemctl disable snapd.service snapd.socket

# Disable unnecessary timers
systemctl disable apt-daily.timer apt-daily-upgrade.timer fstrim.timer

# Disable accounts-daemon
systemctl disable accounts-daemon.service

# Disable LVM monitoring service
systemctl disable lvm2-monitor.service

# Switch default target to multi-user (no GUI)
systemctl set-default multi-user.target

# Optionally disable AppArmor if not required
systemctl disable apparmor.service snapd.apparmor.service

echo "completed disabling systemd services for x86."