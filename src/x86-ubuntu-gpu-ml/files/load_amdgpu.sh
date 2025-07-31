#!/bin/bash

# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD 3-Clause

# Loading the driver with gem5's bare bones ACPI implementation causes ACPI to
# be disabled, causing modprobe to fail. In particular, it fails on WMI init:
# https://elixir.bootlin.com/linux/v6.10.8/source/drivers/platform/x86/wmi.c#L1356
#
# To fix this we manually insmod all of the dependencies of the amdgpu module
# and a bandaid module containing missing ACPI symbols. The missing symbols are
# not important for gem5 so this does not cause any problems.
insmod /home/gem5/gem5_wmi.ko

insmod /lib/modules/`uname -r`/kernel/drivers/acpi/video.ko.zst
insmod /lib/modules/`uname -r`/kernel/drivers/i2c/algos/i2c-algo-bit.ko.zst
insmod /lib/modules/`uname -r`/kernel/drivers/media/rc/rc-core.ko.zst
insmod /lib/modules/`uname -r`/kernel/drivers/media/cec/core/cec.ko.zst
insmod /lib/modules/`uname -r`/kernel/drivers/gpu/drm/display/drm_display_helper.ko.zst
insmod /lib/modules/`uname -r`/kernel/drivers/gpu/drm/drm_suballoc_helper.ko.zst
insmod /lib/modules/`uname -r`/kernel/drivers/gpu/drm/drm_exec.ko.zst
insmod /lib/modules/`uname -r`/updates/dkms/amdkcl.ko.zst
insmod /lib/modules/`uname -r`/updates/dkms/amd-sched.ko.zst
insmod /lib/modules/`uname -r`/updates/dkms/amdxcp.ko.zst
insmod /lib/modules/`uname -r`/updates/dkms/amddrm_buddy.ko.zst
insmod /lib/modules/`uname -r`/updates/dkms/amdttm.ko.zst
insmod /lib/modules/`uname -r`/updates/dkms/amddrm_ttm_helper.ko.zst

insmod /lib/modules/`uname -r`/updates/dkms/amdgpu.ko.zst ip_block_mask=0x6f ppfeaturemask=0 dpm=0 audio=0 ras_enable=0 discovery=2
