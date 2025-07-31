# gem5_wmi WMI/ACPI workaround module

Starting around ROCm 6.0, the amdgpu ROCk driver's Kconfig enables the WMI ACPI extensions whenever ACPI is enabled (more specific: this seems to only be a problem for Linux 6.x kernels. ROCm 6.0 on Linux 5.x seems to work). This always seems to be the case as the display manager code has two acpi calls, notably related to backlight features. gem5 does not have enough ACPI support for the wmi kernel module to load. Previous to ROCm 6.0, the kernel module acpi_video would load even if ACPI was disabled. The wmi kernel module will fail to load if ACPI is disabled, causing the amdgpu kernel module to also fail.

As a workaround, this module implements a WMI stub that continues to allow the acpi_video module to load. This is acceptable for simulating compute GPUs which turn off the display manager IP block.

This code requires having the Linux kernel source available. The disk image creation tool automatically downloads the linux-headers-`uname -r` package. This is fine as the kernel used to create the disk image must be used with the disk image.

## Build module

The disk image creation tool does this for you. However if you want to add more stubs a Makefile is provided which by default points to the currently running kernel. Instead one should point to the Linux source just built and run make from the directory containing this README:

`make -C /usr/src/linux-headers-$(uname -r) M=${PWD}`

This should create a gem5_wmi.ko object. This can be copied to the gem5 disk image and placed in /home/ubuntu/gem5_wmi.ko by default.
