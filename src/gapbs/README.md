---
title: GAP Benchmark Suite (GAPBS) tests
tags:
    - x86
    - fullsystem
permalink: resources/gapbs
shortdoc: >
    This resource implementes the [GAP benchmark suite](http://gap.cs.berkeley.edu/benchmark.html).
author: ["Harshil Patel"]
license: BSD-3-Clause
---

This document provides instructions to create a GAP Benchmark Suite (GAPBS) disk image, which, along with an example script, may be used to run GAPBS within gem5 simulations. The example script uses a pre-built disk-image.

A pre-built disk image, for X86, can be found, gzipped, here: <https://resources.gem5.org/resources/x86-ubuntu-24.04-gapbs-img/versions?database=gem5-resources&version=1.0.0>

A pre-built disk image, for ARM, can be found, gzipped, here: <https://resources.gem5.org/resources/arm-ubuntu-24.04-gapbs-img/versions?database=gem5-resources&version=1.0.0>

A pre-built disk image, for RISC-V, can be found, gzipped, here: <https://resources.gem5.org/resources/riscv-ubuntu-24.04-gapbs-img/versions?database=gem5-resources&version=1.0.0>

## Building the Disk Image

Assuming that you are in the `src/gapbs/` directory, run

```sh
./build-{ISA}.sh          # the script downloading packer binary and building the disk image. {ISA} is `x86`, `arm`, or `riscv`.
```

After this process succeeds, the disk image can be found on the `src/gapbs/disk-image-ubuntu-24-04`.

This gapbs image uses the prebuilt ubuntu 24.04 image as a base image. The gapbs image also throws the same exit events as the base image. For more details on the exit events, check out the [Boot Sequences](#boot-sequences) section.

Each benchmark also has its regions of interest annotated and they throw a `gem5-bridge hypercall 4` (work_begin hypercall) and `gem5-bridge hypercall 5` (work_end hypercall) exit event.

## What's on the disk?

- username: gem5
- password: 12345

- The `gem5-bridge`(m5) utility is installed in `/usr/local/bin/gem5-bridge`.
- `libm5` is installed in `/usr/local/lib/`.
- The headers for `libm5` are installed in `/usr/local/include/gem5-bridge`. For more information on m5, read the following [documentation](https://www.gem5.org/documentation/general_docs/m5ops/).
- `gapbs` benchmark suite with ROI annotations. The built binaries are in the path `/home/gem5/gapbs/`.
- Two real graphs: `soc-LiveJournal1` and `facebook_combined`

Thus, you should be able to build packages on the disk and easily link to the gem5-bridge library.

The disk has network disabled by default to improve boot time in gem5.

If you want to enable networking, you need to modify the disk image and move the file `/etc/netplan/00-installer-config.yaml.bak` to `/etc/netplan/00-installer-config.yaml`.

## Init Process and Exit Events

This section outlines the disk image's boot process variations and the impact of specific boot parameters on its behavior.
By default, the disk image boots with systemd in a non-interactive mode.
Users can adjust this behavior through kernel arguments at boot time, influencing the init system and session interactivity.

### Boot Parameters

The disk image supports two main kernel arguments to adjust the boot process:

- `no_systemd=true`: Disables systemd as the init system, allowing the system to boot without systemd's management.
- `interactive=true`: Enables interactive mode, presenting a shell prompt to the user for interactive session management.

Combining these parameters yields four possible boot configurations:

1. **Default (Systemd, Non-Interactive)**: The system uses systemd for initialization and runs non-interactively.
2. **Systemd and Interactive**: Systemd initializes the system, and the boot process enters an interactive mode, providing a user shell.
3. **Without Systemd and Non-Interactive**: The system boots without systemd and proceeds non-interactively, executing predefined scripts.
4. **Without Systemd and Interactive**: Boots without systemd and provides a shell for interactive use.

### Note on Print Statements and Exit Events

- The bold points in the sequence descriptions are `printf` statements in the code, indicating key moments in the boot process.
- The `**` symbols mark gem5 exit events, essential for simulation purposes, dictating system shutdown or reboot actions based on the configured scenario.

### Boot Sequences

#### Default Boot Sequence (Systemd, Non-Interactive)

- Kernel output
- **Kernel Booted print message** **
- Running systemd print message
- Systemd output
- autologin
- **Running after_boot script** **
- Print indicating **non-interactive** mode
- **Reading run script file**
- Script output
- Exit **

#### With Systemd and Interactive

- Kernel output
- **Kernel Booted print message** **
- Running systemd print message
- Systemd output
- autologin
- **Running after_boot script** **
- Shell

#### Without Systemd and Non-Interactive

- Kernel output
- **Kernel Booted print message** **
- autologin
- **Running after_boot script** **
- Print indicating **non-interactive** mode
- **Reading run script file**
- Script output
- Exit **

#### Without Systemd and Interactive

- Kernel output
- **Kernel Booted print message** **
- autologin
- **Running after_boot script** **
- Shell

This detailed overview shows how different boot configurations affect the system's initialization and mode of operation.
By selecting the appropriate parameters, users can customize the boot process for diverse environments, ranging from automated setups to hands-on interactive sessions.

## Using the disk image and kernel in gem5

All GAPBS benchmarks are available on <https://resources.gem5.org/>.
You can use these benchmarks directly in your gem5 config scripts, like this:

```python
board.set_workload(
    obtain_resource("arm-ubuntu-24.04-gapbs-bc-test", resource_version="1.0.0")
)
```

If you have made the disk image locally and want to run it on gem5 you can use the following templates:

**Note**: You would need to use the kernels mentioned in the templates to make sure gem5 can run the workload.

This is because these disk images are built with the [gem5 bridge driver module](https://github.com/gem5/gem5/tree/stable/util/gem5_bridge), added in [PR 1480](https://github.com/gem5/gem5/pull/1480), which makes it so that gem5 simulations don't need sudo permissions when running commands.
The gem5 bridge driver module has the limitation of needing to be run with the same kernel version as what it was built with.
For ARM:

```python
disk_img = DiskImageResource("/path/to/arm-gapbs-img", root_partition="2")
board.set_kernel_disk_workload(
        disk_image=disk_img,
        bootloader=obtain_resource("arm64-bootloader-foundation", resource_version="2.0.0"),
        kernel=obtain_resource("arm64-linux-kernel-6.8.12", resource_version="1.0.0"),
        readfile_contents="/home/gem5/gapbs/bin/bc -u 19 -k 4 -s",
        )
```

For X86:

```python
disk_img=DiskImageResource("/path/to/x86-gapbs-img")
board.set_kernel_disk_workload(
    kernel=obtain_resource(
        "x86-linux-kernel-6.8.0-52-generic", resource_version="1.0.0"
    ),
    disk_image=disk_img,
    readfile_contents="/home/gem5/gapbs/bin/bc -u 19 -k 4 -s",
    kernel_args=[
        "earlyprintk=ttyS0",
        "console=ttyS0",
        "lpj=7999923",
        "root=/dev/sda2",
    ]

)
```

For RISC-V:

```python
disk_img=DiskImageResource("/path/to/riscv-gapbs-img", root_partition="1")
board.set_kernel_disk_workload(
    kernel=obtain_resource(
        "riscv-linux-6.8.12-kernel", resource_version="1.0.0"
    ),
    disk_image=disk_img,
    bootloader=obtain_resource("riscv-bootloader-opensbi-1.3.1", resource_version="1.0.0"),
    readfile_contents="/home/gem5/gapbs/bin/bc -u 19 -k 4 -s",
)
```
