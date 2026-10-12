---
title: Lmbench ubuntu 24.04 disk images
tags:
    - x86
    - arm
    - riscv
    - fullsystem
permalink: resources/lmbench-24.04-imgs
shortdoc: >
    This resource implementes the lmbench benchmark .
author: ["Harshil Patel"]
license: BSD-3-Clause
---

This document provides instructions to create a lmbench ubuntu 24.04 disk image, which, along with an example script, may be used to run lmbench within gem5 simulations. The example script uses a pre-built disk-image.

A pre-built disk image, for X86, can be found, gzipped, here: TODO add after the disk is uploaded
A pre-built disk image, for arm, can be found, gzipped, here: TODO add after the disk is uploaded

## What's on the disk?

- username: gem5
- password: 12345

- The `gem5-bridge`(m5) utility is installed in `/usr/local/bin/gem5-bridge`.
- `libm5` is installed in `/usr/local/lib/`.
- The headers for `libm5` are installed in `/usr/local/include/gem5-bridge`.
- `lmbench` benchmark sutie

Thus, you should be able to build packages on the disk and easily link to the gem5-bridge library.

The disk has network disabled by default to improve boot time in gem5.

If you want to enable networking, you need to modify the disk image and move the file `/etc/netplan/50-cloud-init.yaml.bak` to `/etc/netplan/50-cloud-init.yaml`.

## Changes from the original lmbench

- Added `results-with-config` target in `src/Makefile` to support non-interactive configuration
- Allows passing a pre-configured config file via command line: `make results-with-config CONFIGFILE=/path/to/config`
- Added some predefined configs that will be used to make workloads to run lmbench via gem5-resources

### Usage

**With pre-configured file:**

```bash
cd src
make results-with-config CONFIGFILE=/path/to/your-config
```

**Standard interactive mode:**

```bash
cd src
make results
```

### Creating a Config File

The easiest way to create a config file is to run the interactive configuration once:

```bash
cd src
make results
# Answer all the prompts
```

The generated config file will be located at `bin/<OS>/<config-name>`. You can then copy and modify this file for your needs. See `scripts/Config` for all available configuration options and their descriptions.

Alternatively, refer to `scripts/config-run` to understand how the config file is used during benchmark execution.

## Building the Disk Image

### Arm specific file requirement

To get the `flash0.img` run the following commands in the `files` directory.

```bash
dd if=/dev/zero of=flash0.img bs=1M count=64
dd if=/usr/share/qemu-efi-aarch64/QEMU_EFI.fd of=flash0.img conv=notrunc
```

**Note**: The `build-arm.sh` will make this file for you.

Assuming that you are in the `src/lmbench-24.04-imgs/` directory, run

```sh
./build-x86.sh          # the script downloading packer binary and building 
```

to build the x86 disk image

After this process succeeds, the disk image can be found on the `lmbench-24.04-imgs/disk-image-x86-lmbench/disk-image-x86-lmbench`.

This lmbench image uses the prebuilt ubuntu 24.04 image as a base image. The lmbench image also throws the same exit events as the base image.

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
