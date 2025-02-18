---
title: NPB ubuntu 24.04 disk images
tags:
    - x86
    - arm
    - fullsystem
permalink: resources/npb-24.04-imgs
shortdoc: >
    This resource implementes the NPB benchmark .
author: ["Harshil Patel"]
license: BSD-3-Clause
---

This document provides instructions to create a NPB ubuntu 24.04 disk image, which, along with an example script, may be used to run NPB within gem5 simulations. The example script uses a pre-built disk-image.

A pre-built disk image, for X86, can be found, gzipped, here: [x86-ubuntu-24.04-npb-img](https://resources.gem5.org/resources/x86-ubuntu-24.04-npb-img?version=2.0.0)

A pre-built disk image, for arm, can be found, gzipped, here:
[arm-ubuntu-24.04-npb-img](https://resources.gem5.org/resources/arm-ubuntu-24.04-npb-img?version=2.0.0)

## What's on the disk?

- username: gem5
- password: 12345

- The `gem5-bridge`(m5) utility is installed in `/usr/local/bin/gem5-bridge`.
- `libm5` is installed in `/usr/local/lib/`.
- The headers for `libm5` are installed in `/usr/local/include/gem5-bridge`.
- `npb` benchmark sutie with ROI annotations

Thus, you should be able to build packages on the disk and easily link to the gem5-bridge library.

The disk has network disabled by default to improve boot time in gem5.

If you want to enable networking, you need to modify the disk image and move the file `/etc/netplan/50-cloud-init.yaml.bak` to `/etc/netplan/50-cloud-init.yaml`.

## Building the Disk Image

### Arm specific file requirement

To get the `flash0.img` run the following commands in the `files` directory.

```bash
dd if=/dev/zero of=flash0.img bs=1M count=64
dd if=/usr/share/qemu-efi-aarch64/QEMU_EFI.fd of=flash0.img conv=notrunc
```

**Note**: The `build-arm.sh` will make this file for you.

Assuming that you are in the `src/npb-24.04-imgs/` directory, run

```sh
./build-x86.sh          # the script downloading packer binary and building 
```

to build the x86 disk image or 

```sh
./build-arm.sh
```

to run the arm disk image.
After this process succeeds, the disk image can be found on the `npb-24.04-imgs/disk-image-x86-npb/disk-image-x86-npb` or `npb-24.04-imgs/disk-image-arm-npb/disk-image-arm-npb` repectively.

This npb image uses the prebuilt ubuntu 24.04 image as a base image. The npb image also throws the same exit events as the base image.

Each benchmark also has its regions of intrests annotated and they throw a `gem5-bridge workbegin` and `gem5-bridge workend` exit event.

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

This detailed overview provides a foundational understanding of how different boot configurations affect the system's initialization and mode of operation.
By selecting the appropriate parameters, users can customize the boot process for diverse environments, ranging from automated setups to hands-on interactive sessions.

## Handling Exit Events in gem5

The disk image triggers five exit events in total:

- 3 `gem5-bridge exit` events
- 1 `gem5-bridge workbegin` event
- 1 `gem5-bridge workend` event

To manage these events in gem5, you need to create three exit event handlers. Below is a code snippet showing how these handlers could be implemented and added to the `simulator` object in gem5:

```python
def handle_workbegin():
    print("Done booting Linux")
    print("Resetting stats at the start of ROI!")
    m5.stats.reset()
    processor.switch()
    yield False

# We expect that the ROI ends with `workend` or `simulate() limit reached`.
def handle_workend():
    print("Dumping stats at the end of the ROI!")
    m5.stats.dump()
    yield True

def exit_event_handler():
    print("First exit: Kernel booted")
    yield False  # gem5 is now executing systemd startup
    print("Second exit: Started `after_boot.sh` script")
    # The after_boot.sh script is executed after the kernel and systemd have booted.
    yield False  # gem5 is now executing the `after_boot.sh` script
    print("Third exit: Finished `after_boot.sh` script")
    # The after_boot.sh script will run a script if passed via m5 readfile. 
    # This is the last exit event before the simulation exits.
    yield True

simulator = Simulator(
    board=board,
    on_exit_event={
        ExitEvent.WORKBEGIN: handle_workbegin(),
        ExitEvent.WORKEND: handle_workend(),
        ExitEvent.EXIT: exit_event_handler(),
    },
)
```

This script defines three handlers for different exit events (`WORKBEGIN`, `WORKEND`, and `EXIT`).
