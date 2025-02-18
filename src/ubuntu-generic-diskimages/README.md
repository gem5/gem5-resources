---
title: Base Linux x86-ubuntu and arm-ubuntu image with Ubuntu 22.04 or 24.04
shortdoc: >
    Resources to build a generic x86-ubuntu or arm-ubuntu disk image and run a "boot-exit" test.
authors: ["Harshil Patel"]
---

The disk images are based on Ubuntu and support both x86 and ARM architectures, specifically Ubuntu 22.04 and 24.04. These images have their .bashrc files modified to execute a script passed from the gem5 configuration files (using the m5 readfile instruction). The boot-exit test passes a script that causes the guest OS to terminate the simulation (using the m5 exit instruction) as soon as the system boots.

## What's on the disk?

- username: gem5
- password: 12345

- The `gem5-bridge`(m5) utility is installed in `/usr/local/bin/gem5-bridge`.
- `libm5` is installed in `/usr/local/lib/`.
- The headers for `libm5` are installed in `/usr/local/include/`.

Thus, you should be able to build packages on the disk and easily link to the gem5-bridge library.

The disk has network disabled by default to improve boot time in gem5.

If you want to enable networking, you need to modify the disk image and move the file `/etc/netplan/00-installer-config.yaml.bak` or `/etc/netplan/50-cloud-init.yaml.bak` to `/etc/netplan/00-installer-config.yaml` or `/etc/netplan/50-cloud-init.yaml` depending on which config file the disk image contains. The x86 ubuntu 22.04 image should have `/etc/netplan/00-installer-config.yaml` and the other images should have ``/etc/netplan/50-cloud-init.yaml`.
For example you can use the following commands to re-enable network:

```sh
sudo mv /etc/netplan/50-cloud-init.yaml.bak /etc/netplan/50-cloud-init.yaml
sudo netplan apply
```

### Installed packages

- `build-essential`
- `git`
- `scons`
- `vim`

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

## Example Run Scripts

Within the gem5 repository, two example scripts are provided which utilize the x86 ubuntu 24.04 image.

The first is `configs/example/gem5_library/x86-ubuntu-run.py`.
This will boot the OS with a Timing CPU.
To run:

```sh
scons build/X86/gem5.opt -j`nproc`
./build/ALL/gem5.opt configs/example/gem5_library/x86-ubuntu-run.py
```

The second is `configs/example/gem5_library/x86-ubuntu-run-with-kvm.py`.
This will boot the OS using KVM cores before switching to Timing Cores after systemd is booted.
To run:

```sh
scons build/X86/gem5.opt -j`nproc`
./build/ALL/gem5.opt configs/example/gem5_library/x86-ubuntu-run-with-kvm.py
```

To use your local disk image you can use the `DiskImageResource` class from `resources.py` in gem5.
Following is an example of how to use your local disk image in gem5:

```python
disk_img = DiskImageResource("/path/to/disk/image/directory/x86-disk-image-24-04/x86-ubuntu")
board.set_kernel_disk_workload(
        disk_image=disk_img,
        kernel=obtain_resource("x86-linux-kernel-5.4.0-105-generic"),
        kernel_args=[
            "earlyprintk=ttyS0",
            "console=ttyS0",
            "lpj=7999923",
            "root=/dev/sda2"
        ]
    )
```

**Note:** the `x86-ubuntu-with-kvm.py` script requires a x86 host machine with KVM to function correctly.

The gem5 respository also has two example scripts that utilize the arm ubuntu 24.04 image.

The first is `configs/example/gem5_library/arm-ubuntu-run.py`.
This will boot the OS with a Timing CPU
To run:

```sh
scons build/ARM/gem5.opt -j `nproc`
./build/ALL/gem5.opt configs/example/gem5_library/arm-ubuntu-run.py
```

The second is `configs/example/gem5_library/arm-ubuntu-run-with-kvm.py`.
This will boot the OS using KVM cores before switching to Timing Cores after systemd is booted.
To run:

```sh
scons build/ARM/gem5.opt -j `nproc`
./build/ALL/gem5.opt configs/example/gem5_library/arm-ubuntu-run-with-kvm.py
```

To use your local disk image you can use the `DiskImageResource` class from `resources.py` in gem5.
Following is an example of how to use your local disk image in gem5:

```python
disk_img = DiskImageResource("/path/to/disk/image/directory/arm-disk-image-22-04/arm-ubuntu", root_partition="2")
board.set_kernel_disk_workload(
        disk_image=disk_img,
        bootloader=obtain_resource("arm64-bootloader-foundation"),
        kernel=obtain_resource("arm64-linux-kernel-5.15.36")
    )
```

**Note:** the `arm-ubuntu-run-with-kvm.py` script requires an Arm host machine with KVM to function correctly.

## Building and modifying the disk image

See [BUILDING.md](BUILDING.md) for instructions on how to build the disk image.

See [using-local-resources](https://www.gem5.org/documentation/gem5-stdlib/using-local-resources) for instructions on how to use customized resources in gem5 experiments.

To boot in qemu, make sure to specify `/sbin/init.old` as the `init=` option on the kernel command line.
Otherwise, you will execute the gem5 magic instructions which will cause illegal instructions or segmentation faults.
