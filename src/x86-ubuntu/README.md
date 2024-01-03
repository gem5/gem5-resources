---
title: Base Linux x86-ubuntu image
shortdoc: >
    Resources to build a generic x86-ubuntu disk image and run a "boot-exit" test.
authors: ["Ayaz Akram"]
---

The x86-ubuntu disk image is based on Ubuntu 22.04 and has its `.bashrc` file modified in such a way that it executes a script passed from the gem5 configuration files (using the `m5 readfile` instruction).
The `boot-exit` test passes a script that causes the guest OS to terminate the simulation (using the `m5 exit` instruction) as soon as the system boots.

## What's on the disk?

- username: gem5
- password: 12345

The `m5` utility is installed in `/usr/local/bin/m5`.
`libm5` is installed in `/usr/local/lib/`.
The headers for `libm5` are installed in `/usr/local/include/m5`.
Thus, you should be able to build packages on the disk and easily link to the m5 library.

The disk has network disabled by default to improve boot time in gem5.
If you want to enable networking, you need to modify the disk image and move the file `/etc/netplan/00-installer-config.yaml.bak` to `/etc/netplan/00-installer-config.yaml`.

### Installed packages

- `build-essential`
- `git`
- `scons`
- `vim`

### Boot options

There are two boot options available on the command line.

- "no_systemd" will boot the system without systemd and immediately drop you to a terminal after the linux kernel is initialized.
- "no_m5_exit_on_boot" will *not* run the `m5 exit` command after the linux kernel initialization.

By default, systemd is enabled and the `m5 exit` command is run after the linux kernel initialization.

## Example Run Scripts

Within the gem5 repository, two example scripts are provided which utilize the x86-ubuntu resource.

The first is `configs/example/gem5_library/x86-ubuntu-run.py`.
This will boot the OS with a Timing CPU.
To run:

```sh
scons build/X86/gem5.opt -j`nproc`
./build/X86/gem5.opt configs/example/gem5_library/x86-ubuntu-run.py
```

The second is `configs/example/gem5_library/x86-ubuntu-run-with-kvm.py`.
This will boot the OS using KVM cores before switching to Timing Cores to run a simple echo command.
To run:

```sh
scons build/X86/gem5.opt -j`nproc`
./build/X86/gem5.opt configs/example/gem5_library/x86-ubuntu-run-with-kvm.py`
```

**Note:** the `x86-ubuntu-with-kvm.py` script requires a host machine with KVM to function correctly.

## Building and modifying the disk image

See [BUILDING.md](BUILDING.md) for instructions on how to build the disk image.