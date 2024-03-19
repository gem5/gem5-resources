---
title: Building the base x86-ubuntu image
authors:
    - Harshil Patel
---

This document provides instructions to create the "x86-ubuntu" image. This image is a 22.04 Ubuntu image.

## Directory map

- `files`: Files that are copied to the disk image.
- `scripts`: Scripts run on the disk image after installation.
- `http`: cloud-init Ubuntu autoinstall files
- `disk-image`: The disk image output directory.

## Disk Image

Run `./build.sh` in the `x86-ubuntu` directory to build the disk image.
This will download the packer binary, initialize packer, and build the disk image.

Note: This can take a while to run.
You will see `qemu.initialize: Waiting for SSH to become available...` while the installation is running.
You can watch the installation with a VNC viewer.
See [Troubleshooting](#troubleshooting) for more information.

## Changes from the base Ubuntu 22.04 image

- The default user is `gem5` with password `12345`.
- The `m5` utility is renamed to `gem5-bridge`
  - `gem5-bridge` utility is installed in `/usr/local/bin/gem5-bridge`.
  - `gem5-bridge` has a simlink to `m5` for backwards compatibility.
  - `libm5` is installed in `/usr/local/lib/` and the headers for `libm5` are installed in `/usr/local/include/m5`.
- The `.bashrc` file checks to see if there is anything in the `m5 readfile` command and executes the script if there is.
- The init process is modified to provide better annotations and more exit event. For more details see the [Init Process and Exit events](#init-process-and-exit-events).
  - The `m5 exit` command is run after the linux kernel initialization by default.
  - If the `no_systemd` boot option is passed, systemd is not run and the user is dropped to a terminal.
  - If the `interactive` boot option is passed, the `m5 exit` command is not run after the linux kernel initialization.
- Networking is disabled by moving the `/etc/netplan/00-installer-config.yaml` file to `/etc/netplan/00-installer-config.yaml.bak`.
  - If you want to enable networking, you need to modify the disk image and move the file `/etc/netplan/00-installer-config.yaml.bak` to `/etc/netplan/00-installer-config.yaml`.

## Init Process and Exit Events

The default boot process for the diskimage is with systemd and in no interactive mode.
The diskimage can have 2 different arguments to modify the default boot process that can be passed as kernel arguments.
These are `no_systemd=true` and `interactive=true`.

Here is the expected output for all the 4 different boot sequences.

- Default (with systemd and no interactive)
  - Kernel output
  - **Kernel Booted print message** **
  - Running systemd print message
  - Systemd output
  - autologin
  - **Running after_boot script** **
  - Print indicating **non interactive**
  - **Reading run script file**
  - Script output
  - Exit **
- With systemd and interactive
  - Kernel output
  - **Kernel Booted print message** **
  - Running systemd print message
  - Systemd output
  - autologin
  - **Running after_boot script** **
  - Shell
- Without systemd and no interactive
  - Kernel output
  - **Kernel Booted print message** **
  - autologin
  - **Running after_boot script** **
  - Print indicating **non interactive**
  - **Reading run script file**
  - Script output
  - Exit **
- Without systemd and interactive
  - Kernel output
  - **Kernel Booted print message** **
  - autologin
  - **Running after_boot script** **
  - Shell

**Note**: The Bold points signify a printf statement and the `**` signify's an `gem5-bridge exit`/`m5 exit` exit event.

## Extending this disk image

You can extend the disk image by specializing what is run after installation by modifying the `x86-ubuntu/post-installation.sh` file.

If you have your own init script then it should replace the  `gem5_init.sh` script in the `x86-ubuntu.pkr.hcl` and `post-installation.sh`. 

The `after-boot.sh` script is run when we first login to shell, it is important to note that `after-boot.sh` will run infinitely as we execute `bin/bash` in `after-boot.sh`. It is necessary to prevent this by adding a check like shown in `post-installation.sh`

```
echo -e "\nif [ -z \"\$AFTER_BOOT_EXECUTED\" ]; then\n   export AFTER_BOOT_EXECUTED=1\n    /home/gem5/after_boot.sh\nfi\n" >> /home/gem5/.bashrc
```

As we are not using root as the default user, the `gem5-bridge` binary needs root permissions to be ran, so we need to use setuid by using he command `chmod u+s file/path`.

## Creating a disk image from scratch

Instead of starting with our base image, if you want to create an image from scratch, you can automate the Ubuntu installation process after the first manual time by getting the `/var/log/installer/autoinstall-user-data` file from the guest OS and using it as the `user-data` file in the `shared` directory.
The `user-data` file included in this directory was created by answering most questions by default (except "minimal install")

## Troubleshooting

To see what `packer` is doing, you can use the environment variable `PACKER_LOG=INFO` when running `./build.sh`.

Packer seems to have a bug that aborts the VM build after 2-5 minutes regardless of the ssh_timeout setting.
As a workaround, set ssh_handshake_attempts to a high value.
Thus, I have `ssh_handshake_attempts = 1000`.
From <https://github.com/rlaun/packer-ubuntu-22.04>

To see what is happening while packer is running, you can connect with a vnc viewer.
The port for the vnc viewer is shown in the terminal while packer is running.

You can mount the disk image to see what is inside.
Use the following command to mount the disk image:
(note `norecovery` is needed if you get the error "cannot mount ... read-only")

```sh
mkdir x86-ubuntu/mount
sudo mount -o loop,offset=2097152,norecovery x86-ubuntu/x86-ubuntu-image/x86-ubuntu x86-ubuntu/mount
```

Useful documentation: https://ubuntu.com/server/docs/install/autoinstall