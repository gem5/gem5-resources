---
title: Building the base x86-ubuntu image
authors:
    - Harshil Patel
---

This document provides instructions to create the "x86-ubuntu" image. This image is a 22.04 Ubuntu image.

## Directory map

- `files`: Files that are copied to the disk image.
- `scripts`: Scripts run on the disk image after installation.
- `http`: cloud-init Ubuntu autoinstall files.
- `disk-image`: The disk image output directory.

## Disk Image

Run `./build.sh` in the `riscv-ubuntu-22.04` directory to build the disk image.
This will download the packer binary, initialize packer, and build the disk image.

Note: This can take a while to run.
You will see `qemu.initialize: Waiting for SSH to become available...` while the installation is running.
You can watch the installation with a VNC viewer.
See [Troubleshooting](#troubleshooting) for more information.

## Changes from the base Ubuntu 22.04 image

- The default user is `gem5` with password `12345`.
- The `m5` utility is renamed to `gem5-bridge`.
  - `gem5-bridge` utility is installed in `/usr/local/bin/gem5-bridge`.
  - `gem5-bridge` has a symlink to `m5` for backwards compatibility.
  - `libm5` is installed in `/usr/local/lib/` and the headers for `libm5` are installed in `/usr/local/include/m5`.
- The `.bashrc` file checks to see if there is anything in the `gem5-bridge readfile` command and executes the script if there is.
- The init process is modified to provide better annotations and more exit event. For more details see the [Init Process and Exit events](README.md#init-process-and-exit-events).
  - The `gem5-bridge exit` command is run after the linux kernel initialization by default.
  - If the `no_systemd` boot option is passed, systemd is not run and the user is dropped to a terminal.
  - If the `interactive` boot option is passed, the `gem5-bridge exit` command is not run after the linux kernel initialization.
- Networking is disabled by moving the `/etc/netplan/00-installer-config.yaml` file to `/etc/netplan/00-installer-config.yaml.bak`.
  - If you want to enable networking, you need to modify the disk image and move the file `/etc/netplan/00-installer-config.yaml.bak` to `/etc/netplan/00-installer-config.yaml`.

## Extending the Disk Image

### Customization of Post-Installation Processes

- **Replace `gem5_init.sh`**: This script is what executes as the Linux init process (pid=0) immediately after Linux boot. If you have a custom initialization script, replace the default `gem5_init.sh` in both `x86-ubuntu.pkr.hcl` and `post-installation.sh` to integrate your custom initialization process.
- **Replace `gem5_init.sh`**: If you have a custom initialization script, replace the default `gem5_init.sh` in both `x86-ubuntu.pkr.hcl` and `post-installation.sh` to integrate your custom initialization process.

### Handling the After-Boot Script

- **Persistent Execution of `after-boot.sh`**: The `after-boot.sh` script executes at first login.
To avoid its infinite execution, we incorporated a conditional check in `post-installation.sh` similar to the following:

  ```sh
  echo -e "\nif [ -z \"\$AFTER_BOOT_EXECUTED\" ]; then\n   export AFTER_BOOT_EXECUTED=1\n    /home/gem5/after_boot.sh\nfi\n" >> /home/gem5/.bashrc
  ```

  This ensures `after-boot.sh` runs only once per session by setting an environment variable.

  For riscv images, the `after-boot.sh` and the filesystem is set to read only for the base image that is used.
  So, we are also updating the `/etc/fstab` file to make sure that the filesystem is mounted with read and write permissions.
  We are also adding execute permissions to the `after-boot.sh` file.

### Adjusting File Permissions

- **Setting Permissions for `gem5-bridge`**: Since the default user is not root, `gem5-bridge` requires root permissions. Apply setuid to grant these permissions:

  ```sh
  chmod u+s /path/to/gem5-bridge
  ```

## Creating a Disk Image from Scratch

### Using a Preinstalled Image

- **Base Image Selection**: Since a live server install image for RISC-V is unavailable, utilize a preinstalled Ubuntu image for RISC-V. Specify the image path or URL in `iso_url` and its SHA256 checksum in `iso_checksum`. Ensure the `qemu` plugin configuration includes `diskimage` set to `true`, indicating the use of a preexisting disk image rather than an ISO.

- **Boot Command Configuration**: The preinstalled disk images require a password reset upon first login. To automate this, use the following boot command sequence:
  - `<wait120>`: Delays execution to allow the image to boot and reach the login prompt. Adjust the duration based on the host system's performance where Packer is running.
  - Auto-login with the default `ubuntu` user and change its password to `12345678`.
  - Create a new user `gem5` with a password `12345`.
  - Grant sudo permissions to the `gem5` user.

### Configuration and Directory Structure

- **Determine QEMU Arguments**: Identify the QEMU arguments required for booting the system. These vary by ISA and mirror the arguments used for booting a disk image in QEMU.
- **Directory Organization**: Arrange your source directory to include any additional content. Utilize the `provisioner` section for transferring extra files into the disk image, ensuring all necessary resources are embedded within your custom disk image.

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
mkdir riscv-ubuntu/mount
sudo mount -o loop,offset=2097152,norecovery riscv-ubuntu/riscv-ubuntu-image/riscv-ubuntu riscv-ubuntu/mount
```

Useful documentation: <https://ubuntu.com/server/docs/install/autoinstall>