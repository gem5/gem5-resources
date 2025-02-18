---
title: Building the base x86-ubuntu and arm-ubuntu image for Ubuntu 22.04 and 24.04
authors:
    - Harshil Patel
---

This document provides instructions to create the "x86-ubuntu" image and "arm-ubuntu" image.
This image can be a 22.04 or 24.04 Ubuntu image.

## Directory map

- `files`: Files that are copied to the disk image.
- `scripts`: Scripts run on the disk image after installation.
- `http`: cloud-init Ubuntu autoinstall files for different versions of Ubuntu for Arm and x86.
  - `arm-22-04`: cloud-init Ubuntu autoinstall files for arm ubuntu 22.04 image.
  - `arm-24-04`: cloud-init Ubuntu autoinstall files for arm ubuntu 24.04 image.
  - `x86`: cloud-init Ubuntu autoinstall files for x86 ubuntu 22.04 and 24.04 images.
- `x86-disk-image-24.04`: Disk image output directory for x86 ubuntu 24.04 image.
- `x86-disk-image-22.04`: Disk image output directory for x86 ubuntu 22.04 image.
- `arm-disk-image-24.04`: Disk image output directory for arm ubuntu 24.04 image.
- `arm-disk-image-22.04`: Disk image output directory for arm ubuntu 22.04 image.

## Disk Image

Run `build-x86.sh` with the argument `22.04` or `24.04` to build the respective x86 disk image in the `ubuntu-generic-diskimages` directory.
Run `build-arm.sh` with the argument `22.04` or `24.04` to build the respective arm disk image in the `ubuntu-generic-diskimages` directory.
Building the arm image assume that we are on an ARM machine as we use kvm to build the image.
You can also run the packer file by adding the "use_kvm=false" in `build-arm.sh` in the `./packer build` command to build the disk image without KVM.
This will download the packer binary, initialize packer, and build the disk image.

## Arm image specific requirements

We need a EFI file to boot the arm image. We use the file named `flash0.img` in the packer file.

To get the `flash0.img` run the following commands in the `files` directory

```bash
dd if=/dev/zero of=flash0.img bs=1M count=64
dd if=/usr/share/qemu-efi-aarch64/QEMU_EFI.fd of=flash0.img conv=notrunc
```

**Note**: The `build-arm.sh` will make this file for you.

Note: Building the image can take a while to run.
You will see `qemu.initialize: Waiting for SSH to become available...` while the installation is running.
You can watch the installation with a VNC viewer.
See [Troubleshooting](#troubleshooting) for more information.

## Kernel

For the x86 disk images a kernel is also extracted from the disk image during the post-installation process.
The extracted kernel does not have a version in its name, but the kernel version is printed before the extraction in `post-installation.sh` script. This extracted kernel can be used as a resource for gem5 simulations and is not limited to just be used with this disk image.
The extracted kernel does not have a version its name, but the kernel version is printed as before the extraction in `post-installation.sh` script. This extracted kernel can be used as a resource for gem5 simulations and is not limited to just be used with this disk image.

The kernel is extracted using packer's file provisioner with `direction=download` which would copy a file from the image to the host machine. The path specifying in the provisioner copies the file `/home/gem5/vmlinux-x86-ubuntu` to the output directory `disk-image`.

## Changes from the base Ubuntu image

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
- Networking is disabled by moving the `/etc/netplan/00-installer-config.yaml` or `/etc/netplan/50-cloud-init.yaml` file to `/etc/netplan/00-installer-config.yaml.bak` or `/etc/netplan/50-cloud-init.yaml.bak` respectively. The `systemd-networkd-wait-online.service` is also disabled.
The x86 22.04 image should have `00-installer-config.yaml` while all the other disk images should have `50-cloud-init.yaml`.
  - If you want to enable networking, you need to modify the disk image and move the file `/etc/netplan/00-installer-config.yaml.bak` or `/etc/netplan/50-cloud-init.yaml.bak` to `/etc/netplan/00-installer-config.yaml` or `/etc/netplan/50-cloud-init.yaml` depending on which config file the disk image contains.
  To re-enable `systemd-networkd-wait-online.service`, first, unmask the service with `sudo systemctl unmask systemd-networkd-wait-online.service` and then enable the service to start with `sudo systemctl enable systemd-networkd-wait-online.service`.
  If you require the service to start immediately without waiting for the next boot then also run the following:
  `sudo systemctl start systemd-networkd-wait-online.service`.

### Customization of the boot Processes

- **`gem5_init.sh` replaces /sbin/init**: This script is what executes as the Linux init process (pid=0) immediately after Linux boot. This script adds an `gem5-bridge exit` when the file is executed. It also checks the `no_systemd` kernel arg to redirect to the user or boot with systemd.

### Details of the After-Boot Script

- **Persistent Execution of `after-boot.sh`**: The `after-boot.sh` script executes at first login.
To avoid its infinite execution, we incorporated a conditional check in `post-installation.sh` similar to the following:

```sh
echo -e "\nif [ -z \"\$AFTER_BOOT_EXECUTED\" ]; then\n   export AFTER_BOOT_EXECUTED=1\n    /home/gem5/after_boot.sh\nfi\n" >> /home/gem5/.bashrc
```

This ensures `after-boot.sh` runs only once per session by setting an environment variable.

### Adjusting File Permissions

- **Setting Permissions for `gem5-bridge`**: Since the default user is not root, `gem5-bridge` requires root permissions. Apply setuid to grant these permissions:

  ```sh
  chmod u+s /path/to/gem5-bridge
  ```

## Extending the disk image with custom files and scripts

- You can add more packages to the disk image by updating the `post-installation.sh` script.
- To add files from host to the disk image you can add a file provisioner with source as path in host and destination as path in the image.

```hcl
provisioner "file" {
    destination = "/home/gem5/"
    source      = "path/to/files"
  }
```

### Extending Disk Image Size

If you need to increase the size of the disk image when adding more libraries and files, follow these steps:

1. **Update the Partition Size in `user-data`:**
   - Modify the `size` of the partition in the relevant `http/*/user-data` file. The size in `user-data` is specified in **bytes**.

2. **Update the `disk_size` Parameter in Packer:**
   - Adjust the `disk_size` parameter in the Packer script to be at least **1 MB more** than the total size specified in the `user-data` file. Note that `disk_size` in Packer is specified in **megabytes**.

#### Example: Setting the Main Partition Size to 10 GB

To ensure that the **main partition** is exactly **10 GB**, follow these steps:

1. **Calculate the Total Disk Size:**
   - The total disk size needs to account for the main partition, boot partition, and additional space required by Packer.
   - Add the respective boot partition size and Packer’s required space to **10 GB (10,000,000,000 bytes)**:
     - **x86 boot partition:** `1,048,576 bytes` (1 MB).
     - **ARM boot partition:** `564,133,888 bytes`.
     - Additional space required: `1,048,576 bytes` (1 MB).
   - Compute the total disk size:
     - **x86 disk image:** `10,000,000,000 + 1,048,576 + 1,048,576 = 10,002,097,152 bytes`.
     - **ARM disk image:** `10,000,000,000 + 564,133,888 + 1,048,576 = 10,565,182,464 bytes`.

2. **Update the Packer `disk_size`:**
   - Set `disk_size` to `10,003` MB for **x86**.
   - Set `disk_size` to `10,566` MB for **ARM**.

3. **Update the `user-data` Storage Section:**

   **For x86 Disk Images:**

   ```yaml
   storage:
     config:
     - ptable: gpt
       path: /dev/vda
       wipe: superblock-recursive
       preserve: false
       name: ''
       grub_device: true
       type: disk
       id: disk-vda
     - device: disk-vda
       size: 1048576    # size of boot partition
       flag: bios_grub
       number: 1
       preserve: false
       grub_device: false
       offset: 1048576
       type: partition
       id: partition-0
     - device: disk-vda
       size: 10000000000    # size of main partition (10 GB)
       wipe: superblock
       number: 2
       preserve: false
       grub_device: false
       offset: 2097152
       type: partition
       id: partition-1
     - fstype: ext4
       volume: partition-1
       preserve: false
       type: format
       id: format-0
     - path: /
       device: format-0
       type: mount
       id: mount-0
   ```

   **For ARM Disk Images:**

   ```yaml
   storage:
     config:
     - ptable: gpt
       path: /dev/vda
       wipe: superblock-recursive
       preserve: false
       name: ''
       grub_device: false
       type: disk
       id: disk-vda
     - device: disk-vda
       size: 564133888    # size of boot partition
       wipe: superblock
       flag: boot
       number: 1
       preserve: false
       grub_device: true
       type: partition
       id: partition-0
     - fstype: fat32
       volume: partition-0
       preserve: false
       type: format
       id: format-0
     - device: disk-vda
       size: 10000000000    # size of main partition (10 GB)
       wipe: superblock
       flag: ''
       number: 2
       preserve: false
       grub_device: false
       type: partition
       id: partition-1
     - fstype: ext4
       volume: partition-1
       preserve: false
       type: format
       id: format-1
     - path: /
       device: format-1
       type: mount
       id: mount-1
     - path: /boot/efi
       device: format-0
       type: mount
       id: mount-0
   ```

To take a pre-built image and add new files or packages, take a look at the following [documentation](https://www.gem5.org/documentation/gem5-stdlib/extending-disk-images).

## Creating a Disk Image from Scratch

### Automated Ubuntu Installation

- **Ubuntu Autoinstall**: We leveraged Ubuntu's autoinstall feature for an automated setup process.
- **Acquire `user-data` File**: To get the `user-data` file, install your desired Ubuntu version on a machine or VM. Post-installation, retrieve the `autoinstall-user-data` from `/var/log/installer/autoinstall-user-data` after the system's first reboot.
The `user-data` file in this repo, is made by selecting all default options except a minimal server installation.

### Configuration and Directory Structure

- **Determine QEMU Arguments**: Identify the QEMU arguments required for booting the system. These vary by ISA and mirror the arguments used for booting a disk image in QEMU.
- **Directory Organization**: Arrange your source directory to include the `user-data` file and any additional content. Utilize the `provisioner` section for transferring extra files into the disk image, ensuring all necessary resources are embedded within your custom disk image.

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

Useful documentation: <https://ubuntu.com/server/docs/install/autoinstall>
