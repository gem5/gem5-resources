---
title: Linux arm-ubuntu image
tags:
    - arm
    - fullsystem
layout: default
permalink: resources/arm-ubuntu
shortdoc: >
    Resources to build a generic arm-ubuntu disk image.
author: ["Hoa Nguyen", "Kaustav Goswami"]
---

This document provides instructions to create the "arm-ubuntu" image and
points to the gem5 component that would work with the disk image. The
arm-ubuntu disk image is based on Ubuntu's server cloud image for
arm available at (https://cloud-images.ubuntu.com/focal/current/).
The `.bashrc` file would be modified in such a way that it executes
a script passed from the gem5 configuration files (using the `m5 readfile`
instruction).

The instructions for bringing up QEMU emulation are based on
[Ubuntu's Wiki](https://wiki.ubuntu.com/ARM64/QEMU),
and the instructions for creating a cloud disk image are based on
[this guide](https://gist.github.com/oznu/ac9efae7c24fd1f37f1d933254587aa4).
More information about cloud-init can be found
[here](https://cloudinit.readthedocs.io/en/latest/topics/examples.html).

We assume the following directory structure while following the instructions
in this README file:

```
arm-ubuntu/
  |___ gem5/                                   # gem5 source code (to be cloned here)
  |
  |___ disk-image/
  |      |___ aarch64-ubuntu.img               # The ubuntu disk image should be downloaded and modified here
  |      |___ shared/                          # Auxiliary files needed for disk creation
  |      |      |___ serial-getty@.service     # Auto-login script
  |      |___ arm-ubuntu/
  |             |___ cloud.txt                 # the cloud config, to be created
  |             |___ gem5_init.sh              # The script to be appended to .bashrc on the disk image
  |             |___ post-installation.sh      # The script manipulating the disk image
  |             |___ arm-ubuntu.json           # The Packer script
  |
  |
  |___ README.md                               # This README file
```

## Building the disk image

This requires an ARM cross compiler to be installed. The disk image is a 64-bit
ARM 64 (aarch64) disk image. Therefore, we only focus on the 64-bit version of
the cross compiler. It can be installed by:

```sh
sudo apt-get install g++-10-aarch64-linux-gnu gcc-10-aarch64-linux-gnu
```

In order to build the ARM based Ubuntu disk-image for with gem5, build the m5
utility in `gem5/util/m5` using the following:

```sh
git clone https://gem5.googlesource.com/public/gem5
cd gem5/util/m5
scons build/arm64/out/m5
```

Troubleshooting: You may need to edit the SConscript to point to the correct
cross compiler.
```
...
main['CXX'] = '${CROSS_COMPILE}g++-10'
...
```

# Installing QEMU for aarch64

On the host machine,

```sh
sudo apt-get install qemu-system-arm qemu-efi
```

# Installing cloud utilities

On the host machine,

```sh
sudo apt-get install cloud-utils
```

# Downloading the cloud disk image

In the `arm-ubuntu/disk-image/` directory,

```sh
wget https://cloud-images.ubuntu.com/focal/current/focal-server-cloudimg-arm64.img
```

# Booting the disk image using QEMU

## Generating an SSH key pair
```sh
ssh-keygen -t rsa -b 4096
```
Leave all prompted fields empty and hit enter. This will generate a public and
private key pair in files `~/.ssh/id_rsa` and `~/.ssh/id_rsa.pub`. If your username
is different from what is printed by command `whoami`, then add `-C "<username>*<hostname>"`
to the command:
```sh
ssh-keygen -t rsa -b 4096 -C "<username>*<hostname>"
```

## Making a cloud config file

First, we need a cloud config that will have the authorization information
keys for logging in the machine.

In the `arm-ubuntu/disk-image/arm-ubuntu` directory, create a `cloud.txt` file
with the following content,

```sh
#cloud-config
users:
  - name: ubuntu                            <- change this name to the current user (use `whoami`)
    ssh-authorized-keys:
      - ssh-rsa AAAAABBCCCCCCCrNJfweeeeee   <- insert the rsa key here (typically `cat ~/.ssh/id_rsa.pub`)
    sudo: ['ALL=(ALL) NOPASSWD:ALL']
    groups: sudo
    shell: /bin/bash
    homedir: /home/ubuntu                   <- change this to the home directory of `whoami`
```

* Note: Do not leave stray spaces in the `cloud.txt` file.
* If your username is different from what is printed by command `whoami`, then use that for parameter "name" in cloud.txt

More information about generating an ssh rsa key is available
[here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key).
You can ignore the GitHub email address part.

## Booting the cloud disk image with the cloud config file
In the `arm-ubuntu/disk-image` directory,

```sh
dd if=/dev/zero of=flash0.img bs=1M count=64
dd if=/usr/share/qemu-efi/QEMU_EFI.fd of=flash0.img conv=notrunc
dd if=/dev/zero of=flash1.img bs=1M count=64
cloud-localds --disk-format qcow2 cloud.img arm-ubuntu/cloud.txt
wget https://releases.linaro.org/components/kernel/uefi-linaro/latest/release/qemu64/QEMU_EFI.fd
qemu-system-aarch64 \
    -smp 2 \
    -m 1024 \
    -M virt \
    -cpu cortex-a57 \
    -bios QEMU_EFI.fd \
    -nographic \
    -device virtio-blk-device,drive=image \
    -drive if=none,id=image,file=focal-server-cloudimg-arm64.img \
    -device virtio-blk-device,drive=cloud \
    -drive if=none,id=cloud,file=cloud.img \
    -netdev user,id=user0 -device virtio-net-device,netdev=eth0 \
    -netdev user,id=eth0,hostfwd=tcp::5555-:22
```

## Manipulating the disk image

When the qemu instance has fully booted, cloud-init has completed, and while it
is still running, we will use Packer to connect to the virtual machine and
manipulate the disk image. Before doing that, we need to add the private key using `ssh-add`.

```sh
ssh-add ~/.ssh/id_rsa
```

If the image was booted in qemu on a port number other than 5555, edit the `ssh_port`
parameter in `arm-ubuntu/arm-ubuntu.json` accordingly. The disk manipulation
process is automated. If your username is different from what is printed by
command `whoami`, then edit `build.sh` and change the value of `USER` to `"<your_username>"`.
Then in the `arm-ubuntu/disk-image/` directory,

```sh
chmod +x build.sh
./build.sh
```

`build.sh` also verifies the cloud.txt and modifies the arm-ubuntu.json
accordingly. The packer script, executed by `build.sh` disables systemd. In
case you need to enable systemd stuff, remove the last two provisioners from
the arm-ubuntu.json file.

Note that after executing the packer script, you will not be able to emulate
this disk image in qemu.

## Preparing the disk image for gem5

We need to finalize the image before we can use it with gem5. This is done by:

```sh
qemu-img convert -f qcow2 -O raw focal-server-cloudimg-arm64.img arm64-ubuntu-focal-server.img
rm focal-server-cloudimg-arm64.img
```
