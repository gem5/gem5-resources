---
title: Linux riscv-ubuntu image
tags:
    - riscv
    - fullsystem
layout: default
permalink: resources/riscv-ubuntu
shortdoc: >
    Resources to build a generic riscv-ubuntu disk image.
author: ["Hoa Nguyen"]
---

This document provides instructions to create the "riscv-ubuntu" image and
points to the gem5 component that would work with the disk image. The
riscv-ubuntu disk image is based on Ubuntu's preinstalled server image for
RISC-V SiFive HiFive Unmatched available at
(https://cdimage.ubuntu.com/releases/20.04.3/release/).
The `.bashrc` file would be modified in such a way that it executes
a script passed from the gem5 configuration files (using the `m5 readfile`
instruction).

We assume the following directory structure while following the instructions in this README file:

```
riscv-ubuntu/
  |___ gem5/                                   # gem5 source code (to be cloned here)
  |
  |___ riscv-gnu-toolchain/
  |
  |___ qemu/
  |
  |___ disk-image/
  |      |___ ubuntu.img                       # The ubuntu disk image should be downloaded and modified here
  |      |___ shared/                          # Auxiliary files needed for disk creation
  |      |      |___ serial-getty@.service     # Auto-login script
  |      |___ riscv-ubuntu/
  |             |___ gem5_init.sh              # The script to be appended to .bashrc on the disk image
  |             |___ post-installation.sh      # The script manipulating the disk image
  |             |___ riscv-ubuntu.json         # The Packer script
  |             |___ exit.sh                  # A script that calls m5 exit
  |
  |
  |___ README.md                               # This README file
```

# Installing the RISCV toolchain and QEMU

```sh
# Install QEMU dependencies
sudo apt-get install ninja-build

cd riscv-ubuntu/

# QEMU
git clone https://github.com/qemu/qemu --recursive
cd qemu
git checkout 0021c4765a6b83e5b09409b75d50c6caaa6971b9
./configure --target-list=riscv64-softmmu
make -j $(nproc)
make install

cd ..

# RISCV toolchain
git clone https://github.com/riscv-collab/riscv-gnu-toolchain --recursive
cd riscv-gnu-toolchain
git checkout 1a36b5dc44d71ab6a583db5f4f0062c2a4ad963b
# --prefix parameter specifying the installation location
./configure --prefix=/opt/riscv
make linux -j $(nproc)

cd ..
```

# Downloading the Preinstalled Disk Image

There are more versions of Ubuntu that are supported for RISCV, they
are available at (https://wiki.ubuntu.com/RISC-V).
In the following command, we will use the Ubuntu 20.04.4 disk image.

```sh
# downloading the disk image
wget https://cdimage.ubuntu.com/releases/20.04.3/release/ubuntu-20.04.4-preinstalled-server-riscv64+unmatched.img.xz
# unpacking/decompressing the disk image
xz -dk ubuntu-20.04.4-preinstalled-server-riscv64+unmatched.img.xz
# renaming the disk image
mv ubuntu-20.04.4-preinstalled-server-riscv64+unmatched.img ubuntu.img
# adding 4GB to the disk
qemu-img resize -f raw ubuntu.img +4G
```

# Installing Ubuntu Packages Containing Necessary Files for Booting the Disk Image with QEMU

According to (https://wiki.ubuntu.com/RISC-V),

>  Prerequisites:
>
>    apt install qemu-system-misc opensbi u-boot-qemu qemu-utils
>
> u-boot-qemu version 2022.01 (or newer) is required to boot in qemu.

To use this version of u-boot-qemu, we will download the package from here,
(https://mirrors.edge.kernel.org/ubuntu/pool/main/u/u-boot/). The following command will
download and install the package.

```sh
wget https://mirrors.edge.kernel.org/ubuntu/pool/main/u/u-boot/u-boot-qemu_2022.01%2Bdfsg-2ubuntu2_all.deb
dpkg -i u-boot-qemu_2022.01+dfsg-2ubuntu2_all.deb
apt-get install -f
```

The correct version of u-boot-qemu is also available in the Ubuntu 22.04 or later.
```sh
apt install u-boot-qemu
```

The following command will install the rest of the dependencies,
```sh
apt install qemu-system-misc opensbi qemu-utils
```

# Download gem5 and Compiling m5

```sh
# Within the `src/riscv-ubuntu` directory.
git clone https://gem5.googlesource.com/public/gem5
cd gem5/util/m5
scons build/riscv/out/m5
cd ../../..
```

**Note**: the default cross-compiler is `riscv64-unknown-linux-gnu-`.
To change the cross-compiler, you can set the cross-compiler using the scons
sticky variable `riscv.CROSS_COMPILE`. For example,
```sh
scons riscv.CROSS_COMPILE=riscv64-linux-gnu- build/riscv/out/m5
```

# Booting the Disk Image with QEMU

The following qemu command will boot the system using the disk image and the
bootloader downloaded earlier.
```sh
./qemu/build/qemu-system-riscv64 -machine virt -nographic \
     -m 16384 -smp 8 \
     -bios /usr/lib/riscv64-linux-gnu/opensbi/generic/fw_jump.elf \
     -kernel /usr/lib/u-boot/qemu-riscv64_smode/uboot.elf \
     -device virtio-net-device,netdev=eth0 \
     -netdev user,id=eth0,hostfwd=tcp::5555-:22 \
     -drive file=ubuntu.img,format=raw,if=virtio
```
**Note:** the above command will forward the guest's port 22 to the host's
port 5555. This is done so that we can transfer and install benchmarks
to the guest system from the host via SSH (facilitated by Packer).

On the first boot, the guest OS will ask to input username and password.
The default username and password is,
```
Username: ubuntu
Password: ubuntu
```

After changing the password and login to the guest OS, you can stop cloud-init
and launch the SSH server,

```sh
sudo touch /etc/cloud/cloud-init.disabled # stop cloud-init
/etc/init.d/ssh start # start the SSH server
sudo apt-get update
sudo apt-get upgrade
```

# Building the disk image

The disk image can be built using (Packer)[https://www.packer.io/].

While the qemu instance is still running, we will use Packer to connect to the
virtual machine and manipulate the disk image.

```sh
cd riscv-ubuntu/
scp -P 5555 gem5/util/m5/build/riscv/out/m5 ubuntu@localhost:/home/ubuntu/
scp -P 5555 disk-image/shared/serial-getty@.service ubuntu@localhost:/home/ubuntu/
scp -P 5555 disk-image/riscv-ubuntu/gem5_init.sh ubuntu@localhost:/home/ubuntu/
scp -P 5555 disk-image/riscv-ubuntu/exit.sh ubuntu@localhost:/home/ubuntu/
```

Connecting to the guest,
```sh
ssh -p 5555 ubuntu@localhost
```

In the guest,
```sh
sudo -i
# input password

mv /home/ubuntu/serial-getty@.service /lib/systemd/system/

mv /home/ubuntu/m5 /sbin
ln -s /sbin/m5 /sbin/gem5

mv /home/ubuntu/gem5_init.sh /root/
chmod +x /root/gem5_init.sh
echo "/root/gem5_init.sh" >> /root/.bashrc

mv /home/ubuntu/exit.sh /root/
chmod +x /root/exit.sh
```

# Pre-built disk image

A pre-build, gzipped, disk image is available at <http://dist.gem5.org/dist/v22-1/images/riscv/ubuntu-20-04/riscv-ubuntu-20220318.img.gz>. **Note**: The password set for the `ubuntu` user is `helloworld`.

# Using the Disk Image
This disk image is used in the following gem5 example RISCV config files, found within the gem5 repository:
* `gem5/configs/example/gem5_library/riscv-fs.py`, which simulates a full system running with RISCV ISA.
* `gem5/configs/example/gem5_library/riscv-ubuntu-run.py`, which simulates a full system with RISCV based Ubuntu 20.04 disk-image. Upon successful start-up, a `m5_exit instruction encountered` is encountered. The simulation ends then.

Note: To use the locally built Ubuntu image in gem5, make sure to pass the `root_partition`
parameter to `CustomResource` or `CustomDiskImageResource`. For example,
```py
board.set_kernel_disk_workload(
    kernel=Resource("riscv-bootloader-vmlinux-5.10"),
    disk_image=CustomDiskImageResource(
        local_path="<RESOURCES_PARENT_PATH>/gem5-resources/src/riscv-ubuntu/ubuntu.img",
        disk_root_partition="1",
    )
)
```