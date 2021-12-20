---
title: RISC-V full system
tags:
    - fullsystem
    - riscv
layout: default
permalink: resources/riscv-fs
shortdoc: >
    Resources to build a riscv disk image, a riscv boot loader and points to the gem5 scripts to run riscv Linux FS simulations.
author: ["Ayaz Akram"]
---

# RISCV Full System

This document provides instructions to create a riscv disk image, a riscv boot loader (`berkeley bootloader (bbl)`) and also points to the associated gem5 scripts to run riscv Linux full system simulations.
The boot loader `bbl` is compiled with a Linux kernel and a device tree as well.

The used disk image is based on [busybox](https://busybox.net/) and [UCanLinux](https://github.com/UCanLinux/). It is built using the instructions, mostly from [here](https://github.com/UCanLinux/riscv64-sample).

**Note:** All components are cross compiled on an x86 host using a riscv tool chain. We used `88b004d4c2a7d4e4f08b17ee32d2` commit of the riscv tool chain source while building the source (riscv gcc version 10.2.0).

We assume the following directory structure while following the instructions in this README file:

```
riscv-fs/
  |___ gem5/                                   # gem5 source code (to be cloned here)
  |
  |___ riscv-disk                              # built disk image will go here
  |
  |___ riscv-gnu-toolchain                     # riscv tool chain for cross compilation
  |
  |___ riscv64-sample                          # UCanLinux source
  |       |__linux                             # linux source
  |       |__busybox                           # busybox source
  |       |__riscv-pk                          # riscv proxy kernel source (bbl)
  |       |__RootFS                            # root file system for disk image
  |
  |
  |___ configs
  |      |___ system                           # gem5 system config files
  |      |___ run_riscv.py                     # gem5 run script
  |
  |
  |___ README.md                               # This README file
```

## RISCV Toolchain

We use `RISC-V GNU Compiler Toolchain`. To build the toolchain, follow the following instructions, assuming you are in the `riscv-fs` directory.

```sh
# install required libraries
sudo apt-get install -y autoconf automake autotools-dev curl python3 libmpc-dev libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo gperf libtool patchutils bc zlib1g-dev libexpat-dev

# clone riscv gnu toolchain source
git clone https://github.com/riscv/riscv-gnu-toolchain
cd riscv-gnu-toolchain
git checkout 88b004d4c2a7d4e4f08b17ee32d2

# change the prefix to your directory
# of choice for installation of the
# toolchain
./configure --prefix=/opt/riscv

# build the toolchain
make linux -j$(nproc)
```

Update the `PATH` environment variable so that the following instructions can figure out where to find the riscv toolchain.

```sh
export PATH=$PATH:/opt/riscv/bin/
```

**Note:** The above step is necessary and might cause errors while cross compiling different components for riscv if other methods are used to point to the toolchain.

## UCanLinux Source

Clone the `UCanLinux source.`

```sh
# going back to base riscv-fs directory
cd ../

git clone https://github.com/UCanLinux/riscv64-sample
```

The following sections provide instructions to build both `bbl` and disk images.

## Linux Kernel

Clone the latest LTS Linux kernel (v5.10):

```sh
cd riscv64-sample/
git clone --depth 1 --branch v5.10 https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git
```

To configure and compile the kernel:

```sh
cd linux

# copy the kernel config from the riscv64-sample
# directory (cloned previously)

cp ../kernel.config .config

# configure the kernel and build it
make ARCH=riscv CROSS_COMPILE=riscv64-unknown-linux-gnu- menuconfig
make ARCH=riscv CROSS_COMPILE=riscv64-unknown-linux-gnu-  all -j$(nproc)
```

This should generate a `vmlinux` image in the `linux` directory.
A pre-built RISC-V 5.10 linux kernel can be downloaded [here](http://dist.gem5.org/dist/v21-2/kernels/riscv/static/vmlinux-5.10).

## Bootloader (bbl)

To build the bootloader, clone the RISCV proxy kernel (`pk`) source, which is an application execution environment and contains the bbl source as well.

```sh
# going back to base riscv64-sample directory
cd ../
git clone https://github.com/riscv/riscv-pk.git

cd riscv-pk

mkdir build
cd build

apt-get install device-tree-compiler

# configure bbl build
../configure --host=riscv64-unknown-linux-gnu --with-payload=../../linux/vmlinux --prefix=/opt/riscv/

make -j$(nproc)

chmod 755 bbl

# optional: strip the bbl binary
riscv64-unknown-linux-gnu-strip bbl
```

This will produce a `bbl` bootloader binary with linux kernel in `riscv-pk/build` directory.
A pre-built copy of this bootloard binary, with the linux kernel can be downloaded [here](http://dist.gem5.org/dist/v21-2/kernels/riscv/static/bootloader-vmlinux-5.10).

## Busy Box

Clone and compile the busybox:

```sh
# going back to riscv64-sample directory
cd ../..
git clone git://busybox.net/busybox.git
cd busybox
git checkout 1_30_stable  # checkout the latest stable branch
make menuconfig
cp ../busybox.config .config  # optional
make menuconfig
make CROSS_COMPILE=riscv64-unknown-linux-gnu- all -j$(nproc)
make CROSS_COMPILE=riscv64-unknown-linux-gnu- install
```

## Root File System for Disk Image

Next, we will be setting up a root file system:

```sh
# going back to riscv64-sample directory
cd ../

mkdir RootFS
cd RootFS
cp -a ../skeleton/* .

# copy linux tools/binaries from busbybox (created above)
cp -a ../busybox/_install/* .

# install modules from linux kernel compiled above
cd ../linux/
make ARCH=riscv CROSS_COMPILE=riscv64-unknown-linux-gnu- INSTALL_MOD_PATH=../RootFS modules_install

# install libraries from the toolchain built above
cd ../RootFS
cp -a /opt/riscv/sysroot/lib  .

# create empty directories
mkdir dev home mnt proc sys tmp var
cd etc/network
mkdir if-down.d  if-post-down.d  if-pre-up.d  if-up.d

# build m5 util for riscv and move
# it to the root file system as well
cd ../../../../
cd gem5/util/m5
scons build/riscv/out/m5
cp build/riscv/out/m5 ../../../riscv64-sample/RootFS/sbin/
```

**Note**: the default cross-compiler is `riscv64-unknown-linux-gnu-`. To change the cross-compiler, you can set the cross-compiler using the scons sticky variable `riscv.CROSS_COMPILE`. For example,
```sh
scons riscv.CROSS_COMPILE=riscv64-linux-gnu- build/riscv/out/m5
```
## Disk Image

Create a disk of 512MB size.

```sh
cd ../../../
dd if=/dev/zero of=riscv_disk bs=1M count=512
```

Making and mounting a root file system on the disk:

```sh
mkfs.ext2 -L riscv-rootfs riscv_disk

sudo mkdir /mnt/rootfs
sudo mount riscv_disk /mnt/rootfs

sudo cp -a riscv64-sample/RootFS/* /mnt/rootfs

sudo chown -R -h root:root /mnt/rootfs/
df /mnt/rootfs
sudo umount /mnt/rootfs
```

The disk image `riscv_disk` is ready to use.
A pre-built, gzipped, disk image can be downloaded [here](http://dist.gem5.org/dist/v21-2/images/riscv/busybox/riscv-disk.img.gz).

**Note:** If you need to resize the disk image once it is created, you can do the following:

```sh
e2fsck -f riscv_disk
resize2fs ./riscv_disk 512M
```

Also, if it is required to change the contents of the disk image, it can be mounted as:

```sh
mount -o loop riscv_disk [some mount directory]
```

## gem5 Run Scripts

gem5 scripts which can configure a riscv full system and run simulation are available in configs/.
The main script `run_riscv.py` expects following arguments:

**bbl:** path to the bbl (berkeley bootloader) binary with kernel payload (located at `riscv64-sample/riscv-pk/build/bbl`).

**disk:** path to the disk image to use (located at `riscv64-sample/riscv_disk`).

**cpu_type:** cpu model (`atomic` for AtomicSimpleCPU, `simple` for TimingSimpleCPU).

**num_cpus:** number of cpu cores.

An example use of this script is the following:

```sh
[gem5 binary] configs/run_riscv.py [path to bbl] [path to the disk image] atomic 1
```

To interact with the simulated system's console, you can use `telnet`,

```sh
telnet localhost <port>
```

Another option is to use `m5term` provided by gem5. To compile and launch `m5term`,
```sh
cd gem5/util/term
make                         # compiling
./m5term localhost <port>    # launching the terminal
```

The default linux system based on this README, has both `login` and `password` set as `root`.
