# RISCV Full System

This document provides instructions to create a riscv disk image, a riscv boot loader (`berkeley bootloader (bbl)`) and also points to the associated gem5 scripts to run riscv Linux full system simulations.
The boot loader `bbl` is compiled with a Linux kernel and a device tree as well.

The used disk image is based on [busybox](https://busybox.net/) and [UCanLinux](https://github.com/UCanLinux/). It is built using the instructions, mostly from [here](https://github.com/UCanLinux/riscv64-sample).

All components are cross compiled on an x86 host using a riscv tool chain.

We assume the following directory structure while following the instructions in this README file:

```
riscv-fs/
  |___ gem5/                                   # gem5 source code (to be cloned here)
  |
  |___ riscv-disk                              # built disk image will go here
  |
  |___ device.dts                              # device tree file to use with bbl
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
  |___ configs-riscv-fs
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

***Note:** The above step is necessary and might cause errors while cross compiling different components for riscv if other methods are used to point to the toolchain.

## UCanLinux Source

Clone the `UCanLinux source.`

```sh
# going back to base riscv-fs directory
cd ../

git clone https://github.com/UCanLinux/riscv64-sample
```

This source contains already built bootloader and disk images as well. Though the given disk image might be usable with gem5, the `bbl` (bootloader image) will not work with gem5 and we need to compile `bbl` with an input device tree (`.dts`) file separately. The following sections provide instructions to build both `bbl` and disk images.

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

# copy the device tree file from riscv-fs
cp ../../../device.dts .

../configure --host=riscv64-unknown-linux-gnu --with-payload=../../linux/vmlinux --prefix=/opt/riscv/ --with-dts=device.dts
make -j$(nproc)

chmod 755 bbl

# optional: strip the bbl binary
riscv64-unknown-linux-gnu-strip bbl
```

This will produce a `bbl` bootloader binary with linux kernel in `riscv-pk/build` directory.

## Busy Box

Clone and compile the busybox:

```sh
# going back to riscv64-sample directory
cd ../..
git clone git://busybox.net/busybox.git
cd busybox
git checkout 1_30_stable  # checkout the latest stable branch
make menuconfig
cp ../sample/busybox.config .config  # optional
make menuconfig
make CROSS_COMPILE=riscv64-unknown-linux-gnu- all -j$(nproc)
make CROSS_COMPILE=riscv64-unknown-linux-gnu- install
```

## Root File System for Disk Image

Next, we will be setting up a root file system:

```sh
# going back to riscv64-sample directory
cd ../..

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
cd ../../../
cd gem5/util/m5
scons -C util/m5 build/riscv/out/m5
cp build/riscv/out/m5 ../../../RootFS/sbin/
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

sudo cp -a RootFS/* /mnt/rootfs

sudo chown -R -h root:root /mnt/rootfs/
df /mnt/rootfs
# make sure you are in riscv64-sample dir
cd ../riscv64-sample
sudo umount /mnt/rootfs
```

The disk image `riscv_disk` is ready to use.

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

gem5 scripts which can configure a riscv full system and run simulation are available in configs-riscv-fs/.
The main script `run_riscv.py` expects following arguments:

**bbl:** path to the bbl (berkeley bootloader) binary with kernel payload.

**disk:** path to the disk image to use.

**cpu_type:** cpu model (`atomic`, `simple`).

**num_cpus:** number of cpu cores.

An example use of this script is the following:

```sh
[gem5 binary] -re configs/run_exit.py [path to bbl] [path to the disk image] atomic 4
```

To interact with the simulated system's console:

```sh
telnet localhost 3457 (this port number comes from `simerr` file)
```

The default linux system based on this README, has both `login` and `password` set as `root`.
