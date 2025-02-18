---
title: RISC-V full system with no disk
tags:
    - riscv
    - fullsystem
    - nodisk
layout: default
permalink: resources/riscv-fs-nodisk
shortdoc: >
    Resources to build a riscv bootloader containing a linux kernel and a workload expected to run at early userspace.
author: ["Hoa Nguyen"]
---

This resource provides the possibility of conducting a RISC-V full system
simulation without a block device by leveraging
[Linux's userspace support] (https://www.kernel.org/doc/html/latest/driver-api/early-userspace/early_userspace_support.html).

# Overview

This document provides instructions to create a RISCV bootloader
(`berkeley bootloader (bbl)`) and also points to the associated gem5 scripts to
run riscv Linux full system simulations without using a disk image. The
bootloader `bbl` is compiled with a Linux kernel, a device tree, and a
workload. Similar to the `riscv-fs` resource, we'll also rely on
[BusyBox](https://www.busybox.net/) for basic Linux utilities, on
`UCanLinux` for the configuration of the Linux kernel and the configuration of
BusyBox, and on `riscv-pk` for building a proxy kernel.

```
riscv-fs-nodisk/
  |___ gem5/                                   # gem5 source code (to be cloned here)
  |
  |___ riscv-gnu-toolchain/                    # riscv tool chain for cross compilation
  |
  |___ riscv64-sample/                         # UCanLinux source
  |
  |___ linux/                                  # linux source
  |
  |___ busybox/                                # busybox source
  |
  |___ riscv-pk/                               # riscv proxy kernel source (bbl)
  |
  |___ cpio/                                   # contains the .cpio files
  |
  |___ initdir/                                # contains the structure of initramfs
  |
  |___ configs/
  |      |___ system                           # gem5 system config files
  |      |___ run_riscv.py                     # gem5 run script
  |
  |___ README.md                               # This README file
```

# How does it work?

When Linux kernel booting process takes place, `initramfs`, a root filesystem
embedded into the kernel, will be loaded to memory. When `initramfs` is loaded,
the kernel will try to execute one of the following scripts located in that
filesystem,
[{`/init, /sbin/init, /etc/init, /bin/init, /bin/sh`}](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/init/main.c?h=v5.10&id=2c85ebc57b3e1817b6ce1a6b703928e113a90442#n1467).
Instead of using the default `/init` script, we will use our version of `/init`
to execute the desired workload right after the early userspace is loaded.

**Note:** Since the `initramfs` decompressing process takes place while
Linux kernel is booting (which means it will happen *during* the full system
simulation), we'll try to minimize the size of the `initramfs`.

# Building the resource
## Step 1. Building the `riscv-gnu-toolchain`
In this step, we'll use
[GNU toolchain for RISC-V](https://github.com/riscv-collab/riscv-gnu-toolchain).

This step is necessary if you do not have basic libraries built for RISCV or
if you're cross-compiling RISCV.

```sh
cd riscv-fs-nodisk/
git clone https://github.com/riscv-collab/riscv-gnu-toolchain --recursive
cd riscv-gnu-toolchain
git checkout 1a36b5dc44d71ab6a583db5f4f0062c2a4ad963b
# --prefix parameter specifying the installation location
./configure --prefix=/opt/riscv
make linux -j $(nproc)
```

To update the PATH environment variable so that the RISCV compilers can be
found,
```sh
export PATH=$PATH:/opt/riscv/bin/
```

## Step 2. Getting the `UCanLinux` Source
This repo contains a Linux configuration for RISCV at
`riscv64-sample/kernel.config` and a BusyBox configuration at
`riscv64-sample/busybox.config`.

```sh
# going back to base riscv-fs directory
cd riscv-fs-nodisk/
git clone https://github.com/UCanLinux/riscv64-sample
```

## Step 3. Getting and Building `busybox`
More information about Busybox is [here](https://www.busybox.net/).
```sh
cd riscv-fs-nodisk/
git clone git://busybox.net/busybox.git
cd busybox
git checkout 1_34_stable  # checkout the a stable branch
cp ../riscv64-sample/busybox.config .config
yes "" | make CROSS_COMPILE=riscv64-unknown-linux-gnu- oldconfig
make CROSS_COMPILE=riscv64-unknown-linux-gnu- all -j$(nproc)
make CROSS_COMPILE=riscv64-unknown-linux-gnu- install
```
The files of interest are in `busybox/_install/bin`.

## Step 4. Getting and Compiling the `Linux kernel`
We'll compiling the Linux kernel to get the `linux/usr/gen_init_cpio`, which
would be used later.
```sh
cd riscv-fs-nodisk/
git clone --depth 1 --branch v5.10 https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git
cd linux
cp ../riscv64-sample/kernel.config .config
yes "" | make ARCH=riscv CROSS_COMPILE=riscv64-unknown-linux-gnu- oldconfig
make ARCH=riscv CROSS_COMPILE=riscv64-unknown-linux-gnu- menuconfig
# Go to "General setup --->"
#   Check on "Initial RAM filesystem and RAM disk (initramfs/initrd) support"
make ARCH=riscv CROSS_COMPILE=riscv64-unknown-linux-gnu- all -j $(nproc)
```

## Step 5. Compiling the Workload (e.g. gem5's m5)
```sh
cd riscv-fs-nodisk/
git clone https://gem5.googlesource.com/public/gem5
cd gem5/util/m5
scons build/riscv/out/m5
```

**Note**: the default cross-compiler is `riscv64-unknown-linux-gnu-`.
To change the cross-compiler, you can set the cross-compiler using the scons
sticky variable `riscv.CROSS_COMPILE`. For example,
```sh
scons riscv.CROSS_COMPILE=riscv64-linux-gnu- build/riscv/out/m5
```

## Step 6. Determining the Structure of `initramfs`
```sh
cd riscv-fs-nodisk/
mkdir cpio
mkdir misc
mkdir initdir
```
### Userspace
We'll use the `riscv64-sample/initdir` to define the structure of `initramfs`.
```sh
cd riscv-fs-nodisk/initdir
cp -r ../busybox/_install/bin/ .
mkdir lib
cp /opt/riscv/sysroot/lib/ld-linux-riscv64-lp64d.so.1 lib/ # busybox' dependency
cp /opt/riscv/sysroot/lib/libc.so.6 lib/ # busybox' dependency
cp /opt/riscv/sysroot/lib/libm.so.6 lib/ # busybox' dependency
cp /opt/riscv/sysroot/lib/libresolv.so.2 lib/ # busybox' dependency
mkdir proc
mkdir sys
mkdir sbin
cp ../gem5/util/m5/build/riscv/out/m5 sbin/m5 # replace m5 by the desired workload
```

Create `initdir/init` script with the following content,
```
#!/bin/busybox sh

exec /sbin/init # script to execute the workload
```

Create `initdir/sbin/init` script with the following content,
```
#!/bin/busybox sh

/sbin/m5 exit
```

Make the scripts executable,
```sh
chmod +x init
chmod +x sbin/init
```

To create the cpio file of the `initdir` folder,
```sh
cd riscv-fs-nodisk/linux
usr/gen_initramfs.sh -o ../cpio/disk.cpio ../initdir/
lsinitramfs ../cpio/disk.cpio # checking the file structure of the created cpio file
```

### `/dev/` folder
By default, `initramfs` would have a `/dev/console` and `/dev/tty`. Without
these devices, we cannot see what is written to `stdout` and `stderr`.

The following commands will build a `.cpio` file with `/dev/console` and
`/dev/tty`,
```sh
cd riscv-fs-nodisk/misc
mkdir dev
fakeroot -- mknod -m 622 dev/console c 5 1
fakeroot -- mknod -m 622 dev/tty c 5 0
fakeroot -- mknod -m 622 dev/ttyprintk c 5 3
fakeroot -- mknod -m 622 dev/null c 1 3
fakeroot -- find . -print0 | cpio --owner root:root --null -o --format=newc > ../cpio/dev.cpio
cd ../
rm -r misc
```
**Note:** `mknod -m 622 /dev/tty c 5 0` means we're creating `/dev/tty` with
permission of `622`. `c` means a character device being created, `5` is the
major number, and `0` is the minor number. More information about the
major/minor numbering is available at
(https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/Documentation/admin-guide/devices.txt).

### Merging .cpio files to a single .cpio file
```sh
cd riscv-fs-nodisk/cpio
cat disk.cpio dev.cpio > init.cpio
```

## Step 7. Compiling `Linux Kernel` with a customized `initramfs`
```sh
cd riscv-fs-nodisk/linux
make ARCH=riscv CROSS_COMPILE=riscv64-unknown-linux-gnu- menuconfig
# Go to "General setup --->"
#   Check on "Initial RAM filesystem and RAM disk (initramfs/initrd) support"
#   Change "Initramfs source file(s)" to the absoblute path of riscv-fs-nodisk/cpio/init.cpio
make ARCH=riscv CROSS_COMPILE=riscv64-unknown-linux-gnu- all -j $(nproc)
```

The file of interest is at `arch/riscv/boot/Image`.

## Step 8. Compiling `bbl` with the Linux kernel as the payload
```sh
cd riscv-fs-nodisk/
git clone https://github.com/riscv/riscv-pk.git
cd riscv-pk
mkdir build
cd build

# configure bbl build
../configure --host=riscv64-unknown-linux-gnu --with-payload=../../linux/arch/riscv/boot/Image --prefix=/opt/riscv/

make -j$(nproc)

chmod 755 bbl

riscv64-unknown-linux-gnu-strip bbl
cp bbl bbl-m5-exit
```

The desired bootloader is file is at `riscv-fs-nodisk/riscv-pk/build/bbl` or
`riscv-fs-nodisk/riscv-pk/build/bbl-m5-exit`.


## Example
This resource is used in gem5/SST integration.
For instructions to run the integeration, please refer to `ext/sst/README.md`
of the gem5 repo.
