---
title: LupV Booloader/Kernel and Disk image
tags:
    - lupio
    - riscv
    - fullsystem
layout: default
permalink: resources/lupv
shortdoc: >
    Sources for the LupV bootloader/kernel and disk image.
author: ["Joël Porquet-Lupine"]
---

This README will cover how to create a bootloader/kernel, and a disk image which
may be used to run LupV (LupIO with RISC-V) in gem5.

An example script which uses these resources is provided here: <https://gem5.googlesource.com/public/gem5/+/refs/tags/v22.0.0.0/configs/example/lupv/>.


## Toolchain

Install a 64-bit RISCV toolchain:

- Ubuntu: `apt-get install gcc-riscv64-linux-gnu`
- Arch: `pacman -S riscv64-linux-gnu-gcc`
- From source: https://github.com/riscv-collab/riscv-gnu-toolchain

## Firmware

The software stack is composed of two parts: the bootloader (riscv-pk) and the
kernel (linux). They are combined together in a single file.

### Linux kernel

First compile the linux kernel.

```terminal
$ git clone https://gitlab.com/luplab/lupio/linux
$ cd linux
$ git checkout lupio-dev-v5.8
$ make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- lupv_defconfig
$ make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- -j8
```

Note:
- Adapt `CROSS_COMPILE` to your toolchain's prefix
- Adapt `-j8` to the number of parallel jobs you want to run

The resulting kernel image is in `arch/riscv/boot/Image`.

### Bootloader

Now, compile the bootloader and include the linux image into the bootloader's
payload.

```terminal
$ git clone https://github.com/riscv-software-src/riscv-pk
$ cd riscv-pk
$ mkdir build && cd build
$ ../configure --host=riscv64-linux-gnu --with-payload=/path/to/linux/arch/riscv/boot/Image
$ make
```

The resulting firmware which includes the bootloader and the kernel, and can be
loaded as a kernel image in gem5, is in `build/bbl`.

<<<<<<< HEAD
A pre-built bootloader/kernel binary can be obtained from
[here](http://dist.gem5.org/dist/v21-2/kernels/riscv/static/lupio-linux).
=======
A pre-built bootloader/kernel binary can be obtained from [here](http://dist.gem5.org/dist/v22-1/kernels/riscv/static/lupio-linux).
>>>>>>> stable

## Root filesystem

The root filesystem is based on busybox. First, we need to compile busybox for
RISC-V, and then we need to create a complete directory tree structure.

Note that you need root access in order to create the root filesystem.

### Busybox

Copy `config_busybox_rv64_092021` file to your computer and compile busybox
using this configuration.

```terminal
$ git clone https://git.busybox.net/busybox/
$ cd busybox
$ mv /path/to/config_busybox_rv64_092021 .config
$ make -j8
```

The resulting busybox binary is `./busybox`.

### Rootfs image

Finally, create an entire ext2-formatted filesystem that is based on busybox.

First, populate the contents of this filesystem:

```terminal
$ mkdir -p rootfs/contents && cd rootfs/contents
$ mkdir -p bin etc dev lib proc sbin sys tmp usr usr/bin usr/lib usr/sbin
$ cp </path/to/busybox>/busybox bin/busybox
$ ln -s bin/busybox sbin/init
$ ln -s bin/busybox init
$ cat << EOF > etc/inittab
# Mount special filesystems
::sysinit:/bin/busybox mount -t proc proc /proc
::sysinit:/bin/busybox mount -t tmpfs tmpfs /tmp
# Remount root partition in RW
::sysinit:/bin/busybox mount -o remount,rw /
# Install all of the BusyBox applets
::sysinit:/bin/busybox --install -s
# Run shell directly
/dev/console::sysinit:-/bin/ash
EOF
$ sudo mknod dev/console c 5 1
$ cd ..
```

Then, create an image:

```terminal
$ qemu-img create rootfs.img 2M
$ echo ";" | sfdisk rootfs.img
$ sudo kpartx -a rootfs.img
$ sudo mkfs -t ext2 /dev/mapper/loop0p1
$ mkdir -p mountfs
$ sudo mount /dev/mapper/loop0p1 mountfs
$ sudo cp -a contents/* mountfs
$ sudo chown -R root:root mountfs
$ sudo umount mountfs
$ rm -rf mountfs
$ sudo kpartx -d rootfs.img
```

The resulting filesystem image, which can be loaded as a partition image in
gem5, is `rootfs.img`.

<<<<<<< HEAD
A pre-built, gzipped, image can be obtained
[here](http://dist.gem5.org/dist/v21-2/images/riscv/busybox/riscv-lupio-busybox.img.gz).
=======
A pre-built, gzipped, image can be obtained [here](http://dist.gem5.org/dist/v22-1/images/riscv/busybox/riscv-lupio-busybox.img.gz).
>>>>>>> stable
