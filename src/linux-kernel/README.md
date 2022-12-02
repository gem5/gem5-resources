---
title: Linux kernel configs for x86 and arm64
tags:
    - x86
    - arm64
    - fullsystem
permalink: resources/linux-kernel
shortdoc: >
    Linux kernel config files that have been tested with gem5.
---

# Creating Linux Kernel Binary

This document provides instructions to create a Linux kernel binary. The kernel
binary can be used during a gem5 Full System simulation. We assume the
following initial directory structure before following the instructions in this
README file:

```
Linux-kernel/
  |
  |___ linux-configs                           # Folder with Linux kernel configuration files
  |
  |___ README.md                               # This README file
```

## Linux Kernels

We have tested this resource compiling the following five LTS (long term
support) releases of the Linux kernel:

- 4.4.186
- 4.9.186
- 4.14.134
- 4.19.83
- 5.4.49

In addition, we also have compiled and tested the following LTS releases for
arm64 architecture only:

- 5.10.110
- 5.15.36

To compile the Linux binaries, follow these instructions (assuming that you are
in `src/Linux-kernel/` directory):

```sh
# will create a `linux` directory and download the initial kernel files into it.
git clone https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git

cd linux
# replace version with any of the above listed version numbers
git checkout v[version]

# copy the appropriate Linux kernel configuration file from linux-configs/
cp ../linux-configs/config.[arch].[version] .config

make -j`nproc`
```

After this process succeeds, the compiled Linux binary, named  `vmlinux`, can
be found in the `src/Linux-kernel/linux`. The final structure of the
`src/Linux-kernel/` directory will look as following:

```
Linux-kernel/
  |
  |___ linux-configs                           # Folder with Linux kernel configuration files
  |
  |___ linux                                   # Linux source
  |       |
  |       |___ vmlinux                         # The compiled Linux kernel binary
  |
  |___ README.md                               # This README file
```

**Note on x86 Kernel Binaries:** The above instructions for compiling x86
kernel binaries have been tested using the `gcc 7.5.0` compile

The pre-build compiled x86 Linux binaries can be downloaded from the following
links:

- [vmlinux-4.4.186](http://dist.gem5.org/dist/v22-1/kernels/x86/static/vmlinux-4.4.186)
- [vmlinux-4.9.186](http://dist.gem5.org/dist/v22-1/kernels/x86/static/vmlinux-4.9.186)
- [vmlinux-4.14.134](http://dist.gem5.org/dist/v22-1/kernels/x86/static/vmlinux-4.14.134)
- [vmlinux-4.19.83](http://dist.gem5.org/dist/v22-1/kernels/x86/static/vmlinux-4.19.83)
- [vmlinux-5.4.49](http://dist.gem5.org/dist/v22-1/kernels/x86/static/vmlinux-5.4.49)

**Note on ARM Kernel Binaries:** A cross-compiler is needed to compile ARM
kernel binaries on a x86 machine. We used `gcc 10.3.0
(`aarch64-linux-gnu-gcc-10)` to compile and test. In this resource, we are
limited to arm64 architecture only.

The pre-build compiled ARM Linux binaries can be downloaded from the following links:

- [arm64-vmlinux-4.4.186](http://dist.gem5.org/dist/v22-1/kernels/arm/static/arm64-vmlinux-4.4.186)
- [arm64-vmlinux-4.9.186](http://dist.gem5.org/dist/v22-1/kernels/arm/static/arm64-vmlinux-4.9.186)
- [arm64-vmlinux-4.14.134](http://dist.gem5.org/dist/v22-1/kernels/arm/static/arm64-vmlinux-4.14.134)
- [arm64-vmlinux-4.19.83](http://dist.gem5.org/dist/v22-1/kernels/arm/static/arm64-vmlinux-4.19.83)
- [arm64-vmlinux-5.10.110](http://dist.gem5.org/dist/v22-1/kernels/arm/static/arm64-vmlinux-5.10.110)
- [arm64-vmlinux-5.15.36](http://dist.gem5.org/dist/v22-1/kernels/arm/static/arm64-vmlinux-5.15.36)
- [arm64-vmlinux-5.4.49](http://dist.gem5.org/dist/v22-1/kernels/arm/static/arm64-vmlinux-5.4.49)

Alternatively to the vanilla kernel + linux-configs described above, it is also
possible to compile a gem5 port of the Linux Kernel 4.14.0 for ARM. The
repository can be cloned and built by following the following steps. It has a
gem5 specific config file located in `arch/arm64/configs/gem5\_defconfig` under
the `linux` directory.

```bash
git clone https://gem5.googlesource.com/arm/linux/
cd linux
make ARCH=<arch> CROSS_COMPILE=<cross_compiler> gem5_defconfig
make ARCH=<arch> CROSS_COMPILE=<cross_compiler> -j<num_cores>
```

**Licensing:**
Linux is released under the GNU General Public License version 2 (GPLv2), but
it also contains several files under other compatible licenses. For more
information about Linux Kernel Copy Right please refer to
[here](https://www.kernel.org/legal.html) and
[here](https://www.kernel.org/doc/html/latest/process/license-rules.html#kernel-licensing).
