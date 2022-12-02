---
title: Linux x86-ubuntu image
tags:
    - x86
    - fullsystem
layout: default
permalink: resources/x86-ubuntu
shortdoc: >
    Resources to build a generic x86-ubuntu disk image and run a "boot-exit" test.
author: ["Ayaz Akram"]
---

This document provides instructions to create the "x86-ubuntu" image, the Linux kernel binaries, and also points to gem5 configuration files that use the image.
The x86-ubuntu disk image is based on Ubuntu 18.04 and has its `.bashrc` file modified in such a way that it executes a script passed from the gem5 configuration files (using the `m5 readfile` instruction).
The boot-exit test passes a script that causes the guest OS to terminate the simulation (using the `m5 exit` instruction) as soon as the system boots.

We assume the following directory structure while following the instructions in this README file:

```
x86-ubuntu/
  |___ gem5/                                   # gem5 source code (to be cloned here)
  |
  |___ disk-image/
  |      |___ build.sh                         # the script downloading packer binary and building the disk image
  |      |___ shared/                          # Auxiliary files needed for disk creation
  |      |___ x86-ubuntu/
  |            |___ x86-ubuntu-image/           # Will be created once the disk is generated
  |            |      |___ x86-ubuntu           # The generated disk image
  |            |___ x86-ubuntu.json             # The Packer script
  |            |___ exit.sh                    # Exits the simulated guest upon booting
  |            |___ post-installation.sh       # Moves exit.sh to guest's .bashrc
  |
  |___ linux                                   # Linux source will be downloaded in this folder
  |
  |___ README.md                               # This README file
```


## Disk Image

Assuming that you are in the `src/x86-ubuntu/` directory (the directory containing this README), first build `m5` (which is needed to create the disk image):

```sh
git clone https://gem5.googlesource.com/public/gem5
cd gem5/util/m5
scons build/x86/out/m5
```

Next (within the `src/x86-ubuntu/` directory),

```sh
cd disk-image
./build.sh          # the script downloading packer binary and building the disk image
```

If you see errors or warnings from `packer validate` you can modify the file `disk-image/x86-ubuntu/x86-ubuntu.json` to update the file.
Specifically, you may see the following error.

```
Error: Failed to prepare build: "qemu"

1 error(s) occurred:

* Bad source '../gem5/util/m5/build/x86/out/m5': stat
../gem5/util/m5/build/x86/out/m5: no such file or directory
```

In this case, the `gem5` directory is in a different location than when this script was written.
You can update the following line in the `x86-ubuntu.json` file.

```
         "destination": "/home/gem5/",
-        "source": "../gem5/util/m5/build/x86/out/m5",
+        "source": "<your path to gem5>/util/m5/build/x86/out/m5",
         "type": "file"
```

Once this process succeeds, the disk image can be found on `x86-ubuntu/x86-ubuntu-image/x86-ubuntu`.
A disk image already created following the above instructions can be found, gzipped, [here](http://dist.gem5.org/dist/v22-1/images/x86/ubuntu-18-04/x86-ubuntu.img.gz).


## Example Run Scripts

Within the gem5 repository, two example scripts are provided which utilize the x86-ubuntu resource.

The first is `configs/example/gem5_library/x86-ubuntu-run.py`.
This will boot the OS with a Timing CPU.
To run:

```sh
scons build/X86/gem5.opt -j`nproc`
./build/X86/gem5.opt configs/example/gem5_library/x86-ubuntu-run.py
```

The second is `configs/example/gem5_library/x86-ubuntu-run-with-kvm.py`.
This will boot the OS using KVM cores before switching to Timing Cores to run a simple echo command.
To run:

```sh
scons build/X86/gem5.opt -j`nproc`
./build/X86/gem5.opt configs/example/gem5_library/x86-ubuntu-run-with-kvm.py`
```

**Note:** the `x86-ubuntu-with-kvm.py` script requires a host machine with KVM to function correctly.
