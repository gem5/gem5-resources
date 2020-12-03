---
title: Linux boot-exit image
tags:
    - x86
    - fullsystem
layout: default
permalink: resources/boot-exit
shortdoc: >
    Resources to build a disk image and run "boot-exit" test.
author: ["Ayaz Akram"]
---

This document provides instructions to create the "boot-exit" image, the Linux kernel binaries, and also points to the gem5 configuration files needed to run the boot.
The boot-exit disk image is based on Ubuntu 18.04 and has its `.bashrc` file modified in such a way that the guest OS terminates the simulation (using the `m5 exit` instruction) as soon as the system boots.

We assume the following directory structure while following the instructions in this README file:

```
boot-exit/
  |___ gem5/                                   # gem5 source code (to be cloned here)
  |
  |___ disk-image/
  |      |___ build.sh                         # the script downloading packer binary and building the disk image
  |      |___ shared/                          # Auxiliary files needed for disk creation
  |      |___ boot-exit/
  |            |___ boot-exit-image/           # Will be created once the disk is generated
  |            |      |___ boot-exit           # The generated disk image
  |            |___ boot-exit.json             # The Packer script
  |            |___ exit.sh                    # Exits the simulated guest upon booting
  |            |___ post-installation.sh       # Moves exit.sh to guest's .bashrc
  |
  |___ configs
  |      |___ system                           # gem5 system config files
  |      |___ run_exit.py                      # gem5 run script
  |
  |___ linux                                   # Linux source will be downloaded in this folder
  |
  |___ README.md                               # This README file
```


## Disk Image

Assuming that you are in the `src/boot-exit/` directory (the directory containing this README), first build `m5` (which is needed to create the disk image):

```sh
git clone https://gem5.googlesource.com/public/gem5
cd gem5/util/m5
scons build/x86/out/m5
```

Next (within the `src/boot-exit/` directory),

```sh
cd disk-image
./build.sh          # the script downloading packer binary and building the disk image
```

If you see errors or warnings from `packer validate` you can modify the file `disk-image/boot-exit/boot-exit.json` to update the file.
Specifically, you may see the following error.

```
Error: Failed to prepare build: "qemu"

1 error(s) occurred:

* Bad source '../gem5/util/m5/build/x86/out/m5': stat
../gem5/util/m5/build/x86/out/m5: no such file or directory
```

In this case, the `gem5` directory is in a different location than when this script was written.
You can update the following line in the `boot-exit.json` file.

```
         "destination": "/home/gem5/",
-        "source": "../gem5/util/m5/build/x86/out/m5",
+        "source": "<your path to gem5>/util/m5/build/x86/out/m5",
         "type": "file"
```

Once this process succeeds, the disk image can be found on `boot-exit/boot-exit-image/boot-exit`.
A disk image already created following the above instructions can be found, gzipped, [here](http://dist.gem5.org/dist/v21-1/images/x86/ubuntu-18-04/boot-exit.img.gz).


## gem5 Run Scripts

gem5 scripts which configure the system and run simulation are available in configs-boot-tests/.
The main script `run_exit.py` expects following arguments:

**kernel:** path to the Linux kernel (see `src/linux-kernel/` for information on building/obtaining a linux kernel binary).

**disk:** path to the disk image.

**cpu_type:** cpu model (`kvm`, `atomic`, `simple`, `o3`).

**mem_sys:** memory system (`classic`, `MI_example`, `MESI_Two_Level`, `MOESI_CMP_directory`).

**num_cpus:** number of cpu cores.

**boot_type:** Linux kernel boot type (`init`, `systemd`).

An example use of this script is the following:

```sh
[gem5 binary] configs/run_exit.py [path to the Linux kernel] [path to the disk image] kvm classic 4 init
```

## Working Status

Working status of these tests for gem5-20 can be found [here](https://www.gem5.org/documentation/benchmark_status/gem5-20).
