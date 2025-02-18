---
title: Hack Back Checkpointing Disk Image
tags:
    - x86
    - fullsystem
permalink: resources/hack-back
shortdoc: >
  This resource creates a disk image for which you can create a checkpoint after linux boot
  and then restore using a different scriptfile.
author: ["Ayaz Akram"]
---

This document provides instructions to create a disk image with `hack_back_ckpt.rcS` script (located in `gem5/configs/boot/`) installed.
This script creates a checkpoint once the Linux system boots up.
Using environment variables the script makes sure that the checkpoint is not created if one already exists.
On restoring the simulation from the checkpoint, a new script can be provided to execute the desired applications.

**Note:** The instructions in this README are based on experiments with gem5-20.

We assume the following directory structure while following the instructions in this README file:

```
npb/
  |___ gem5/                                # gem5 source code
  |
  |___ disk-image/
  |      |___ build.sh                      # The script downloading packer binary and building the disk image
  |      |___ shared/                       # Auxiliary files needed for disk creation
  |      |___ hack-back/
  |            |___ hack-back-image/        # Will be created once the disk is generated
  |            |      |___ hack-back        # The generated disk image
  |            |___ hack-back.json          # The Packer script to build the disk image
  |            |___ hack_back_ckpt.rcS      # Main script responsible for checkpointing
  |            |___ post-installation.sh    # Moves hack_back_ckpt.rcS to guest's .bashrc
  |
  |
  |___ README.md                           # This README file
```

## Disk Image

Assuming that you are in the `src/hack-back/` directory (the directory containing this README), first build `m5` (which is needed to create the disk image):

```sh
git clone https://gem5.googlesource.com/public/gem5
cd gem5/util/m5
scons build/x86/out/m5
```

Next,

```sh
cd disk-image
./build.sh
```

Once this process succeeds, the created disk image can be found on `hack-back/hack-back-image/hack-back`.

## Using this Disk Image

The details of how to use this disk image are following:

- On starting a gem5 simulation using this disk image, after the kernel boot, the hack back script is the first thing which will run and exit the simulation with an `exit_event` cause of `checkpoint`.
- Your gem5 run script should be able to recognize this `exit_event` and take a checkpoint using `m5.checkpoint(m5.options.outdir)`.
- Later, on restoring the simulation from the checkpoint, you should be able to pass a new script to run your benchmarks, which will start simulating your benchmarks right from the point where the checkpoint was taken.
- To restore from the checkpoint (assuming that it is in the gem5 outdir), use `m5.instantiate(m5.options.outdir)`.
- Your benchmark script can be passed using `system.readfile=[path to the script]` in your gem5 run script.
