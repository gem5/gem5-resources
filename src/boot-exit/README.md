# Linux boot-exit image

This document provides instructions to create the "boot-exit" image, the Linux kernel binaries, and also points to the gem5 configuration files needed to run the boot.
The boot-exit disk image is based on Ubuntu 18.04 and has its `.bashrc` file modified in such a way that the guest OS terminates the simulation (using the `m5 exit` instruction) as soon as the system boots.

We assume the following directory structure while following the instructions in this README file:

```
boot-exit/
  |___ gem5/                                   # gem5 source code
  |
  |___ disk-image/
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
  |___ linux-configs                           # Folder with Linux kernel configuration files
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

Next,

```sh
cd disk-image
# if packer (tool to build the disk) is not already installed
wget https://releases.hashicorp.com/packer/1.6.0/packer_1.6.0_linux_amd64.zip
unzip packer_1.6.0_linux_amd64.zip

# validate the packer script
./packer validate boot-exit/boot-exit.json
# build the disk image
./packer build boot-exit/boot-exit.json
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
A disk image already created following the above instructions can be found, gzipped, [here](http://dist.gem5.org/dist/v20-1/images/x86/ubuntu-18-04/boot-exit.img.gz).


## gem5 Run Scripts

gem5 scripts which configure the system and run simulation are available in configs-boot-tests/.
The main script `run_exit.py` expects following arguments:

**kernel:** path to the Linux kernel.

**disk:** path to the disk image.

**cpu_type:** cpu model (`kvm`, `atomic`, `simple`, `o3`).

**mem_sys:** memory system (`classic`, `MI_example`, `MESI_Two_Level`, `MOESI_CMP_directory`).

**num_cpus:** number of cpu cores.

**boot_type:** Linux kernel boot type (`init`, `systemd`).

An example use of this script is the following:

```sh
gem5/build/X86/gem5.opt configs/run_exit.py [path to the Linux kernel] [path to the disk image] kvm classic 4 init
```

## Linux Kernels

Boot tests use following five LTS (long term support) releases of the Linux kernel:

- 4.4.186
- 4.9.186
- 4.14.134
- 4.19.83
- 5.4.49

To compile the Linux binaries, follow these instructions (assuming that you are in `src/boot-exit/` directory):

```sh
git clone https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git
cd linux
# replace version with any of the above listed version numbers
git checkout v[version]
# copy the appropriate Linux kernel configuration file from linux-configs/
cp ../linux-configs/config.[version] .config
make -j8;
```

The compiled Linux binary will be named as `vmlinux`.

**Note:** The above instructions are tested with `gcc 7.5.0` and the compiled Linux binaries can be downloaded from the following links:

- [vmlinux-4.4.186](http://dist.gem5.org/dist/v20-1/kernels/x86/static/vmlinux-4.4.186)
- [vmlinux-4.9.186](http://dist.gem5.org/dist/v20-1/kernels/x86/static/vmlinux-4.9.186)
- [vmlinux-4.14.134](http://dist.gem5.org/dist/v20-1/kernels/x86/static/vmlinux-4.14.134)
- [vmlinux-4.19.83](http://dist.gem5.org/dist/v20-1/kernels/x86/static/vmlinux-4.19.83)
- [vmlinux-5.4.49](http://dist.gem5.org/dist/v20-1/kernels/x86/static/vmlinux-5.4.49)

## Working Status

Working status of these tests for gem5-20 can be found [here](https://www.gem5.org/documentation/benchmark_status/gem5-20).
