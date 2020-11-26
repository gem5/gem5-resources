# GAP Benchmark Suite (GAPBS) tests
This document provides instructions to create a GAP Benchmark Suite (GAPBS) disk image, which, along with provided configuration scripts, may be used to run GAPBS within gem5 simulations.

A pre-build disk image, for X86, can be found, gzipped, here: <http://dist.gem5.org/dist/v20-1/images/x86/ubuntu-18-04/gapbs.img.gz>.

## Building the Disk Image
Assuming that you are in the `src/gapbs/` directory, first create `m5` (which is needed to create the disk image):

```sh
git clone https://gem5.googlesource.com/public/gem5
cd gem5/util/m5
scons build/x86/out/m5
```

To create the disk image you need to add the packer binary in the disk-image directory:

```sh
cd disk-image/
wget https://releases.hashicorp.com/packer/1.6.0/packer_1.6.0_linux_amd64.zip   # (if packer is not already installed)
unzip packer_1.6.0_linux_amd64.zip # (if packer is not already installed)
./packer validate gapbs/gapbs.json
./packer build gapbs/gapbs.json
```

After this process succeeds, the disk image can be found on the `src/gapbs/disk-image/gapbs-image/gapbs`.

## gem5 Configuration Scripts

gem5 scripts which configure the system and run the simulation are available in `configs/`.
The main script `run_gapbs.py` expects following arguments:

**--kernel** : path to the Linux kernel. GAPBS has been tested with [vmlinux-5.2.3](http://dist.gem5.org/dist/v20-1/kernels/x86/static/vmlinux-5.2.3).

**--disk** : Path to the disk image.

**--cpu\_type** : Cpu model (`kvm`, `atomic`, `simple`, `o3`).

**--num\_cpus** : Number of cpu cores.

**--mem\_sys** : Memory model (`classic`, `MI_example`, `MESI_Two_Level`).

**--benchmark** : The graph workload (`cc`, `bc`, `bfs`, `tc`, `pr`, `sssp`).

**--synthetic** : Type of graph (if synthetic graph 1, if real world graph 0).

**--graph** : Size of graph (if synthetic then number of nodes, else name of the graph).

Example usage:

```sh
gem5/build/X86/gem5.opt configs/run_gapbs.py [path to the linux kernel] [path to the disk image] kvm 1 classic cc 1 20
```
