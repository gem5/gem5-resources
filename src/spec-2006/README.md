# SPEC 2006
This document aims to provide instructions to create a gem5-compatible disk
image containing the SPEC 2006 benchmark suite and also to provide necessary
configuration files.

## Building the Disk Image
Creating a disk-image for SPEC 2006 requires the benchmark suite ISO file.
More info about SPEC 2006 can be found <https://www.spec.org/cpu2006/>.

In this tutorial, we assume that the file `CPU2006v1.0.1.iso` contains the SPEC
benchmark suite, and we provide the scripts that are made specifically for
SPEC 2006 version 1.0.1.
Throughout the this document, the root folder is `src/spec-2006/`.
All commands should be run from this root folder.

The layout of the folder after the scripts are run is as follows,

```
spec-2006/
  |___ gem5/                                   # gem5 folder
  |
  |___ disk-image/
  |      |___ shared/
  |      |___ spec-2006/
  |             |___ spec-2006-image/
  |             |      |___ spec-2006          # the disk image will be generated here
  |             |___ spec-2006.json            # the Packer script
  |             |___ CPU2006v1.0.1.iso         # SPEC 2006 ISO (add here)
  |
  |___ configs
  |      |___ system
  |      |___ run_spec.py                      # gem5 config file
  |
  |___ vmlinux-4.19.83                         # download link below
  |
  |___ README.md
```

First, to build `m5` (required for interactions between gem5 and the guest):

```sh
git clone https://gem5.googlesource.com/public/gem5
cd gem5
cd util/m5
scons build/x86/out/m5
```

We use [Packer](https://www.packer.io/), an open-source automated disk image
creation tool, to build the disk image.
In the root folder,

```sh
cd disk-image
wget https://releases.hashicorp.com/packer/1.6.0/packer_1.6.0_linux_amd64.zip #(download the packer binary)
unzip packer_1.6.0_linux_amd64.zip
./packer validate spec-2006/spec-2006.json #validate the Packer script
./packer build spec-2006/spec-2006.json
```

## gem5 Configuration Scripts
gem5 scripts which configure the system and run the simulation are available
in `configs/`.
The main script `run_spec.py` expects following arguments:

`usage: run_spec.py [-h] [-l] [-z] kernel disk cpu mem_sys benchmark size`

`-h`: show this help message and exit.

`-l`, `--no-copy-logs`: optional, to not copy SPEC run logs to the host system,
logs are copied by default, and are available in the result folder.

`-z`, `--allow-listeners`: optional, to turn on GDB listening ports, the ports
are off by default.

`kernel`: required, a positional argument specifying the path to the Linux
kernel. We have tested using version 4.19.83, which can be downloaded from
<http://dist.gem5.org/dist/v20-1/kernels/x86/static/vmlinux-4.19.83>. Info on
building Linux kernels for gem5 can be found in `src/linux-kernel`

`disk`: required, a positional argument specifying the path to the disk image
containing SPEC 2006 benchmark suite.

`cpu`: required, a positional argument specifying the name of either a
detailed CPU model or KVM CPU model.

The available CPU models are,

| cpu      | Corresponding CPU model in gem5 |
| ---------| ------------------------------- |
| `kvm`    |                                 |
| `o3`     | DerivO3CPU                      |
| `atomic` | AtomicSimpleCPU                 |
| `timing` | TimingSimpleCPU                 |

`mem_sys`: required, a positional argument specifying the memory system.
The available memory systems are,

| mem\_sys              | Notes                  |
| --------------------- | ---------------------- |
| `classic`             | classic memory system  |
| `MI_example`          | Ruby memory system     |
| `MESI_Two_Level`      | Ruby memory system     |
| `MOESI_CMP_directory` | Ruby memory system     |

`benchmark`: required, a positional argument specifying the name of the SPEC
2006 workload to run. The available benchmarks are,

* 401.bzip2
* 403.gcc
* 410.bwaves
* 416.gamess
* 429.mcf
* 433.milc
* 434.zeusmp
* 435.gromacs
* 436.cactusADM
* 437.leslie3d
* 444.namd
* 445.gobmk
* 453.povray
* 454.calculix
* 456.hmmer
* 458.sjeng
* 459.GemsFDTD
* 462.libquantum
* 464.h264ref
* 465.tonto
* 470.lbm
* 471.omnetpp
* 473.astar
* 481.wrf
* 482.sphinx3
* 998.specrand
* 999.specrand

`size`: required, a positional argument specifying the input data size. Valid
values are `test`, `train`, and `ref`.

As a minimum the following parameters must be specified:

```
<gem5 X86 binary> --outdir <output directory> configs/run_spec.py <kernel> <disk> <cpu> <mem_sys> <benchmark> <size>
```

**Note**: `--outdir` is a required argument when running the gem5 binary with SPEC 2006.
The path to the output directory must be an absoblute path.

## Working Status
Status of these benchmarks runs with respect to gem5-20, linux kernel version
4.19.83 and gcc version 7.5.0 can be found
[here](https://www.gem5.org/documentation/benchmark_status/gem5-20#spec-2006-tests).
