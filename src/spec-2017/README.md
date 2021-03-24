# SPEC 2017
This document aims to provide instructions to create a gem5-compatible disk
image containing the SPEC 2017 benchmark suite and also to provide necessary
configuration files.

## Building the Disk Image
Creating a disk-image for SPEC 2017 requires the benchmark suite ISO file.
More info about SPEC 2017 can be found at <https://www.spec.org/cpu2017/>.

In this tutorial, we assume that the file `cpu2017-1.1.0.iso` contains the SPEC
benchmark suite, and we provide the scripts that are made specifically for
SPEC 2017 version 1.1.0.
Throughout the this document, the root folder is `src/spec-2017/`.
All commands should be run from the assumed root folder.

The layout of the folder after the scripts are run is as follows,

```
spec-2017/
  |___ gem5/                                   # gem5 folder
  |
  |___ disk-image/
  |      |___ shared/
  |      |___ spec-2017/
  |             |___ spec-2017-image/
  |             |      |___ spec-2017          # the disk image will be generated here
  |             |___ spec-2017.json            # the Packer script
  |             |___ cpu2017-1.1.0.iso         # SPEC 2017 ISO (add here)
  |
  |___ configs
  |      |___ system/
  |      |___ run_spec.py                      # gem5 run script
  |
  |___ vmlinux-4.19.83                         # Linux kernel, link to download provided below
  |
  |___ README.md

```

First, to build `m5` (required for interactions between gem5 and the system under simuations):

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
wget https://releases.hashicorp.com/packer/1.6.0/packer_1.6.0_linux_amd64.zip # download the packer binary
unzip packer_1.6.0_linux_amd64.zip
./packer validate spec-2017/spec-2017.json # validate the Packer script
./packer build spec-2017/spec-2017.json
```

## gem5 Configuration Scripts
gem5 scripts which configure the system and run the simulation are available
in `configs/`.
The main script `run_spec.py` expects following arguments:

`usage: run_spec.py [-h] [-l] [-z] kernel disk cpu benchmark size`

`-h`: show this help message and exit.

`-l`, `--no-copy-logs`: optional, to not copy SPEC run logs to the host system,
logs are copied by default, and are available in the result folder.

`-z`, `--allow-listeners`: optional, to turn on GDB listening ports, the ports
are off by default.

`kernel`: required, a positional argument specifying the path to the Linux
kernel. This has been tested with version 4.19.83, available at
<http://dist.gem5.org/dist/v20-1/kernels/x86/static/vmlinux-4.19.83>. Info on
building Linux kernels can be found in `src/linux-kernels`.

`disk`: required, a positional argument specifying the path to the disk image
containing SPEC 2017 benchmark suite.

`cpu`: required, a positional argument specifying the name of either a
detailed CPU model or KVM CPU model.

The available CPU models are,

| cpu    | Corresponding CPU model in gem5 |
| ------ | ------------------------------- |
| kvm    |                                 |
| o3     | DerivO3CPU                      |
| atomic | AtomicSimpleCPU                 |
| timing | TimingSimpleCPU                 |

`benchmark`: required, a positional argument specifying the name of the SPEC
2017 to run. Listed below are valid options:

* 500.perlbench_r
* 502.gcc_r
* 503.bwaves_r
* 505.mcf_r
* 507.cactuBSSN_r
* 508.namd_r
* 510.parest_r
* 511.povray_r
* 519.lbm_r
* 520.omnetpp_r
* 521.wrf_r
* 523.xalancbmk_r
* 525.x264_r
* 526.blender_r
* 527.cam4_r
* 531.deepsjeng_r
* 538.imagick_r
* 541.leela_r
* 544.nab_r
* 548.exchange2_r
* 549.fotonik3d_r
* 554.roms_r
* 557.xz_r
* 600.perlbench_s
* 602.gcc_s
* 603.bwaves_s
* 605.mcf_s
* 607.cactuBSSN_s
* 619.lbm_s
* 620.omnetpp_s
* 621.wrf_s
* 623.xalancbmk_s
* 625.x264_s
* 627.cam4_s
* 628.pop2_s
* 631.deepsjeng_s
* 638.imagick_s
* 641.leela_s
* 644.nab_s
* 648.exchange2_s
* 649.fotonik3d_s
* 654.roms_s
* 657.xz_s
* 996.specrand_fs
* 997.specrand_fr
* 998.specrand_is
* 999.specrand_ir

`size`: required, a positional argument specifying the input data size. Valid
values are `test`, `train`, and `ref`.

As a minimum the following parameters must be specified:

```
<gem5 X86 binary> --outdir <output directory> configs/run_spec.py <kernel> <disk> <cpu> <mem_sys> <benchmark> <size>
```

**Note**: `--outdir` is a required argument when running the gem5 binary with SPEC 2006.


## Working Status
Status of these benchmarks runs with respect to gem5-20, linux kernel version
4.19.83 and gcc version 7.5.0 can be found
[here](https://www.gem5.org/documentation/benchmark_status/gem5-20#spec-2017-tests)

