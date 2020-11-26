# SPEC 2006
This document aims to provide instructions to create a gem5-compatible disk
image containing the SPEC 2006 benchmark suite and also to provide necessary
configuration files.

## Building the Disk Image
Creating a disk-image for SPEC 2006 requires the benchmark suite ISO file.
More info about SPEC 2006 can be found [here](https://www.spec.org/cpu2006/).

In this tutorial, we assume that the file `CPU2006v1.0.1.iso` contains the SPEC
benchmark suite, and we provide the scripts that are made specifically for
SPEC 2006 version 1.0.1.
Throughout the this document, the root folder is `src/spec-2006/`.
All commands should be run from the assumed root folder.

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
git checkout origin/develop
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

The path to the disk image is `spec-2006/spec-2006-image/spec-2006`.
Please refer to [this tutorial](https://gem5art.readthedocs.io/en/latest/tutorials/spec2006-tutorial.html#preparing-scripts-to-modify-the-disk-image)
for more information about the scripts used in this document.

## Linux Kernel
The following link contains the compiled Linux kernel that was tested by
running gem5-20 with SPEC 2006,
- [vmlinux-4.19.83](http://dist.gem5.org/dist/v20-1/kernels/x86/static/vmlinux-4.19.83)

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
kernel.

`disk`: required, a positional argument specifying the path to the disk image
containing SPEC 2006 benchmark suite.

`cpu`: required, a positional argument specifying the name of either a
detailed CPU model or KVM CPU model.

The available CPU models are,

| cpu    | Corresponding CPU model in gem5 |
| ------ | ------------------------------- |
| kvm    |                                 |
| o3     | DerivO3CPU                      |
| atomic | AtomicSimpleCPU                 |
| timing | TimingSimpleCPU                 |

`benchmark`: required, a positional argument specifying the name of the
[SPEC 2006 workload](https://gem5art.readthedocs.io/en/latest/tutorials/spec2006-tutorial.html#appendix-i-working-spec-2006-benchmarks-x-cpu-model-table) to run.

`size`: required, a positional argument specifying the input data size,
must be one of {test, train, ref}.

Assume the compiled Linux kernel is available in the assumed root folder, the
following is an example of running a SPEC 2006 workload in full system mode,
`
gem5/build/X86/gem5.opt --outdir [path to the gem5 output directory] configs/run_spec.py -z vmlinux-4.19.83 disk-image/spec-2006/spec-2006-image/spec-2006 atomic 403.gcc test
`

## Working Status
Status of these benchmarks runs with respect to gem5-20, linux kernel version
4.19.83 and gcc version 7.5.0 can be found
[here](https://www.gem5.org/documentation/benchmark_status/gem5-20#spec-2006-tests)
