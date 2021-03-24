# PARSEC

This document includes instructions on how to create an Ubuntu 18.04 disk-image with PARSEC benchmark installed. The disk-image will be compatible with the gem5 simulator.

This is how the `src/parsec-tests/` directory will look like if all the artifacts are created correctly.

```
parsec/
  |___ gem5/                                   # gem5 folder
  |
  |___ disk-image/
  |      |___ shared/
  |      |___ parsec/
  |             |___ parsec-image/
  |             |      |___ parsec             # the disk image will be here
  |             |___ parsec.json               # the Packer script
  |             |___ parsec-install.sh         # the script to install PARSEC
  |             |___ post-installation.sh      # the script to install m5
  |             |___ runscript.sh              # script to run each workload
  |             |___ parsec-benchmark          # the parsec benchmark suite
  |
  |___ configs
  |      |___ system                           # system config directory
  |      |___ run_parsec.py                    # gem5 run script
  |
  |___ configs-mesi-two-level
  |      |___ system                           # system config directory
  |      |___ run_parsec_mesi_two_level.py     # gem5 run script
  |
  |___ README.md
```

Notice that there are two sets of system configuration directories and run scripts. For further detail on the config files look [here](#gem5-run-scripts).

## Building the disk image

In order to build the disk-image for PARSEC tests with gem5, build the m5 utility in `src/parsec-tests/` using the following:

```sh
git clone https://gem5.googlesource.com/public/gem5
cd gem5/util/m5
scons build/x86/out/m5
```

We use packer to create our disk-image. The instructions on how to install packer is shown below:

```sh
cd disk-image
wget https://releases.hashicorp.com/packer/1.6.0/packer_1.6.0_linux_amd64.zip
unzip packer_1.6.0_linux_amd64.zip
```

In order to build the disk-image first the script needs to be validated. Run the following command to validate `disk-image/parsec/parsec.json`.

```sh
./packer validate parsec/parsec.json
```

After the script has been successfuly validated you can create the disk-image by runnning:

```sh
./packer build parsec/parsec.json
```

You can find the disk-image in `parsec/parsec-image/parsec`.

## gem5 run scripts

There are two sets of run scripts and system configuration files in the directory. The scripts found in `configs` use the classic memory system while the scripts in `configs-mesi-two-level` use the ruby memory system with MESI_Two_Level cache coherency protocol. The parameters used in the both sets of experiments are explained below:

* **kernel**: The path to the linux kernel. We have verified capatibility with kernel version 4.19.83 which you can download at <http://dist.gem5.org/dist/v20-1/kernels/x86/static/vmlinux-4.19.83>. More information on building kernels for gem5 can be around in `src/linux-kernel`.
* **disk**: The path to the PARSEC disk-image.
* **cpu**: The type of cpu to use. There are two supported options: `kvm` (KvmCPU) and `timing` (TimingSimpleCPU).
* **benchmark**: The PARSEC workload to run. They include `blackscholes`, `bodytrack`, `canneal`, `dedup`, `facesim`, `ferret`, `fluidanimate`, `freqmine`, `raytrace`, `streamcluster`, `swaptions`, `vips`, `x264`. For more information on the workloads can be found at <https://parsec.cs.princeton.edu/>.
* **size**: The size of the chosen workload. Valid sizes are `simsmall`, `simmedium`, and `simlarge`.
* **num_cpus**: The number of cpus to simulate. When using `configs`, the only valid option is `1`. When using `configs-mesi-two-level` the number of supported cpus is show in the table below:


| CPU Model       | Core Counts |
|-----------------|-------------|
| KvmCPU          | 1,2,8       |
| TimingSimpleCPU | 1,2         |

Below are the examples of running an experiment with the two configurations.

```sh
<gem5 X86 binary> configs/run_parsec.py <kernel> <disk> <cpu> <benchmark> <size> <num_cpus>
<gem5 X86_MESI_Two_Level binary> configs-mesi-two-level/run_parsec.py <kernel> <disk> <cpu> <benchmark> <size> <num_cpus>
```

## Working Status

The working status of PARSEC runs for gem5-20 has been documented [here](https://www.gem5.org/documentation/benchmark_status/gem5-20#parsec-tests).
