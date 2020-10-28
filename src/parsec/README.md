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

* kernel: The path to the linux kernel used to run the experiments with. In these experiments we only used kernel version 4.19.83 (You can download the binary [here](http://dist.gem5.org/dist/v20-1/kernels/x86/static/vmlinux-4.19.83)).
* disk: The path to the PARSEC disk-image (The disk-image created above will work for both set of experiments).
* cpu: The type of cpu that is used to run the simulation with. There are two possible options: kvm (KvmCPU) and timing (TimingSimpleCPU).
* benchmark: The workload among 13 workloads of PARSEC. They include `blackscholes`, `bodytrack`, `canneal`, `dedup`, `facesim`, `ferret`, `fluidanimate`, `freqmine`, `raytrace`, `streamcluster`, `swaptions`, `vips`, `x264`. For further information on the workloads read [here](https://parsec.cs.princeton.edu/).
* size: The size of chosen workload. In our experiments we used only three different sizes. For the experiments with kvm size `simsmall`, `simlarge`, and `native` are used. For the experiments with timing only size `simsmall` has been used.
* num_cpus: Number of cpus used to run the simulation with. For experiments with classic memory (located in `configs`) only valid option is `1`. However for the experiments with ruby memory (located in `configs-mesi-two-level`) the number of cores differ based on what cpu model is used the below table shows what core counts have been used with each cpu model.

| CPU Model       | Core Counts |
|-----------------|-------------|
| KvmCPU          | 1,2,8       |
| TimingSimpleCPU | 1,2         |

Below are the examples of running an experiment with the two configurations.

```sh
gem5/build/X86/gem5.opt configs/run_parsec.py linux-stable/vmlinux-4.19.83 disk-image/parsec/parsec-image/parsec timing bodytrack simsmall 1

gem5/build/X86_MESI_Two_Level/gem5.opt configs-mesi-two-level/run_parsec_mesi_two_level.py linux-stable/vmlinux-4.19.83 disk-image/parsec/parsec-image/parsec timing raytrace simsmall 2
```

## Working Status

The working status of PARSEC runs for gem5-20 has been documented [here](https://www.gem5.org/documentation/benchmark_status/gem5-20#parsec-tests).
