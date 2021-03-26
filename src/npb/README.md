# NAS Parallel Benchmarks (NPB) Tests

This document provides instructions to create a disk image needed to run the NPB tests with gem5 and points to the gem5 configuration files needed to run these tests.
The NAS parallel benchmarks ([NPB](https://www.nas.nasa.gov/)) are high performance computing (HPC) workloads consisting of different kernels and pseudo applications:

Kernels:
- **IS:** Integer Sort, random memory access
- **EP:** Embarrassingly Parallel
- **CG:** Conjugate Gradient, irregular memory access and communication
- **MG:** Multi-Grid on a sequence of meshes, long- and short-distance communication, memory intensive
- **FT:** discrete 3D fast Fourier Transform, all-to-all communication

Pseudo Applications:
- **BT:** Block Tri-diagonal solver
- **SP:** Scalar Penta-diagonal solver
- **LU:** Lower-Upper Gauss-Seidel solver

There are different classes (A,B,C,D,E and F) of each workload based on the input data size. Detailed discussion of the data sizes is available [here](https://www.nas.nasa.gov/publications/npb_problem_sizes.html).

We make use of a modified source of the NPB suite for these tests, which can be found in `disk-images/npb/npb-hooks`.
We have added ROI (region of interest) annotations for each benchmark which is used by gem5 to separate simulation statistics between different regions of each benchmark. gem5 magic instructions are used before and after each ROI to exit the guest and transfer control to gem5 the gem5 configuration script. This can then dump and reset stats, or switch to cpus of interest.

We assume the following directory structure while following the instructions in this README file:

```
npb/
  |___ gem5/                               # gem5 source code
  |
  |___ disk-image/
  |      |___ shared/                      # Auxiliary files needed for disk creation
  |      |___ npb/
  |            |___ npb-image/             # Will be created once the disk is generated
  |            |      |___ npb             # The generated disk image
  |            |___ npb.json               # The Packer script to build the disk image
  |            |___ runscript.sh           # Executes a user provided script in simulated guest
  |            |___ post-installation.sh   # Moves runscript.sh to guest's .bashrc
  |            |___ npb-install.sh         # Compiles NPB inside the generated disk image
  |            |___ npb-hooks              # The NPB source (modified to function better with gem5).
  |
  |___ configs
  |      |___ system                       # gem5 system config files
  |      |___ run_npb.py                   # gem5 run script to run NPB tests
  |
  |___ linux                               # Linux source and binary will live here
  |
  |___ README.md                           # This README file
```

## Disk Image

Assuming that you are in the `src/npb/` directory (the directory containing this README), first build `m5` (which is needed to create the disk image):

```sh
git clone https://gem5.googlesource.com/public/gem5
cd gem5/util/m5
scons build/x86/out/m5
```

Next,

```sh
cd disk-image
# if packer is not already installed
wget https://releases.hashicorp.com/packer/1.6.0/packer_1.6.0_linux_amd64.zip
unzip packer_1.6.0_linux_amd64.zip

# validate the packer script
./packer validate npb/npb.json
# build the disk image
./packer build npb/npb.json
```

Once this process succeeds, the created disk image can be found on `npb/npb-image/npb`.
A disk image already created following the above instructions can be found, gzipped, [here](http://dist.gem5.org/dist/v21-0/images/x86/ubuntu-18-04/npb.img.gz).


## gem5 Run Scripts

The gem5 scripts which configure the system and run simulation are available in configs-npb-tests/.
The main script `run_npb.py` expects following arguments:

**kernel:** path to the Linux kernel. This disk image has been tested with version 4.19.83, available at <http://dist.gem5.org/dist/v21-0/kernels/x86/static/vmlinux-4.19.83>. More info on building Linux Kernels can be found in the `src/linux-kernels` directory.

**disk:** path to the npb disk image.

**cpu:** CPU model (`kvm`, `atomic`, `timing`).

**mem_sys:** memory system (`classic`, `MI_example`, `MESI_Two_Level`, or `MOESI_CMP_directory`).

**benchmark:** NPB benchmark to execute (`bt.A.x`, `cg.A.x`, `ep.A.x`, `ft.A.x`, `is.A.x`, `lu.A.x`, `mg.A.x`,  `sp.A.x`).

**Note:**
We have only tested class `A` of the NPB suite, though `A`,`B`,`C` and `D` of NPB are available in the disk image
For example, for build class `F` of the `bt` benchmark `bt.F.x` can be specified (replacinv `A` with `F` from above).

**num_cpus:** number of CPU cores.

An example of how to use these scripts:

```sh
gem5/build/X86/gem5.opt configs/run_npb.py <kernel> <disk> <cpu> <mem_sys> <benchmark> <num_cpus>
```

## Working Status

The working status of these tests for gem5-20 can be found [here](https://www.gem5.org/documentation/benchmark_status/gem5-20#npb-tests).
