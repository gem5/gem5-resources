---
layout: default
---

# gem5 Resources

This repository contains the sources needed to compile the gem5 resources.
The compiled resources are found in the gem5 resources bucket,
http://dist.gem5.org/dist. Though these resources are not needed to compile or
run gem5, they may be required to execute some gem5 tests or may be useful
when carrying out specific simulations.

The following sections outline our versioning policy, how to make changes
to this repository, and describe each resource and how they may be built.

## Versioning

We ensure that for each version of the [gem5 source](
https://gem5.googlesource.com/public/gem5/) there is a corresponding version of
the gem5-resources, with the assumption that version X of the gem5 source will
be used with version X of the gem5-resources. The gem5-resources repository
contains two branches, develop and stable. The stable branch's HEAD points
towards the latest gem5 resources release, which will be the same version id
as the that of the latest gem5 source. E.g., if the latest release of gem5 is
v20.2.0.0, then the latest release of gem5-resources will be v20.2.0.0, with
the HEAD of its stable branch tagged as v20.2.0.0. Previous versions will be
tagged within the stable branch. Past versions gem5-resources can thereby be
checked out with `git checkout <VERSION>`. A complete list of versions can be
found with `git tag`. The develop branch contains code under development and
will be merged into the stable branch, then tagged, as part of the next release
of gem5. More information on gem5 release procedures can be found [here](
https://gem5.googlesource.com/public/gem5/+/refs/heads/stable/CONTRIBUTING.md#releases).
Any release procedures related to the gem5 source can be assumed to be
applicable to gem5-resources.

The compiled resources for gem5 can be found under
http://dist.gem5.org/dist/{VERSION}. E.g. compiled resources for gem5 v20.2
are under http://dist.gem5.org/dist/v20-2 and are compiled from
gem5-resources v20.2. http://dist.gem5.org/dist/develop is kept in sync
with the develop branch, and therefore should not be depended upon for stable,
regular usage.

**Note: Resource files for gem5 v19.0.0.0, our legacy release, can be found
under http://dist.gem5.org/dist/current**.

## Submitting a contribution

We utilize GitHub to review changes made to the gem5-resources. To make changes, 
follow the steps below.

1. Fork the gem5 repository on GitHub from https://github.com/gem5/gem5-resources/.
2. Create a new branch in your forked repository for your changes.
3. Commit your changes to the new branch.
4. Push the branch to your forked repository.
5. Open a pull request from your branch in your forked repository to the main gem5 
resources repository.


If you have not signed up for an account on the github
(https://github.com/), you first have to create an account.

 1. Go to https://github.com/
 2. Click "Sign up" in the upper right corner.

Changes are required to have a `Change-ID`, which can be added using the 
pre-commit hook. This can be installed via the following:

``` bash
pip install pre-commit
pre-commit install
```

### Stable vs. Develop branch

The rule for when to work on the stable vs. develop branch is as follows:

* If the change applies to the current gem5 stable, then the change should be 
on the stable branch of gem5-resources.

* If the change cannot work on gem5 stable and requires updates to gem5 that 
are only found on gem5 develop, then the change should be on the develop branch 
of gem5-resources.

When a new version of gem5 is released, the develop branch is merged into the 
stable branch. When gem5-resources's stable and develop branches diverge, we 
merge stable into develop.

### Code Review

Once a change has been submitted to GitHub, you may view the change at
<https://github.com/gem5/gem5-resources/pulls>.

Through the GitHub pull request we strongly advise you add reviewers to your 
change. GitHub will automatically notify those you assign. We recommend you add 
both **Bobby R. Bruce <bbruce@ucdavis.edu>** (@BobbyRBruce) and 
**Jason Lowe-Power <jason@lowepower.com>** (@powerjg) as reviewers.

Reviewers will review the change. For non-trivial edits, it is not unusual for 
a change to receive feedback from reviewers that they want incorporated before 
flagging as acceptable for merging into the gem5-resources repository. 
**All communications between reviewers and contributors should be done in a 
polite manner. Rude and/or dismissive remarks will not be tolerated.**

Once your change has been accepted by reviewers a maintainer will squash and 
merge your pull request into the gem5-resources repository. 

## Resource: RISCV Tests

The RISCV Tests source can be found in the `src/riscv-tests` directory. More
information about these tests can be found in `src/riscv-tests/README.md`.

### RISCV Tests Origins

The RISCV Tests in this repository were obtained from
<https://github.com/riscv-software-src/riscv-tests.git>, revision
e65ecdf941a5484af27f9be223fb655ebcb0398b.

### RISCV Tests Compilation

To compile the RISCV Tests the [RISCV GNU Compiler](
https://github.com/riscv/riscv-gnu-toolchain) must be installed.

Then, to compile:

```
cd src/riscv-tests
autoconf
./configure --prefix=/opt/riscv/target
RISCV_PREFIX=<COMPILER_PREFIX> make
```
As an example for `make`, if the binary name for the RISCV compiler is
`riscv64-linux-gnu-gcc`, then the make command is the following:
```
RISCV_PREFIX=riscv64-linux-gnu- make
```
This RISCV binaries can then be found within the `src/riscv-tests/benchmarks`
directory.

### RISCV Tests Pre-built binaries

<http://dist.gem5.org/dist/v22-0/test-progs/riscv-tests/dhrystone.riscv>

<http://dist.gem5.org/dist/v22-0/test-progs/riscv-tests/median.riscv>

<http://dist.gem5.org/dist/v22-0/test-progs/riscv-tests/mm.riscv>

<http://dist.gem5.org/dist/v22-0/test-progs/riscv-tests/mt-matmul.riscv>

<http://dist.gem5.org/dist/v22-0/test-progs/riscv-tests/mt-vvadd.riscv>

<http://dist.gem5.org/dist/v22-0/test-progs/riscv-tests/multiply.riscv>

<http://dist.gem5.org/dist/v22-0/test-progs/riscv-tests/pmp.riscv>

<http://dist.gem5.org/dist/v22-0/test-progs/riscv-tests/qsort.riscv>

<http://dist.gem5.org/dist/v22-0/test-progs/riscv-tests/rsort.riscv>

<http://dist.gem5.org/dist/v22-0/test-progs/riscv-tests/spmv.riscv>

<http://dist.gem5.org/dist/v22-0/test-progs/riscv-tests/towers.riscv>

<http://dist.gem5.org/dist/v22-0/test-progs/riscv-tests/vvadd.riscv>

## Resource: simple

The simple resources are small binaries, often used to run quick tests and
checks in gem5. They are baremetal.

### simple Compilation

Simple single source file per executable userland or baremetal examples.

The toplevel executables under `src/simple` can be built for any ISA that we
have a cross compiler for. The current cross compilers supported are :

- `x86_64` (as installed via APT with `sudo apt install build-essential`)
- [`aarch64-linux-gnu-gcc/arch64-linux-gnu-g++`](
https://preshing.com/20141119/how-to-build-a-gcc-cross-compiler/)
- [`arm-linux-gnueabihf-gcc/arm-linux-gnueabihf-g++`](
https://preshing.com/20141119/how-to-build-a-gcc-cross-compiler/)
- [`riscv64-linux-gnu-gcc/riscv64-linux-gnu-g++`](
https://preshing.com/20141119/how-to-build-a-gcc-cross-compiler/)

Examples that build only for some ISAs specific ones are present under
`src/simple/<ISA>` subdirs, e.g. `src/simple/aarch64/`,

The ISA names are meant to match `uname -m`, e.g.:

- `aarch64`
- `arm`
- `riscv`
- `x86_64`
- `sparc64`

You have to specify the path to the gem5 source code with `GEM5_ROOT` variable 
so that `m5ops` can be used from there. For example for a native build:

    cd src/simple
    make -j`nproc` GEM5_ROOT=../../../

The default of that variable is such that if you place this repository and the 
gem5 repository in the same directory:

    ./gem5/
    ./gem5-resources/

you can omit that variable and build just with:

    make

After the building, the generated files are located under:

    ./out/<ISA>/

For example, some of the userland executables built on x86 are:

    ./out/x86_64/user/hello.out
    ./out/x86_64/user/x86_64/mwait.out

Or if you build for a different ISA:

    make ISA=aarch64

some of the executables would be:

    ./out/aarch64/user/hello.out
    ./out/aarch64/user/aarch64/futex_ldxr_stxr.out

By default, only userland executables are built. You can build just the baremetal
ones instead with:

    make ISA=aarch64 bare

or both userland and baremetal with:

    make ISA=aarch64 all

A sample baremetal executable generated by this is:

    out/aarch64/bare/m5_exit.out

Only ISAs that have a corresponding `src/simple/bootloader/` file can build for
baremetal, e.g. `src/simple/bootloader/aarch64.S`.

Note that some C source files can produce both a baremetal and an userland.
For example `m5_exit.c` produces both:

    out/aarch64/bare/m5_exit.out
    out/aarch64/user/m5_exit.out

However, since the regular userland toolchain is used rather than a more
specialized baremetal toolchain, the C standard library is not available.
Therefore, only very few C examples can build for baremetal, notably the ones
that use `m5ops`.

There are also examples that can only build for baremetal, e.g.
`aarch64/semihost_exit` only builds for baremetal, as semihosting is not
available on userland.

The `simple` directory is also able to generate squashfs images containing
only a single userland executable at `/sbin/init` for any of the userland
executables. This can be done with a command of type:

    make ISA=aarch64 out/aarch64/squashfs/m5_exit.squashfs

Squashfs is a filesystem type that the Linux kernel understands natively,
exactly like ext4, except that it is a bit more convenient to create, and
write-only.

You can therefore give those squashfs images to gem5 exactly as you
would give a normal ext4 raw image, by pointing to it for example with
`fs.py --disk-image=m5_exit.squashfs` as shown at:
https://www.gem5.org/documentation/general_docs/fullsystem/building_arm_kernel
Linux will then run the given userland executable after Linux boots as the
init program.

The initial motivation for this was to generate simple test images for
Linux boot.

Since this is a less common use case, squashfs images are not currently
generated by any single phony target all at once.

### simple Pre-built binaries

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/x86/test_pthread_create_seq>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/x86/test_pthread_create_para>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/x86/test_pthread_mutex>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/x86/test_atomic>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/x86/test_pthread_cond>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/x86/test_std_thread>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/x86/test_std_mutex>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/x86/test_std_condition_variable>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/aarch32/test_pthread_create_seq>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/aarch32/test_pthread_create_para>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/aarch32/test_pthread_mutex>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/aarch32/test_atomic>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/aarch32/test_pthread_cond>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/aarch32/test_std_thread>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/aarch32/test_std_mutex>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/aarch32/test_std_condition_variable>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/aarch64/test_pthread_create_seq>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/aarch64/test_pthread_create_para>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/aarch64/test_pthread_mutex>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/aarch64/test_atomic>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/aarch64/test_pthread_cond>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/aarch64/test_std_thread>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/aarch64/test_std_mutex>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/aarch64/test_std_condition_variable>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/riscv64/test_pthread_create_seq>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/riscv64/test_pthread_create_para>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/riscv64/test_pthread_mutex>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/riscv64/test_atomic>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/riscv64/test_pthread_cond>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/riscv64/test_std_thread>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/riscv64/test_std_mutex>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/riscv64/test_std_condition_variable>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/sparc64/test_pthread_create_seq>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/sparc64/test_pthread_create_para>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/sparc64/test_pthread_mutex>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/sparc64/test_atomic>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/sparc64/test_pthread_cond>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/sparc64/test_std_thread>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/sparc64/test_std_mutex>

<http://dist.gem5.org/dist/v22-0/test-progs/pthreads/sparc64/test_std_condition_variable>

## Resource: Square

### Square Compilation

To compile:

**Note**: Make sure you are in gem5-resources directory (resources like square 
are not present in the gem5 repository). To clone the gem5-resources repository, 
run the following command:

```
git clone https://github.com/gem5/gem5-resources.git
```


```
cd src/gpu/square
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu make gfx8-apu
```

The compiled binary can be found in `src/gpu/square/bin`

### Square Pre-built binary

<http://dist.gem5.org/dist/v22-0/test-progs/square/square>

# Resource: HSA Agent Packet Example

Based off of the Square resource in this repository, this resource serves as
an example for using an HSA Agent Packet to send commands to the GPU command
processor included in the GCN_X86 build of gem5.

The example command extracts the kernel's completion signal from the domain
of the command processor and the GPU's dispatcher. Initially this was a
workaround for the hipDeviceSynchronize bug, now fixed. The method of
waiting on a signal can be applied to other agent packet commands though.

Custom commands can be added to the command processor in gem5 to control
the GPU in novel ways.

## Compilation

To compile:

```
cd src/gpu/hsa-agent-pkt
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu make gfx8-apu
```

The compiled binary can be found in `src/gpu/hsa-agent-pkt/bin`

# Resource: HIP Sample Applications

The [HIP sample apps](
https://github.com/ROCm-Developer-Tools/HIP/tree/roc-1.6.0/samples) contain
applications that introduce various GPU programming concepts that are usable
in HIP.

The samples cover topics such as using and accessing different parts of GPU
memory, running multiple GPU streams, and optimization techniques for GPU code.

Certain apps aren't included due to complexities with either ROCm or Docker
(hipEvent, profiler), or due to lack of feature support in gem5 (peer2peer)

## Compilation

```
cd src/gpu/hip-samples
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu make
```

Individual programs can be made by specifying the name of the program

By default, this code builds for gfx801, a GCN3-based APU. This can be
overridden by specifying `-e HCC_AMDGPU_TARGET=<target>` in the build command.

## Pre-built binary

<http://dist.gem5.org/dist/v22-0/test-progs/hip-samples/2dshfl>

<http://dist.gem5.org/dist/v22-0/test-progs/hip-samples/dynamic_shared>

<http://dist.gem5.org/dist/v22-0/test-progs/hip-samples/inline_asm>

<http://dist.gem5.org/dist/v22-0/test-progs/hip-samples/MatrixTranspose>

<http://dist.gem5.org/dist/v22-0/test-progs/hip-samples/sharedMemory>

<http://dist.gem5.org/dist/v22-0/test-progs/hip-samples/shfl>

<http://dist.gem5.org/dist/v22-0/test-progs/hip-samples/stream>

<http://dist.gem5.org/dist/v22-0/test-progs/hip-samples/unroll>

# Resource: Heterosync

[Heterosync](https://github.com/mattsinc/heterosync) is a benchmark suite used
to test the performance of various types of fine-grained synchronization on
tightly-coupled GPUs. The version in gem5-resources contains only the HIP code.

The README in the heterosync folder details the various synchronization primitives
and the other command-line arguments for use with heterosync.

## Compilation
```
cd src/gpu/heterosync
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu make release-gfx8
```

The release-gfx8 target builds for gfx801, a GCN3-based APU, and gfx803, a
GCN3-based dGPU. There are other targets (release) that build for GPU types
that are currently unsupported in gem5.

## Pre-built binary

<http://dist.gem5.org/dist/v22-0/test-progs/heterosync/gcn3/allSyncPrims-1kernel>

# Resource: lulesh

[lulesh](https://computing.llnl.gov/projects/co-design/lulesh) is a DOE proxy
application that is used as an example of hydrodynamics modeling. The version
provided is for use with the gpu-compute model of gem5.

## Compilation and Running
```
cd src/gpu/lulesh
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu make
```

By default, the Makefile builds for gfx801, and is placed in the 
`src/gpu/lulesh/bin` folder.

lulesh is a GPU application, which requires that gem5 is built with the GCN3_X86 
architecture. To build GCN3_X86:

```
# Working directory is your gem5 directory
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu scons -sQ -j$(nproc) build/GCN3_X86/gem5.opt
```

The following command shows how to run lulesh

Note: lulesh has two optional command-line arguments, to specify the stop time 
and number of iterations. To set the arguments, add 
`--options="<stop_time> <num_iters>` to the run command. The default arguments 
are equivalent to `--options="1.0e-2 10"`.


```
# Assuming gem5 and gem5-resources are in your working directory
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu gem5/build/GCN3_X86/gem5.opt gem5/configs/example/apu_se.py -n3 --mem-size=8GB --benchmark-root=gem5-resources/src/gpu/lulesh/bin -clulesh
```

## Pre-built binary

<http://dist.gem5.org/dist/v22-0/test-progs/lulesh/lulesh>

# Resource: halo-finder (HACC)

[HACC](https://asc.llnl.gov/coral-2-benchmarks) is a DoE application designed 
to simulate the evolution of the universe by simulating the formation of 
structure in collisionless fluids under the influence of gravity. The halo-finder 
code can be GPU accelerated by using the code in RCBForceTree.cxx.

`src/gpu/halo-finder/src` contains the code required to build and run 
ForceTreeTest from `src/halo_finder` in the main HACC codebase.
`src/gpu/halo-finder/src/dfft` contains the dfft code from `src/dfft` in the 
main HACC codebase.

## Compilation and Running

halo-finder requires that certain libraries that aren't installed by default in 
the GCN3 docker container provided by gem5, and that the environment is 
configured properly in order to build. We provide a Dockerfile that installs 
those libraries and sets the environment.

In order to test the GPU code in halo-finder, we compile and run ForceTreeTest.

To build the Docker image and the benchmark:
```
cd src/gpu/halo-finder
docker build -t <image_name> .
docker run --rm -v ${PWD}:${PWD} -w ${PWD}/src -u $UID:$GID <image_name> make hip/ForceTreeTest
```

The binary is built for gfx801 by default and is placed at 
`src/gpu/halo-finder/src/hip/ForceTreeTest`

ForceTreeTest is a GPU application, which requires that gem5 is built with the 
GCN3_X86 architecture.
To build GCN3_X86:
```
# Working directory is your gem5 directory
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID <image_name> scons -sQ -j$(nproc) build/GCN3_X86/gem5.opt
```

To run ForceTreeTest:
```
# Assuming gem5 and gem5-resources are in the working directory
docker run --rm -v $PWD:$PWD -w $PWD -u $UID:$GID <image_name> gem5/build/GCN3_X86/gem5.opt gem5/configs/example/apu_se.py -n3 --benchmark-root=gem5-resources/src/gpu/halo-finder/src/hip -cForceTreeTest --options="0.5 0.1 64 0.1 1 N 12 rcb"
```

## Pre-built binary

<http://dist.gem5.org/dist/v22-0/test-progs/halo-finder/ForceTreeTest>

# Resource: DNNMark

[DNNMark](https://github.com/shidong-ai/DNNMark) is a benchmark framework used
to characterize the performance of deep neural network (DNN) primitive workloads.

## Compilation and Running

To build DNNMark:
**NOTE**: Due to DNNMark building a library, it's important to mount gem5-resources
to the same directory within the docker container when building and running, as 
otherwise the benchmarks won't be able to link against the library. The example 
commands do this by using `-v ${PWD}:${PWD}` in the docker run commands

```
cd src/gpu/DNNMark
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu ./setup.sh HIP
docker run --rm -v ${PWD}:${PWD} -w ${PWD}/build -u $UID:$GID ghcr.io/gem5/gcn-gpu make
```

DNNMark uses MIOpen kernels, which are unable to be compiled on-the-fly in gem5.
We have provided a python script to generate these kernels for a subset of the
benchmarks for a gfx801 GPU with 4 CUs by default

To generate the MIOpen kernels:
```
cd src/gpu/DNNMark
docker run --rm -v ${PWD}:${PWD} -v${PWD}/cachefiles:/root/.cache/miopen/2.9.0 -w ${PWD} ghcr.io/gem5/gcn-gpu python3 generate_cachefiles.py cachefiles.csv [--gfx-version={gfx801,gfx803}] [--num-cus=N]
```

Due to the large amounts of memory that need to be set up for DNNMark, we have
added in the ability to MMAP a file to reduce setup time, as well as added a
program that can generate a 2GB file of floats.

To make the MMAP file:
```
cd src/gpu/DNNMark
g++ -std=c++0x generate_rand_data.cpp -o generate_rand_data
./generate_rand_data
```

DNNMark is a GPU application, which requires that gem5 is built with the 
GCN3_X86 architecture.
To build GCN3_X86:
```
# Working directory is your gem5 directory
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu scons -sQ -j$(nproc) build/GCN3_X86/gem5.opt
```

To run one of the benchmarks (fwd softmax) in gem5:
```
# Assuming gem5 and gem5-resources are sub-directories of the current directory
docker run --rm -v ${PWD}:${PWD} -v ${PWD}/gem5-resources/src/gpu/DNNMark/cachefiles:/root/.cache/miopen/2.9.0 -w ${PWD} ghcr.io/gem5/gcn-gpu gem5/build/GCN3_X86/gem5.opt gem5/configs/example/apu_se.py -n3 --benchmark-root=gem5-resources/src/gpu/DNNMark/build/benchmarks/test_fwd_softmax -cdnnmark_test_fwd_softmax --options="-config gem5-resources/src/gpu/DNNMark/config_example/softmax_config.dnnmark -mmap gem5-resources/src/gpu/DNNMark/mmap.bin"
```


# Resource: pennant

pennant is an unstructured mesh physics mini-app designed for advanced
architecture research.  It contains mesh data structures and a few
physics algorithms adapted from the LANL rad-hydro code FLAG, and gives
a sample of the typical memory access patterns of FLAG.

## Compiling and Running

```
cd src/gpu/pennant
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu make
```

By default, the binary is built for gfx801 and is placed in `src/gpu/pennant/build`.

pennant is a GPU application, which requires that gem5 is built with the 
GCN3_X86 architecture.

pennant has sample input files located at `src/gpu/pennant/test`. The following 
command shows how to run the sample `noh`:

```
# Assuming gem5 and gem5-resources are in your working directory
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu gem5/build/GCN3_X86/gem5.opt gem5/configs/example/apu_se.py -n3 --benchmark-root=gem5-resources/src/gpu/pennant/build -cpennant --options="gem5-resources/src/gpu/pennant/test/noh/noh.pnt"
```

The output gets placed in `src/gpu/pennant/test/noh/`, and the file `noh.xy`
against the `noh.xy.std` file. 

Note: Only some tests have `.xy.std` files to
compare against, and there may be slight differences due to floating-point 
rounding.

## Pre-built binary

<http://dist.gem5.org/dist/v22-0/test-progs/pennant/pennant>

## Resource: SPEC 2006

The [Standard Performance Evaluation Corporation](
https://www.spec.org/benchmarks.html) (SPEC) CPU 2006 benchmarks are designed
to provide performance measurements that can be used to compare
compute-intensive workloads on different computer systems. SPEC CPU 2006
contains 12 different benchmark tests.

`src/spec-2006` provides resources on creating a SPEC 2006 disk image, and
necessary scripts to run the SPEC 2006 benchmarks within X86 gem5 simulations.
Please consult the `src/spec-2006/README.md` for more information.

**Please note, due to licensing issues, the SPEC 2006 iso cannot be provided
as part of this repository.**

## Resource: SPEC 2017

The [Standard Performance Evaluation Corporation](
https://www.spec.org/benchmarks.html) (SPEC) CPU 2017 benchmarks are designed
to provide performance measurements that can be used to compare
compute-intensive workloads on different computer systems. SPEC CPU 2017
contains 43 benchmarks organized into four suites: SPECspeed 2017 Integer,
SPECspeed 2017 Floating Point, SPECrate 2017 Integer, and SPECrate 2017
Floating Point.

`src/spec-2017` provides resources on creating a SPEC 2017 disk image, and
necessary scripts to run the SPEC 2017 benchmarks within X86 gem5 simulations.
Please consult the `src/spec-2017/README.md` for more information.

**Please note, due to licensing issues, the SPEC 2017 iso cannot be provided
as part of this repository.**

## Resource: GAP Benchmark Suite (GAPBS) tests

[GAPBS](http://gap.cs.berkeley.edu/benchmark.html) is a graph processing 
benchmark suite and it contains 6 kernels: Breadth-First Search, PageRank, 
Connected Components, Betweenness Centrality, Single-Source Shortest Paths, 
and Triangle Counting.

### GAPBS Origin

We obtained the GAPBS benchmark suite from 
<http://gap.cs.berkeley.edu/benchmark.html>

### Building the GAPBS image

`src/gapbs` contains resources to build a GAPBS disk image which may be used to 
run the benchmark on gem5 X86 simulations.
`src/gapbs/README.md` contains build and usage instructions.

### GAPBS Pre-built disk image

<http://dist.gem5.org/dist/v22-0/images/x86/ubuntu-18-04/gapbs.img.gz>

## Resource: PARSEC Benchmark Suite

The [Princeton Application Repository for Shared-Memory Computers (PARSEC)](
https://parsec.cs.princeton.edu/) is a benchmark suite composed of
multithreaded programs.

### PARSEC Origins

We used PARSEC 3.0, available from <https://parsec.cs.princeton.edu>.

### Building the PARSEC image

In `src/parsec` we provide the source to build a disk
image which may be used, alongside configuration files, to run the PARSEC
Benchmark Suite on gem5 architectural simulations. Please consult
`src/parsec/README.md` for build and execution information.

### GAPBS Pre-built disk image

<http://dist.gem5.org/dist/v22-0/images/x86/ubuntu-18-04/parsec.img.gz>.

## Resource: NAS Parallel Benchmarks (NPB) Tests

The NAS Parallel Benchmarks (NPB) are a small set of programs designed to
help evaluate the performance of parallel supercomputers. The set consists of
five Linux Kernels and three pseudo-applications. gem5 resources provides a
disk image, and scripts allowing for the NPB image to be run within gem5 X86
simulations.

### NPB Origins

We use NPB 3.4.1, available from
<https://www.nas.nasa.gov/publications/npb.html>.

### NPB Building

The npb resources can be found in `src/npb`. It consists of:
- npb disk image resources
- gem5 run scripts to execute these tests

The instructions to build the npb disk image, a Linux kernel binary, and how to
use gem5 run scripts to run npb are available in the [README](
src/npb-tests/README.md) file.

### NPB Pre-built disk image

<http://dist.gem5.org/dist/v22-0/images/x86/ubuntu-18-04/npb.img.gz>


## Resource: Linux Boot Tests

The Linux boot tests refer to the tests performed with different gem5 
configurations to check its ability to boot a Linux kernel.
More information on Linux boot tests can be found 
[here](https://www.gem5.org/project/2020/03/09/boot-tests.html).

The boot-tests resources consist of three main components:
- x86-ubuntu disk image
- gem5 run scripts to execute boot tests
- linux kernel configuration files

The instructions to build the x86-ubuntu disk image, the Linux binaries, and 
how to use gem5 run scripts to run boot-tests are available in this 
[README](src/x86-ubuntu/README.md) file.

## Resource: RISCV Full System

The RISCV Full System resource includes a RISCV boot loader 
(`berkeley bootloader (bbl)`) to boot the Linux 5.10 kernel on a RISCV system, 
and an image which includes the BusyBox software suite.
The resource also contains simple gem5 run/config scripts to run Linux full 
system simulations in which a user may telnet into.

Further information on building a riscv disk image, a riscv boot loader, and 
how to use gem5 scripts to run riscv Linux full system simulations, is 
available in the [README](src/riscv-fs/README.md) file.

### RISCV Full System pre-built disk image

<http://dist.gem5.org/dist/v22-0/images/riscv/busybox/riscv-disk.img.gz>

### RISCV Full System pre-built Linux bootloader

<http://dist.gem5.org/dist/v22-0/kernels/riscv/static/bootloader-vmlinux-5.10>


## Resource: RISCV Full System with Disk Image

The RISCV Full System resource includes a RISCV bootloader 
(`berkeley bootloader (bbl)`) to boot the Linux 5.10 kernel on a RISCV system.
The workload and the Linux utils (provided by BusyBox) are also included in 
the bootloader.
The resource also contains simple gem5 run/config scripts to run Linux full 
system simulations in which a user may telnet into.

More details on building such a RISCV bootloader and hwo does it work are 
available in the [README.md](src/riscv-boot-exit-nodisk/README.md) file.

### RISCV Full System pre-built Linux bootloader with embedded workload

<http://dist.gem5.org/dist/v22-0/misc/riscv/bbl-busybox-boot-exit>


## Resource: Insttest


The Insttests test SPARC instructions.

Creating the SPARC Insttest binary requires a SPARC cross compile. Instructions
on creating a cross compiler can be found [here](
https://preshing.com/20141119/how-to-build-a-gcc-cross-compiler).

### Insttest Compilation

To compile:

```
cd src/insttest
make
```

We provide a docker image with a pre-loaded SPARC cross compiler. To use:

```
cd src/insttest
docker run --volume $(pwd):$(pwd) -w $(pwd) --rm ghcr.io/gem5/sparc64-gnu-cross:latest make
```

The compiled binary can be found in `src/insttest/bin`.

### Insttest Pre-built binary

<http://dist.gem5.org/dist/v22-0/test-progs/insttest/bin/sparc/linux/insttest>

## Resource: Linux Kernel Binary

Contains scripts to create a Linux kernel binary.

### Linux Kernel Compilation

Instructions on how to use the scripts can be found here
`src/linux-kernel/README.md`.

### Linux Kernel Pre-built binaries

<http://dist.gem5.org/dist/v22-0/kernels/x86/static/vmlinux-4.4.186>
<http://dist.gem5.org/dist/v22-0/kernels/x86/static/vmlinux-4.9.186>
<http://dist.gem5.org/dist/v22-0/kernels/x86/static/vmlinux-4.14.134>
<http://dist.gem5.org/dist/v22-0/kernels/x86/static/vmlinux-4.19.83>

## Resource: LupV Disk image and Kernel/boot loader

[gem5 supports LupIO](https://www.gem5.org/project/2022/02/07/lupio.html).
An example of using gem5 with LupIO can be found in [`configs/example/lupv`](https://gem5.googlesource.com/public/gem5/+/refs/tags/v22.0.0.0/configs/example/lupv/).

The sources to build a LupV (LupIO with RISC-V) disk image (based on busybox) 
and a LupV bootloader/kernel can be found in `src/lupv`.

### LupV Pre-built disk image

<http://dist.gem5.org/dist/v22-0/images/riscv/busybox/riscv-lupio-busybox.img.gz>

### LupV Pre-built bootloader/kernel

<http://dist.gem5.org/dist/v22-0/kernels/riscv/static/lupio-linux>

## Licensing

There is no universal license encompassing all this repository's contents.
The licences covering the individual gem5 resources are therefore highlighted
below.

* **asmtest** : [`src/asmtest/LICENSE`](
https://gem5.googlesource.com/public/gem5-resources/+/refs/heads/stable/src/asmtest/LICENSE).
* **riscv-tests** : [`src/riscv-tests/LICENSE`](
https://gem5.googlesource.com/public/gem5-resources/+/refs/heads/stable/src/riscv-tests/LICENSE).
* **square**: Consult individual copyright notices of source files in
`src/gpu/square`.
* **hsa-agent-pkt**: `src/gpu/hsa-agent-pkt/square.cpp` is licensed under the
same licence as 'src/gpu/square/square.cpp'.
`src/gpu/hsa-agent-pkt/HSA_Interface.[h|.cpp]` are licensed under a BSD Lisense
(A University of Maryland copyright).
* **hip-samples**: Consult individual copyright notices of the source file in
'src/gpu/hip-samples/src'
* **heterosync**: Consult `src/gpu/heterosync/LICENSE.txt`
* **lulesh**: Consult the copyright notice in `src/gpu/lulesh/src/gpu/lulesh.hip.cc`
* **halo-finder**: halo-finder is a subcomponent of HACC, which is licensed under
a BSD license.
* **DNNMark**: DNNMark is licensed under an MIT license, see `src/gpu/DNNMark/LICENSE`
* **pennant**: pennant is licensed under a BSD license, see `src/gpu/pennant/LICENSE`
[src/gpu/square](
https://gem5.googlesource.com/public/gem5-resources/+/refs/heads/stable/src/gpu/square).
* **spec 2006**: SPEC CPU 2006 requires purchase of benchmark suite from
[SPEC](https://www.spec.org/cpu2006/) thus, it cannot be freely distributed.
Consult individual copyright notices of source files in [`src/spec-2006`](
https://gem5.googlesource.com/public/gem5-resources/+/refs/heads/stable/src/spec-2006).
* **spec 2017**: SPEC CPU 2017 requires purchase of benchmark suite from
[SPEC](https://www.spec.org/cpu2017/) thus, it cannot be freely distributed.
Consult individual copyright notices of source files in [`src/spec-2017`](
https://gem5.googlesource.com/public/gem5-resources/+/refs/heads/stable/src/spec-2017).
* **gapbs**: Consult individual copyright notices of source files in
[`src/gapbs`](
https://gem5.googlesource.com/public/gem5-resources/+/refs/heads/stable/src/gapbs).
* **parsec**: The code of the [PARSEC project](
https://parsec.cs.princeton.edu/)
is covered by a 3-Clause BSD License (
[`src/parsec/disk-image/parsec/parsec-benchmark/LICENSE`](
https://gem5.googlesource.com/public/gem5-resources/+/refs/heads/stable/src/parsec/disk-image/parsec/parsec-benchmark/LICENSE)).
For the remaining files, please consult copyright notices in individual source
files.
* **npb-tests**: Consult individual copyright notices of source files in
[`src/npb`](
https://gem5.googlesource.com/public/gem5-resources/+/refs/heads/stable/src/npb).
The NAS Parallel Benchmarks utilize a permissive BSD-style license.
* **x86-ubuntu**: Consult individual copyright notices of source files in
[`src/x86-ubuntu`](
https://gem5.googlesource.com/public/gem5-resources/+/refs/heads/stable/src/x86-ubuntu).
* **insttest**: Consult individual copyright notices of source files in
[`src/insttest`](
https://gem5.googlesource.com/public/gem5-resources/+/refs/heads/stable/src/insttest).
* **linux-kernel**: Consult individual copyright notices of source files in
[`src/linux-kernel`](
https://gem5.googlesource.com/public/gem5-resources/+/refs/heads/stable/src/linux-kernel).
* **hack-back**: Consult individual copyright notices of source files in
[`src/hack-back`](
https://gem5.googlesource.com/public/gem5-resources/+/refs/heads/stable/src/hack-back).
* **simple**: Consult individual copyright notices of the source files in
[`src/simple`](
https://gem5.googlesource.com/public/gem5-resources/+/refs/heads/stable/src/simple).
