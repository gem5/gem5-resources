# gem5 Resources

This repository contains the sources needed to compile the gem5 resources.
The compiled resources are found in the gem5 resources bucket,
http://dist.gem5.org/dist. Though these resources are not needed to compile or
run gem5, they may be required to execute some gem5 tests or may be useful
when carrying out specific simulations.

The structure of this repository is as follows:

* **src** : The resource sources.

The following sections outline our versioning policy, how to make changes
to this repository, and describe each resource and how they may be built.

# Versioning

We ensure that for each version of the [gem5 source](
https://gem5.googlesource.com/public/gem5/) there is a corresponding version of
the gem5-resources, with the assumption that version X of the gem5 source will
be used with version X of the gem5-resources. The gem5-resources repository
contains two branches, develop and master. The master branch's HEAD points
towards the latest gem5 resources release, which will be the same version id
as the that of the latest gem5 source. E.g., if the latest release of gem5 is
v20.2.0.0, then the latest release of gem5-resources will be v20.2.0.0, with
the HEAD of its master branch tagged as v20.2.0.0. Previous versions will be
tagged within the master branch. Past versions gem5-resources can thereby be
checked out with `git checkout <VERSION>`. A complete list of versions can be
found with `git tag`. The develop branch contains code under development and
will be merged into the master branch, then tagged, as part of the next release
of gem5. More information on gem5 release procedures can be found [here](
https://gem5.googlesource.com/public/gem5/+/refs/heads/stable/CONTRIBUTING.md#releases).
Any release procedures related to the gem5 source can be assumed to be
applicable to gem5-resources.

The compiled resources for gem5 can be found under
http://dist.gem5.org/dist/{VERSION}. E.g. compiled resources for gem5 v20.2.0.2
are under http://dist.gem5.org/dist/v20-1-2-0-2 and are compiled from
gem5-resources v20.2.0.2. http://dist.gem5.org/dist/develop is kept in sync
with the develop branch, and therefore should not be depended upon for stable,
regular usage.

**Note: Resource files for gem5 v19.0.0.0, our legacy release, can be found
under http://dist.gem5.org/dist/current**.

# Making Changes

Changes to this repository are made to the develop branch via our Gerrit
code review system. Therefore, to make changes, first clone the repository
checkout the develop branch:

```
git clone https://gem5.googlesource.com/public/gem5-resources
git checkout --track origin/develop
```

Then make changes and commit. When ready, push to Gerrit with:

```
git push origin HEAD:refs/for/develop
```

The change will then be reviewed via our [Gerrit code review system](
https://gem5-review.googlesource.com). Once fully accepted and merged into
the gem5-resources repository, please contact Bobby R. Bruce
[bbruce@ucdavis.edu](mailto:bbruce@ucdavis.edu) to have the compiled sources
uploaded to the gem5 resources bucket.

# Requirements

These requirements, their prerequisites, and installation instructions have
been written with the assumption that they shall be installed on an x86 Ubuntu
18.04 system. Installation instructions may differ across other systems.

## RISC-V GNU Compiler Toolchain

The RISC-V GNU Compiler Toolchain is needed to cross-compile to the RISCV-V
ISA infrastructure.

### Prerequisites

```
sudo apt-get install autoconf automake autotools-dev curl python3 libmpc-dev \
libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo gperf libtool \
patchutils bc zlib1g-dev libexpat-dev
```

### Installation

```
git clone --recursive https://github.com/riscv/riscv-gnu-toolchain
cd riscv-gnu-toolchain
./configure --prefix=/opt/riscv --enable-multilib
sudo make linux
```

**Ensure `/opt/riscv/bin` is added to the PATH environment variable**.

## GNU ARM-32 bit Toolchain

The GNU ARM-32 bit toolchain is required to cross compile to the ARM-32 bit
ISA.

### Installation

The toolchain may be installed via the apt-get package manager:

```
sudo apt-get install g++-arm-linux-gnueabihf
```

## GNU ARM-64 bit Toolchain

The GNU ARM-64 bit toolchain is required to cross compile to the ARM-64 bit
ISA.

### Installation

The toolchain may be installved via the apt-get package manager:

```
sudo apt-get install g++-aarch64-linux-gnu
```

# Resource: RISCV Tests

Origin: <https://github.com/riscv/riscv-tests.git>

Revision: 19bfdab48c2a6da4a2c67d5779757da7b073811d

Local: `src/riscv-tests`

## Compilation

To compile the RISCV Tests the [RISCV GNU Compiler](
#risc_v-gnu-compiler-toolchain) must be installed.

Then, to compile:

```
cd src/riscv-tests
autoconf
./configure --prefix=/opt/riscv/target
make -C src/riscv-tests
```

This RISCV binaries can then be found within the `src/riscv-tests/benchmarks`
directory.

## Pre-built binary

<http://dist.gem5.org/dist/v20-1/test-progs/riscv-tests/dhrystone.riscv>

<http://dist.gem5.org/dist/v20-1/test-progs/riscv-tests/median.riscv>

<http://dist.gem5.org/dist/v20-1/test-progs/riscv-tests/mm.riscv>

<http://dist.gem5.org/dist/v20-1/test-progs/riscv-tests/mt-matmul.riscv>

<http://dist.gem5.org/dist/v20-1/test-progs/riscv-tests/mt-vvadd.riscv>

<http://dist.gem5.org/dist/v20-1/test-progs/riscv-tests/multiply.riscv>

<http://dist.gem5.org/dist/v20-1/test-progs/riscv-tests/pmp.riscv>

<http://dist.gem5.org/dist/v20-1/test-progs/riscv-tests/qsort.riscv>

<http://dist.gem5.org/dist/v20-1/test-progs/riscv-tests/rsort.riscv>

<http://dist.gem5.org/dist/v20-1/test-progs/riscv-tests/spmv.riscv>

<http://dist.gem5.org/dist/v20-1/test-progs/riscv-tests/towers.riscv>

<http://dist.gem5.org/dist/v20-1/test-progs/riscv-tests/vvadd.riscv>

# Resource: Insttests

The Insttest sources can be found in the `src/insttest` directory.

## Compilation

To compile the Insttests, the [RISCV GNU Compiler](
#risc_v-gnu-compiler-toolchain) must be installed.

To compile:

```
make -C src/insttest
```

The Insttest binaries can then be found within the `src/insttest/bin`
directory.

## Prebuilt binaries

<http://dist.gem5.org/dist/v20-1/test-progs/insttest/bin/riscv/linux/insttest-rv64a>

<http://dist.gem5.org/dist/v20-1/test-progs/insttest/bin/riscv/linux/insttest-rv64c>

<http://dist.gem5.org/dist/v20-1/test-progs/insttest/bin/riscv/linux/insttest-rv64d>

<http://dist.gem5.org/dist/v20-1/test-progs/insttest/bin/riscv/linux/insttest-rv64f>

<http://dist.gem5.org/dist/v20-1/test-progs/insttest/bin/riscv/linux/insttest-rv64i>

<http://dist.gem5.org/dist/v20-1/test-progs/insttest/bin/riscv/linux/insttest-rv64m>


# Resource: simple

Simple single source file per executable userland or baremetal examples.

The toplevel executables under `src/simple` can be built for any ISA that we
have a cross compiler for as mentioned under [Requirements](#requirements).

Examples that build only for some ISAs specific ones are present under
`src/simple/<ISA>` subdirs, e.g. `src/simple/aarch64/`,

The ISA names are meant to match `uname -m`, e.g.:

- `aarch64`
- `arm`
- `riscv`
- `x86_64`
- `sparc64`

You have to specify the path to the gem5 source code with `GEM5_ROOT` variable so that
m5ops can be used from there. For example for a native build:

    cd src/simple
    make -j`nproc` GEM5_ROOT=../../../

The default of that variable is such that if you place this repository and the gem5
repository in the same directory:

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

Note that a some C source files can produce both a baremetal and an userland.
For example `m5_exit.c` produces both:

    out/aarch64/bare/m5_exit.out
    out/aarch64/user/m5_exit.out

However, since the regular userland toolchain is used rather than a more
specialized baremetal toolchain, the C standard library is not available.
Therefore, only very few C examples can build for baremetal, notably the ones
that use m5ops.

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

## Pre-build binaries

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/x86/test_pthread_create_seq>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/x86/test_pthread_create_para>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/x86/test_pthread_mutex>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/x86/test_atomic>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/x86/test_pthread_cond>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/x86/test_std_thread>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/x86/test_std_mutex>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/x86/test_std_condition_variable>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/aarch32/test_pthread_create_seq>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/aarch32/test_pthread_create_para>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/aarch32/test_pthread_mutex>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/aarch32/test_atomic>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/aarch32/test_pthread_cond>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/aarch32/test_std_thread>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/aarch32/test_std_mutex>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/aarch32/test_std_condition_variable>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/aarch64/test_pthread_create_seq>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/aarch64/test_pthread_create_para>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/aarch64/test_pthread_mutex>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/aarch64/test_atomic>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/aarch64/test_pthread_cond>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/aarch64/test_std_thread>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/aarch64/test_std_mutex>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/aarch64/test_std_condition_variable>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/riscv64/test_pthread_create_seq>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/riscv64/test_pthread_create_para>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/riscv64/test_pthread_mutex>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/riscv64/test_atomic>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/riscv64/test_pthread_cond>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/riscv64/test_std_thread>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/riscv64/test_std_mutex>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/riscv64/test_std_condition_variable>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/sparc64/test_pthread_create_seq>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/sparc64/test_pthread_create_para>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/sparc64/test_pthread_mutex>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/sparc64/test_atomic>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/sparc64/test_pthread_cond>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/sparc64/test_std_thread>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/sparc64/test_std_mutex>

<http://dist.gem5.org/dist/v20-1/test-progs/pthreads/sparc64/test_std_condition_variable>


# Resource: Square

## Compilation

To compile:

```
cd src/square
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu make gfx8-apu
```

The compiled binary can be found in `src/square/bin`

## Pre-built binary

<http://dist.gem5.org/dist/v20-1/test-progs/square/square.o>

# Resource: SPEC 2006

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

# Resource: SPEC 2017

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

# Resource: GAP Benchmark Suite (GAPBS) tests

[GAPBS](http://gap.cs.berkeley.edu/benchmark.html) is a graph processing benchmark suite and it contains 6 kernels: Breadth-First Search, PageRank, Connected Components, Betweenness Centrality, Single-Source Shortest Paths, and Triangle Counting.

`src/gapbs` contains resources to build a GAPBS disk image which may
be used to run the benchmark on gem5 X86 simulations.
`src/gapbs/README.md` contains build and usage instructions.

The pre-built GAPBS disk image can be found here:
<http://dist.gem5.org/dist/v20-1/images/x86/ubuntu-18-04/gapbs.img.gz>.

# Resource: PARSEC Benchmark Suite

The [Princeton Application Repository for Shared-Memory Computers (PARSEC)](
https://parsec.cs.princeton.edu/) is a benchmark suite composed of
multithreaded programs. In `src/parsec` we provide the source to build a disk
image which may be used, alongside configuration files, to run the PARSEC
Benchmark Suite on gem5 architectural simulations. Please consult
`src/parsec/README.md` for build and execution information.

A pre-built parsec benchmark image, for X86, can be found here:
<http://dist.gem5.org/dist/v20-1/images/x86/ubuntu-18-04/parsec.img.gz>.

# Resource: NAS Parallel Benchmarks (NPB) Tests

The [NAS Parallel Benchmarks] (NPB) are a small set of programs designed to
help evaluate the performance of parallel supercomputers. The set consists of
five Lunux Kernels and three pseudo-applications. gem5 resources provides a
disk image, and scripts allowing for the NPB image to be run within gem5 X86
simulations. A pre-built npb disk image can be downloaded here:
<http://dist.gem5.org/dist/v20-1/images/x86/ubuntu-18-04/npb.img.gz>.

The npb resources can be found in `src/npb`. It consists of:
- npb disk image resources
- gem5 run scripts to execute these tests

The instructions to build the npb disk image, a Linux kernel binary, and how to use gem5 run scripts to run npb are available in the [README](src/npb-tests/README.md) file.

# Resource: Linux Boot Tests

The Linux boot tests refer to the tests performed with different gem5 configurations to check its ability to boot a Linux kernel.
More information on Linux boot tests can be found [here](https://www.gem5.org/project/2020/03/09/boot-tests.html).

The boot-tests resources consist of three main components:
- boot-tests disk image
- gem5 run scripts to execute boot tests
- linux kernel configuration files

The instructions to build the boot-tests disk image (`boot-exit`), the Linux binaries, and how to use gem5 run scripts to run boot-tests are available in this [README](src/boot-tests/README.md) file.

# Resource: Insttest

The Insttests test SPARC instructions.

Creating the SPARC Insttest binary requires a SPARC cross compile. Instructions
on creating a cross compiler can be found [here](
https://preshing.com/20141119/how-to-build-a-gcc-cross-compiler).

## Compilation

To compile:

```
cd src/insttest
make
```

We provide a docker image with a pre-loaded SPARC cross compiler. To use:

```
cd src/insttest
docker run --volume $(pwd):$(pwd) -w $(pwd) --rm gcr.io/gem5-test/sparc64-gnu-cross:latest make
```

The compiled binary can be found in `src/insttest/bin`.

## Prebuild Binary

<http://dist.gem5.org/dist/v20-1/test-progs/insttest/bin/sparc/linux/insttest>

# Resource: Linux Kernel Binary

Contains scripts to create a Linux kernel binary.
Instructions on how to use the scripts can be found here `src/linux-kernel/README.md`.

# Licensing

Each project under the `src` is under a different license. Before using
any compiled binary, or modifying any source, please consult the corresponding
project's license.

* **riscv-tests** : `src/riscv-tests/LICENSE`.
* **pthreads**: Consult individual copyright notices of source files in
`src/pthreads`.
* **square**: Consult individual copyright notices of source files in
`src/square`.
* **spec 2006**: SPEC CPU 2006 requires purchase of benchmark suite from
[SPEC](https://www.spec.org/cpu2006/) thus, it cannot be freely distributed.
Consult individual copyright notices of source files in `src/spec-2006`.
* **spec 2017**: SPEC CPU 2017 requires purchase of benchmark suite from
[SPEC](https://www.spec.org/cpu2017/) thus, it cannot be freely distributed.
Consult individual copyright notices of source files in `src/spec-2017`.
* **gapbs**: Consult individual copyright notices of source files in
`src/gapbs`.
* **parsec**: The PARSEC license can be found at
`src/parsec/disk-image/parsec/parsec-benchmark/LICENSE`. This is a 3-Clause
BSD License (A Princeton University copyright). For the remaining files, please
consult copyright notices in the source files.
* **npb-tests**: Consult individual copyright notices of source files in
`src/npb`. The NAS Parallel Benchmarks utilize a permissive BSD-style license.
* **boot-tests**: Consult individual copyright notices of source files in
`src/boot-tests`.
* **insttest**: Consult individual copyright notices of source files in
`src/insttest`.
* **linux-kernel**: Consult individual copyright notices of source files in
`src/linux-kernel`.
