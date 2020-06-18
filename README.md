# gem5 Resources

This repository contains the sources needed to compile the gem5 resources.
The compiled resources are found in the gem5 resources bucket,
http://dist.gem5.org/dist. Though these resources are not needed to compile or
run gem5, they may be required to execute some gem5 tests or may be useful
when carrying out specific simulations.

The structure of this repository is as follows:

* **Makefile** : Used to compile the resources. Please consult the sections
below for dependencies that may be required to compile these resources.
* **src** : The resource sources.
* **output** : Where the resources are stored after running the **Makefile**.
The directory structure maps to that under http://dist.gem5.org/dist. E.g.,
`output/test-progs/riscv-tests/median.riscv` can be found at
http://dist.gem5.org/dist/develop/test-progs/riscv-tests/median.riscv.

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
https://gem5.googlesource.com/public/gem5/+/refs/heads/master/CONTRIBUTING.md#releases).
Any release procedures related to the gem5 source can be assumed to be
applicable to gem5-resources.

The compiled resources for gem5 can be found under
http://dist.gem5.org/dist/{VERSION}. E.g. compiled resources for gem5 v20.2.0.0
are under http://dist.gem5.org/dist/v20-2-0-2 and are compiled from
gem5-resources v20.2.0.0. http://dist.gem5.org/dist/develop is kept in sync
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
[mailto:bbruce@ucdavis.edu](bbruce@ucdavis.edu) to have the compiled sources
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

```
make riscv-tests
```

The output of this compilation can be found at
`output/test-progs/riscv-tests/`


# Resource: Insttests

## Compilation

```
make insttests
```

The output of this compilation can be found in
`output/test-progs/insttest/bin/riscv/linux/`

# Resource: PThreads

## Compilation

```
make pthreads
```

This will compile the pthread binaries for the RISCV, ARM-32, ARM-64, and X86
ISAs. Specific ISA compilations can be obtained via the following commands:

```
make pthreads-aarch32
make pthreads-aarch64
make pthreads-riscv64
make pthreads-x86
```

The output of these compilations can be found in
`output/test-progs/pthreads`

# Licensing

Each project under the `src` is under a different license. Before using
any compiled binary, or modifying any source, please consult the corresponding
project's license.

* **riscv-tests** : `src/riscv-tests/LICENSE`.
* **insttests** : Consult individual copyright notices of source files in
`src/insttests`.
* **pthreads**: Consult individual copyright notices of source files in
`src/pthreads`.
