# gem5 Resources

This repository contains the sources needed to compile the gem5 resources.
Outlined in the following sections are build instructions for each of our
gem5 resources.

* **Makefile** : Used to compile the resources. Please consult the sections
below for dependencies that may be required to compile these resources.
* **src** : The resource sources.
* **dist** : Where the resources are stored after running the **Makefile**.
The directory structure maps to that under http://dist.gem5.org/dist. E.g.,
`dist/current/test-progs/riscv-tests/median.riscv` can be found at
http://dist.gem5.org/dist/current/test-progs/riscv-tests/median.riscv.

# Resource: RISCV Tests

Origin: <https://github.com/riscv/riscv-tests.git>

Revision: 19bfdab48c2a6da4a2c67d5779757da7b073811d

Local: `src/riscv-tests`

## Dependencies

The RISCV Tests requires the following dependencies:

### RISC-V GNU Compiler Toolchain

#### Prerequisites

```
sudo apt-get install autoconf automake autotools-dev curl python3 libmpc-dev \
libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo gperf libtool \
patchutils bc zlib1g-dev libexpat-dev
```

#### Installation

```
git clone --recursive https://github.com/riscv/riscv-gnu-toolchain
cd riscv-gnu-toolchain
./configure --prefix=/opt/riscv
sudo make
```

**Ensure `/opt/riscv/bin` is added to the PATH environment variable**.


## Compilation

```
make riscv-tests
```
The output of this compilation can be found at
`dist/current/test-progs/riscv-tests/`

# Resource: Insttests

## Dependencies

The Insttests require the follwing dependencies:

### RISCV-V GNU Compiler Toolchain (with multilib)

#### Installation

```
git clone --recursive https://github.com/riscv/riscv-gnu-toolchain
cd riscv-gnu-toolchain
./configure --prefix=/opt/riscv --enable-multilib
sudo make linux
```

**Ensure `/opt/riscv/bin` is appended to the PATH environment variable**.

## Compilation

```
make insttests
```

The output of this compilation can be found in
`dist/current/test-progs/insttest/bin/riscv/linux/`

# Licensing

Each project under the `src` is under a different license. Before using
any compiled binary, or modifying any source, please consult the corresponding
project's license.

* **riscv-tests** : `src/riscv-tests/LICENSE`.
* **insttests** : Consult individual copyright notices of source files in
`src/insttests`.
