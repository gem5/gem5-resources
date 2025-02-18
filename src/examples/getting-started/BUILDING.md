# Building the Getting Started Suite

The Getting Started Suite is designed to provide a set of pre-compiled familiar workloads to help one get started with gem5.

This document will guide you through the process of building the binaries for the Getting Started Suite.
You may want to follow this guide if you want to customize these binaries, use a different compiler, or compile them to a new ISA.
Note that as described in the README.md file there are precompiled binaries available from the gem5 resources binary repository.```

## Table of Contents

- [Building the Suite Binaries](#building-the-suite-binaries)
  - [gem5-resources](#gem5-resources)
  - [GAPBS](#gapbs)
  - [LLVM Test Suite](#llvm-test-suite)
    - [Cross-compiling the LLVM test suite](#cross-compiling-the-llvm-test-suite)
  - [NAS Parallel Benchmarks](#nas-parallel-benchmarks)
    - [Using the Suite Binaries](#using-the-suite-binaries)

## Building the Suite Binaries

### gem5-resources

First, clone the gem5-resources repository:

```bash
git clone https://github.com/gem5/gem5-resources.git
```

Next, checkout the following commit:

```bash
cd gem5-resources
git checkout dc3e57f6de56496188b06404c7bd4dabee8fffef
```

Next, navigate to the `src/matrix-multiply` directory:

```bash
cd gem5-resources/src/matrix-multiply
```

Then, build the matrix multiply binary:

```bash
make
```

The x86 version of the matrix multiply binary was compiled using the `gcc` compiler native to the system.
The version of `gcc` used to compile the matrix multiply binary was `gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0`.

The ARM version of the matrix multiply binary was compiled using the `gcc` compiler native to the system.
The version of `gcc` used to compile the matrix multiply binary was `gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0`.

If you want to build the binary for a specific architecture, you need to cross-compile the binary.
For example, to build the binary for RISC-V, need to edit the `Makefile` and change the `matrix-multiply` command to `<RISC-V CROSS COMPILER> -o matrix-multiply matrix-multiply.c`.
An example of the `<RISC-V CROSS COMPILER>` is `riscv64-linux-gnu-gcc`, and was used to cross-compile the matrix multiply binary for the RISC-V architecture version of the Getting Started Suite.
The version of `riscv64-linux-gnu-gcc` used to compile the matrix multiply binary was `riscv64-linux-gnu-gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0`, and can be installed using the following command:

```bash
apt-get install gcc-riscv64-linux-gnu
```

To compile the binary, run the `make` command.

### GAPBS

First, clone the GAPBS repository:

```bash
git clone https://github.com/sbeamer/gapbs.git
```

Next, checkout the following commit:

```bash
cd gapbs
git checkout 33f73f46e78807ab4b04f6bca618ed7f02169d2f
```

Next, run make to build the BFS and TC binaries:

```bash
make
```

The x86 version of the matrix multiply binary was compiled using the `g++` compiler native to the system.
The version of `g++` used to compile the matrix multiply binary was `g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0`.

The ARM version of the matrix multiply binary was compiled using the `g++` compiler native to the system.
The version of `g++` used to compile the matrix multiply binary was `g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0`.

If you want to build the binaries for a specific architecture, you need to cross-compile the binaries.
For example, to build the binaries for RISC-V, you need to edit the `Makefile` and add a variable called `CXX` and set it to the RISC-V cross-compiler.
An example of the `CXX` variable is `CXX=riscv64-linux-gnu-g++`, and was used to cross-compile the GAPBS binaries for the RISC-V architecture version of the Getting Started Suite.
The version of `riscv64-linux-gnu-g++` used to compile the GAPBS binaries was `g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0`, and can be installed using the following command:

```bash
apt-get install g++-riscv64-linux-gnu
```

To compile the binary, run the `make` command.

### LLVM Test Suite

First, clone the LLVM test suite repository:

```bash
git clone https://github.com/llvm/llvm-test-suite.git
```

Next, checkout the following commit:

```bash
cd llvm-test-suite
git checkout 10275acc659f99c5fa292236876dee4dcf063d93
```

On a native system, the LLVM test suite can be compiled using the following:

```bash
cmake . -B build -G "Unix Makefiles" -DTEST_SUITE_COLLECT_CODE_SIZE=OFF -DCMAKE_EXE_LINKER_FLAGS="-static" -DBUILD_SHARED_LIBS=OFF -DCMAKE_FIND_LIBRARY_SUFFIXES=".a"
```

This command uses the `cmake` command to generate a build system for the LLVM test suite.
The `-DTEST_SUITE_COLLECT_CODE_SIZE=OFF` flag is used to disable the collection of code size information.
The `-DCMAKE_EXE_LINKER_FLAGS="-static"` flag is used to link the executables statically.
The `-DBUILD_SHARED_LIBS=OFF` flag is used to build the LLVM test suite without shared libraries.
The `-DCMAKE_FIND_LIBRARY_SUFFIXES=".a"` flag is used to search for static libraries.

To build a specific test, use the following command:

```bash
cmake --build build --target <test_name>
```

where `<test_name>` is the name of the test that you want to build.

For the getting started suite, only the `MultiSource/Applications/minisat` test is used.

To build only the `MultiSource/Applications/minisat` test, use the following command:

```bash
cmake --build build --target minisat
```

The x86 version of the LLVM test suite was compiled using the `gcc` and `g++` compilers native to the system.
The versions of `gcc` and `g++` used to compile the LLVM test suite were `gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0` and `g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0`, respectively.

The ARM version of the LLVM test suite was compiled using the `gcc` and `g++` compilers native to the system.
The versions of `gcc` and `g++` used to compile the LLVM test suite were `gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0` and `g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0`, respectively.

For more information about the LLVM test suite, please refer to the [official documentation](https://llvm.org/docs/GettingStartedTutorials.html).

#### Cross-compiling the LLVM test suite

To cross-compile the LLVM test suite, you will need to run a different `cmake` command that specifies the cross-compiler.

For example, to cross-compile the LLVM test suite for the RISC-V architecture, use the following command:

```bash
cmake . -B build -G "Unix Makefiles" -DTEST_SUITE_COLLECT_CODE_SIZE=OFF -DCMAKE_EXE_LINKER_FLAGS="-static" -DBUILD_SHARED_LIBS=OFF -DCMAKE_FIND_LIBRARY_SUFFIXES=".a" -DCMAKE_C_COMPILER=riscv64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=riscv64-linux-gnu-g++
```

This command is similar to the previous command, but it specifies the RISC-V cross-compiler using the `-DCMAKE_C_COMPILER` and `-DCMAKE_CXX_COMPILER` flags.
The above command is the command that was used to cross-compile the LLVM test suite for the RISC-V architecture version of the Getting Started Suite.

The versions of `riscv64-linux-gnu-gcc` and `riscv64-linux-gnu-g++` used to compile the LLVM test suite were `riscv64-linux-gnu-gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0` and `riscv64-linux-gnu-g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0`, respectively.
In order to install the specific cross-compilers used in the `cmake` command, you can use the following command:

```bash
apt-get install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu
```

To build a specific test for the RISC-V architecture, the command is the same as before:

```bash
cmake --build build --target <test_name>
```

where `<test_name>` is the name of the test that you want to build.

### NAS Parallel Benchmarks

A pre-requisite for this tutorial is to have the NAS Parallel Benchmarks installed.
The NAS Parallel Benchmarks can be obtained from the [gem5 Resources repository](https://github.com/gem5/gem5-resources.git) and are located in the `src/npb/disk-image/npb/npb-hooks` directory.

The folder structure of the NAS Parallel Benchmarks is as follows:

``` plaintext
NPB3.3.1/
├── NPB3.3-OMP/
├── NPB3.3-MPI/
└── NPB3.3-SER/
    ├── bin/
    ├── config/
    ├── common/
    ├── sys/
    ├── BT/
    ├── CG/
    ├── DC/
    ├── EP/
    ├── FT/
    ├── IS/
    ├── LU/
    ├── MG/
    ├── SP/
    └── UA/
```

To build the NAS Parallel Benchmarks, navigate to the `NPB3.3.1/NPB3.3-SER/config` directory. There, we need to add a `make.def` file that specifies the compiler and compiler flags, and a `suite.def` file that specifies the benchmarks to build.

Depending on the architecture, the `make.def` file will look different. This is because the compiler and compiler flags will be different for different architectures.
Particularly, the `-mcmodel` flag will be different for different architectures.

For x86, we used an example provided in the `NPB3.3.1/NPB3.3-SER/common/NAS.samples` directory
We modified the `make.def_gcc_x86` file to `make.def` and added flags to compile statically.
Our `make.def` file is provided in `getting-started/make-def-files/make_x86.def`.
The `make_x86.def` file assumes that you are compiling on a native x86 system.
The versions of `gcc` and `gfortan` used to compile the NAS Parallel Benchmarks for the `getting-started suite` were `gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0` and `gfortran (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0`, respectively.

For ARM, we modified the `make_x86.def` file to make the compiler flags compatible with ARM.
The particular flags that we modified were the [`-mcmodel` flag](https://gcc.gnu.org/onlinedocs/gcc/AArch64-Options.html) and the `-static` flag.
Our `make.def` file is provided in `getting-started/make-def-files/make_ARM.def`.
Please note that the `make_ARM.def` file assumes that you are compiling on a native ARM system.
The versions of `gcc` and `gfortran` used to compile the NAS Parallel Benchmarks for the `getting-started suite` were `gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0` and `gfortran (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0`, respectively.

For RISC-V, we modified the `make_x86.def` file to make the compiler flags and the compiler compatible with RISC-V.
The particular flags that we modified were the [`-mcmodel` flag](https://gcc.gnu.org/onlinedocs/gcc/RISC-V-Options.html) and the `-static` flag.
Our `make.def` file is provided in `getting-started/make-def-files/make_RISCV.def`.
The `make_RISCV.def` file assumes that you are compiling using a RISC-V cross-compiler.
The versions of `riscv64-linux-gnu-gcc` and `riscv64-linux-gnu-gfortran` used to compile the NAS Parallel Benchmarks for the `getting-started suite` were `riscv64-linux-gnu-gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0` and `riscv64-linux-gnu-gfortran (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0`, respectively.
In order to install the specific cross-compilers used in the `make_RISCV.def` file, you can use the following command:

```bash
apt-get install gcc-riscv64-linux-gnu gfortran-riscv64-linux-gnu
```

In case you want to cross-compile the NAS Parallel Benchmarks for a different cross-compiler, you will need to modify the `make.def` file accordingly.

The `suite.def` file specifies the benchmarks to build.

For example, to build the `IS` benchmark for size `S`, the `suite.def` file will look like this:

```plaintext
is S
```

Therefore, for the resources that are part of the Getting Started Suite, the `suite.def` file will look like this:

```plaintext
is S
lu S
cg S
bt S
ft S
```

To build the NAS Parallel Benchmarks, navigate to the `NPB3.3.1/NPB3.3-SER` directory and run the following command:

```bash
make suite
```

This command will build the benchmarks specified in the `suite.def` file.

### Using the Suite Binaries

In order to use custom binaries with the [gem5 Standard Library](https://www.gem5.org/documentation/gem5-stdlib/overview), you need to use the gem5 Resources infrastructure.

Documentation on how to use the gem5 Resources infrastructure can be found in the [gem5 Resources documentation](https://www.gem5.org/documentation/general_docs/gem5_resources/), and more specific instructions on how to use local Resources can be found in the [gem5 Local Resources Support](https://www.gem5.org/documentation/gem5-stdlib/local-resources-support).
