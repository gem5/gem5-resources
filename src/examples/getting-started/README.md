# Getting Started Suite

This suite is designed to provide a set of pre-compiled familiar workloads to help one get started with gem5.
These workloads have been cherry-picked from popular benchmarks and applications.
These benchmarks are as follows:

- [Breadth-First Search (BFS)](https://github.com/sbeamer/gapbs/blob/master/src/bfs.cc) from the [GAPBS Benchmark Suite](https://github.com/sbeamer/gapbs)
- [Triangle Counting (TC)](https://github.com/sbeamer/gapbs/blob/master/src/tc.cc) from the [GAPBS Benchmark Suite](https://github.com/sbeamer/gapbs)
- [Minisat](https://github.com/llvm/llvm-test-suite/tree/main/MultiSource/Applications/minisat) from the [LLVM Test Suite](https://github.com/llvm/llvm-test-suite)
- Integer Sort (IS) - Size S from the [NAS Parallel Benchmarks](https://www.nas.nasa.gov/publications/npb.html)
- Lower-Upper Gauss-Seidel (LU) - Size S from the [NAS Parallel Benchmarks](https://www.nas.nasa.gov/publications/npb.html)
- Conjugate Gradient (CG) - Size S from the [NAS Parallel Benchmarks](https://www.nas.nasa.gov/publications/npb.html)
- Block Tri-Diagonal (BT) - Size S from the [NAS Parallel Benchmarks](https://www.nas.nasa.gov/publications/npb.html)
- Fourier Transform (FT) - Size S from the [NAS Parallel Benchmarks](https://www.nas.nasa.gov/publications/npb.html)
- Matrix Multiply from the [gem5 Resources repository](https://github.com/gem5/gem5-resources/tree/stable/src/matrix-multiply)

Since some of these benchmarks accept arguments, the suite has pre-configured these arguments as follows:

- BFS: `-g 10 -n 10`, where `-g` is the $\\log_2{}$ of the number of vertices and `-n` is the number of iterations.
- TC: `-g 10 -n 10`, where `-g` is the $\\log_2{}$ of the number of vertices and `-n` is the number of iterations.
- Minisat: `-verbosity=0 short.cnf`, where `short.cnf` is a short CNF file provided by the LLVM Test Suite, and `verbosity=0` is to suppress the output. The CNF file can be viewed [here](https://github.com/llvm/llvm-test-suite/blob/main/MultiSource/Applications/minisat/short.cnf).

The suite is designed to be compatible with the [gem5 Standard Library](https://www.gem5.org/documentation/gem5-stdlib/overview), using the Suite infrastructure.
More information on how to use the Suite can be found in the [gem5 Standard Library documentation](https://www.gem5.org//documentation/gem5-stdlib/suites).


## Prebuilt resources on gem5 resources

The above mentioned resources are already built for X86, ARM and RISC-V ISAs. They can be found at the following links:

- The suite with all the workloads for the x86 ISA is at [x86-getting-started-benchmark-suite](https://resources.gem5.org/resources/x86-getting-started-benchmark-suite?version=1.0.0). This suite contains the following workloads:

  - [x86-llvm-minisat-run](https://resources.gem5.org/resources/x86-llvm-minisat-run?version=1.0.0)
  - [x86-gapbs-bfs-run](https://resources.gem5.org/resources/x86-gapbs-bfs-run?version=1.0.0)
  - [x86-gapbs-tc-run](https://resources.gem5.org/resources/x86-gapbs-tc-run?version=1.0.0)
  - [x86-npb-is-size-s-run](https://resources.gem5.org/resources/x86-npb-is-size-s-run?version=1.0.0)
  - [x86-npb-lu-size-s-run](https://resources.gem5.org/resources/x86-npb-lu-size-s-run?version=1.0.0)
  - [x86-npb-cg-size-s-run](https://resources.gem5.org/resources/x86-npb-cg-size-s-run?version=1.0.0)
  - [x86-npb-bt-size-s-run](https://resources.gem5.org/resources/x86-npb-bt-size-s-run?version=1.0.0)
  - [x86-npb-ft-size-s-run](https://resources.gem5.org/resources/x86-npb-ft-size-s-run?version=1.0.0)
  - [x86-matrix-multiply-run](https://resources.gem5.org/resources/x86-matrix-multiply-run?version=1.0.0)

- The suite with all the workloads for the ARM ISA is at [arm-getting-started-benchmark-suite](https://resources.gem5.org/resources/arm-getting-started-benchmark-suite?version=1.0.0). This suite contains the following workloads:

  - [arm-llvm-minisat-run](https://resources.gem5.org/resources/arm-llvm-minisat-run?database=gem5-resources&version=1.0.0)
  - [arm-gapbs-bfs-run](https://resources.gem5.org/resources/arm-gapbs-bfs-run?database=gem5-resources&version=1.0.0)
  - [arm-gapbs-tc-run](https://resources.gem5.org/resources/arm-gapbs-tc-run?database=gem5-resources&version=1.0.0)
  - [arm-npb-is-size-s-run](https://resources.gem5.org/resources/arm-npb-is-size-s-run?database=gem5-resources&version=1.0.0)
  - [arm-npb-lu-size-s-run](https://resources.gem5.org/resources/arm-npb-lu-size-s-run?database=gem5-resources&version=1.0.0)
  - [arm-npb-cg-size-s-run](https://resources.gem5.org/resources/arm-npb-cg-size-s-run?database=gem5-resources&version=1.0.0)
  - [arm-npb-bt-size-s-run](https://resources.gem5.org/resources/arm-npb-bt-size-s-run?database=gem5-resources&version=1.0.0)
  - [arm-npb-ft-size-s-run](https://resources.gem5.org/resources/arm-npb-ft-size-s-run?database=gem5-resources&version=1.0.0)
  - [arm-matrix-multiply-run](https://resources.gem5.org/resources/arm-matrix-multiply-run?database=gem5-resources&version=1.0.0)

- The suite with all the workloads for the RISC-V ISA is at [riscv-getting-started-benchmark-suite](https://resources.gem5.org/resources/riscv-getting-started-benchmark-suite?version=1.0.0). This suite contains the following workloads:

  - [riscv-llvm-minisat-run](https://resources.gem5.org/resources/riscv-llvm-minisat-run?version=1.0.0)
  - [riscv-gapbs-bfs-run](https://resources.gem5.org/resources/riscv-gapbs-bfs-run?version=1.0.0)
  - [riscv-gapbs-tc-run](https://resources.gem5.org/resources/riscv-gapbs-tc-run?version=1.0.0)
  - [riscv-npb-is-size-s-run](https://resources.gem5.org/resources/riscv-npb-is-size-s-run?version=1.0.0)
  - [riscv-npb-lu-size-s-run](https://resources.gem5.org/resources/riscv-npb-lu-size-s-run?version=1.0.0)
  - [riscv-npb-cg-size-s-run](https://resources.gem5.org/resources/riscv-npb-cg-size-s-run?version=1.0.0)
  - [riscv-npb-bt-size-s-run](https://resources.gem5.org/resources/riscv-npb-bt-size-s-run?version=1.0.0)
  - [riscv-npb-ft-size-s-run](https://resources.gem5.org/resources/riscv-npb-ft-size-s-run?version=1.0.0)
  - [riscv-matrix-multiply-run](https://resources.gem5.org/resources/riscv-matrix-multiply-run?version=1.0.0)
