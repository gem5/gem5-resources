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
