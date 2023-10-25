---
title: GCN3 HeteroSync Tests
tags:
    - x86
    - amdgpu
layout: default
permalink: resources/heterosync
shortdoc: >
    Resources to build a disk image with the GCN3 HeteroSync workloads.
---

# Resource: HeteroSync

[HeteroSync](https://github.com/mattsinc/heterosync) is a benchmark suite used
to test the performance of various types of fine-grained synchronization on
tightly-coupled GPUs. The version in gem5-resources contains only the HIP code.

Below the README details the various synchronization primitives and the other
command-line arguments for use with heterosync.

## Compilation
```
cd src/gpu/heterosync
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu:v22-1 make release-gfx8
```

The release-gfx8 target builds for gfx801, a GCN3-based APU, and gfx803, a
GCN3-based dGPU. There are other targets (release) that build for GPU types
that are currently unsupported in gem5.

## Running HeteroSync on GCN3_X86/gem5.opt

HeteroSync has multiple applications that can be run (see below).  For example, to run sleepMutex with 10 ld/st per thread, 16 WGs, and 4 iterations of the critical section:

```
docker run -u $UID:$GID --volume $(pwd):$(pwd) -w $(pwd) ghcr.io/gem5/gcn-gpu:v22-1 gem5/build/GCN3_X86/gem5.opt gem5/configs/example/apu_se.py -n 3 -c bin/allSyncPrims-1kernel --options="sleepMutex 10 16 4"
```

## Pre-built binary

<http://dist.gem5.org/dist/v22-1/test-progs/heterosync/gcn3/allSyncPrims-1kernel>

Information from original HeteroSync README included below:

These files are provided AS IS, and can be improved in many aspects. While we performed some performance optimization, there is more to be done. We do not claim that this is the most optimal implementation. The code is presented as a representative case of a CUDA and HIP implementations of these workloads only.  It is NOT meant to be interpreted as a definitive answer to how well this application can perform on GPUs, CUDA, or HIP.  If any of you are interested in improving the performance of these benchmarks, please let us know or submit a pull request on GitHub.

BACKGROUND INFORMATION
----------------------

Structure: All of the HeteroSync microbenchmarks are run from a single main function.  Each of the microbenchmarks has a separate .cu (CUDA) file that contains the code for its lock and unlock functions.  In the HIP version, these files are header files, because of HIP's requirements for compilation.

Contents: The following Synchronization Primitives (SyncPrims) microbenchmarks are included in HeteroSync:

- Centralized Mutexes:
	1.  Spin Mutex Lock: A fairly standard spin-lock implementation.  It repeatedly tries to obtain the lock.  This version has high contention and a lot of atomic accesses since all TBs are spinning on the same lock variable.
	2.  Spin Mutex Lock with Backoff: Standard backoff version of a spin lock where they “sleep” for a short period of time between each unsuccessful acquire.  They use a linear backoff instead of exponential backoff.  On the first failed acquire they will “sleep” for I_min; every subsequent failed read will increase the “sleep” time (up to I_max).
	3.  Fetch-and-Add (FA) Mutex Lock (similar to Ticket/Queue-style Locks): To make their spin lock fair and have a deterministic number of atomic accesses per operation they also implement this queue-style spin lock.  Every TB uses an atomic to get a "ticket" for when they'll get the lock.  The TBs poll the “current ticket” location until their turn arrives (when it does they acquire the lock).  FAMutex uses backoff in the polling section of this lock to reduce contention.
	4.  Ring Buffer-based Sleeping Mutex Lock: Each TB places itself on the end of the buffer and repeatedly checks if is now at the front of the buffer.  To unlock they increment the head pointer.  In the original paper they found that performance is bad for this one because it requires more reads and writes to the head pointer are serialized.
- Centralized Semaphores:
	1.  Spin Lock Semaphore: To approximate the "perform OP if &gt; 0" feature of semaphores (on CPUs) they use atomicExch's to block the TB until the exchange returns true.  Requires more reads and writes on a GPU than a mutex.  Each TB sets the semaphore to the appropriate new values in the post and wait phases depending on the current capacity of the semaphore.
	2.  Spin Lock Semaphore with Backoff: As with the mutexes, they add a linear backoff to decrease contention.  The backoff is only in the wait() phase because usually more TBs are waiting, not posting.
- Barriers:
	1.  Atomic Barrier: a two-stage atomic counter barrier.  There are several versions of this barrier: a tree barrier and a second version that exchanges data locally on a CU before joining the global tree barrier.
	2.  Lock-Free Barrier: a decentralized, sleeping based approach that doesn't require atomics.  Each TB sets a flag in a distinct memory location.  Once all TBs have set their flag, then each TB does an intra-block barrier between its warps.  Like the atomic barrier, there are two versions.

All microbenchmarks access shared data that requires synchronization.

A subsequent commit will add the Relaxed Atomics microbenchmarks discussed in our paper.

USAGE
-----

Compilation:

Since all of the microbenchmarks run from a single main function, users only need to compile the entire suite once in order to use any of the microbenchmarks.  You will need to set CUDA_DIR in the Makefile in order to properly compile the code.  To use HIP, you will need to set HIP_PATH for compilation to work correctly.

Running:

The usage of the microbenchmarks is as follows:

./allSyncPrims-1kernel &lt;syncPrim&gt; &lt;numLdSt&gt; &lt;numTBs&gt; &lt;numCSIters&gt;

where &lt;syncPrim&gt; is a string that differs for each synchronization primitive to be run:
	// Barriers use hybrid local-global synchronization
	* atomicTreeBarrUniq: atomic tree barrier
	* atomicTreeBarrUniqLocalExch: atomic tree barrier with local exchange
	* lfTreeBarrUniq: lock*free tree barrier
	* lfTreeBarrUniqLocalExch: lock*free tree barrier with local exchange
	// global synchronization versions
	* spinMutex: spin lock mutex
	* spinMutexEBO: spin lock mutex with exponential backoff
	* sleepMutex: decentralized ticket lock
	* faMutex: centralized ticket lock (aka, fetch*and*add mutex)
	* spinSem1: spin lock semaphore, semaphore size 1
	* spinSem2: spin lock semaphore, semaphore size 2
	* spinSem10: spin lock semaphore, semaphore size 10
	* spinSem120: spin lock semaphore, semaphore size 120
	* spinSemEBO1: spin lock semaphore with exponential backoff, semaphore size 1
	* spinSemEBO2: spin lock semaphore with exponential backoff, semaphore size 2
	* spinSemEBO10: spin lock semaphore with exponential backoff, semaphore size 10
	* spinSemEBO120: spin lock semaphore with exponential backoff, semaphore size 120
	// local synchronization versions
	* spinMutexUniq: local spin lock mutex
	* spinMutexEBOUniq: local spin lock mutex with exponential backoff
	* sleepMutexUniq: local decentralized ticket lock
	* faMutexUniq: local centralized ticket lock
	* spinSemUniq1: local spin lock semaphore, semaphore size 1
	* spinSemUniq2: local spin lock semaphore, semaphore size 2
	* spinSemUniq10: local spin lock semaphore, semaphore size 10
	* spinSemUniq120: local spin lock semaphore, semaphore size 120
	* spinSemEBOUniq1: local spin lock semaphore with exponential backoff, semaphore size 1
	* spinSemEBOUniq2: local spin lock semaphore with exponential backoff, semaphore size 2
	* spinSemEBOUniq10: local spin lock semaphore with exponential backoff, semaphore size 10
	* spinSemEBOUniq120: local spin lock semaphore with exponential backoff, semaphore size 120

&lt;numLdSt&gt; is a positive integer representing how many loads and stores each thread will perform.  For the mutexes and semaphores, these accesses are all performed in the critical section.  For the barriers, these accesses use barriers to ensure that multiple threads are not accessing the same data.

&lt;numTBs&gt; is a positive integer representing the number of thread blocks (TBs) to execute.  For many of the microbenchmarks (especially the barriers), this number needs to be divisible by the number of SMs on the GPU.

&lt;numCSIters&gt; is a positive integer representing the number of iterations of the critical section.

HIP UVM VERSION
----------------

The HIP UVM version is based on HIP 4.0, and uses HIP's unified virtual memory to avoid making explicit copies of some of the arrays and structures.  Unlike the IISWC '17 version, this version does not make any assumptions about ordering atomics provide.  Nor does it require epilogues.  Instead, it adds the appropriate HIP fence commands around atomic accesses to ensure the SC-for-DRF ordering is provided.  This version has been tested on a Vega 20 GPU, but has not been tested as rigorously as the IISWC '17 version.

CITATION
--------

If you publish work that uses these benchmarks, please cite the following papers:

1.  M. D. Sinclair, J. Alsop, and S. V. Adve, HeteroSync: A Benchmark Suite for Fine-Grained Synchronization on Tightly Coupled GPUs, in the IEEE International Symposium on Workload Characterization (IISWC), October 2017

2.  J. A. Stuart and J. D. Owens, “Efficient Synchronization Primitives for GPUs,” CoRR, vol. abs/1110.4623, 2011

ACKNOWLEDGEMENTS
----------------

This work was supported in part by a Qualcomm Innovation Fellowship for Sinclair, the National Science Foundation under grants CCF 13-02641 and CCF 16-19245, the Center for Future Architectures Research (C-FAR), a Semiconductor Research Corporation program sponsored by MARCO and DARPA, and the Center for Applications Driving Architectures (ADA), one of six centers of JUMP, a Semiconductor Research Corporation program co-sponsored by DARPA.
