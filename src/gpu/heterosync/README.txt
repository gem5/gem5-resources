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
	1.  Spin Lock Semaphore: To approximate the "perform OP if > 0" feature of semaphores (on CPUs) they use atomicExch's to block the TB until the exchange returns true.  Requires more reads and writes on a GPU than a mutex.  Each TB sets the semaphore to the appropriate new values in the post and wait phases depending on the current capacity of the semaphore.
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

./allSyncPrims-1kernel <syncPrim> <numLdSt> <numTBs> <numCSIters>

<syncPrim> is a string that differs for each synchronization primitive to be run:
	// Barriers use hybrid local-global synchronization
	- atomicTreeBarrUniq - atomic tree barrier
	- atomicTreeBarrUniqLocalExch - atomic tree barrier with local exchange
	- lfTreeBarrUniq - lock-free tree barrier
	- lfTreeBarrUniqLocalExch - lock-free tree barrier with local exchange
	// global synchronization versions
	- spinMutex - spin lock mutex
	- spinMutexEBO - spin lock mutex with exponential backoff
	- sleepMutex - decentralized ticket lock
	- faMutex - centralized ticket lock (aka, fetch-and-add mutex)
	- spinSem1 - spin lock semaphore, semaphore size 1
	- spinSem2 - spin lock semaphore, semaphore size 2
	- spinSem10 - spin lock semaphore, semaphore size 10
	- spinSem120 - spin lock semaphore, semaphore size 120
	- spinSemEBO1 - spin lock semaphore with exponential backoff, semaphore size 1
	- spinSemEBO2 - spin lock semaphore with exponential backoff, semaphore size 2
	- spinSemEBO10 - spin lock semaphore with exponential backoff, semaphore size 10
	- spinSemEBO120 - spin lock semaphore with exponential backoff, semaphore size 120
	// local synchronization versions
	- spinMutexUniq - local spin lock mutex
	- spinMutexEBOUniq - local spin lock mutex with exponential backoff
	- sleepMutexUniq - local decentralized ticket lock
	- faMutexUniq - local centralized ticket lock
	- spinSemUniq1 - local spin lock semaphore, semaphore size 1
	- spinSemUniq2 - local spin lock semaphore, semaphore size 2
	- spinSemUniq10 - local spin lock semaphore, semaphore size 10
	- spinSemUniq120 - local spin lock semaphore, semaphore size 120
	- spinSemEBOUniq1 - local spin lock semaphore with exponential backoff, semaphore size 1
	- spinSemEBOUniq2 - local spin lock semaphore with exponential backoff, semaphore size 2
	- spinSemEBOUniq10 - local spin lock semaphore with exponential backoff, semaphore size 10
	- spinSemEBOUniq120 - local spin lock semaphore with exponential backoff, semaphore size 120

<numLdSt> is a positive integer representing how many loads and stores each thread will perform.  For the mutexes and semaphores, these accesses are all performed in the critical section.  For the barriers, these accesses use barriers to ensure that multiple threads are not accessing the same data.

<numTBs> is a positive integer representing the number of thread blocks (TBs) to execute.  For many of the microbenchmarks (especially the barriers), this number needs to be divisible by the number of SMs on the GPU.

<numCSIters> is a positive integer representing the number of iterations of the critical section.

IISWC '17 VERSION
-----------------

The version used in our IISWC '17 paper assumes a unified address space between the CPU and GPU.  Thus, it does not require any copies.  Moreover, this version is based on CUDA SDK 3.1 and HIP version 1.6, as this is the last version of CUDA that is fully supported by GPGPU-Sim and gem5, respectively, as of the release.  Later versions of CUDA and HIP allow additional C++ features, which may simplify the code or allow other optimizations.  Finally, this version is designed to run in the DeNovo ecosystem, which simulates a unified address space with multiple CPU cores and GPU CUs using a combination of Simics, GEMS, Garnet, and GPGPU-Sim.  In this ecosystem, we assume a SC-for-DRF style memory consistency model.  SC-for-DRF's ordering requirements are enforced by the epilogues and atomic operations.  We assume that the epilogues will self-invalidate all valid data in the local (L1) caches and flush per-CU/core store buffers to write through or obtain ownership for dirty data.

Similarly, to enforce the appropriate ordering requirements, we assume that the CUDA and HIP atomic operations have specific semantics:
 
Atomic      | Reprogrammed? | Load Acquire | Store Release |  Unpaired  |
atomicAdd   |               |              |               | X (LD, ST) |
atomicSub   |               |              |               | X (LD, ST) |
atomicExch  |      X        |              |      X (ST)   |            |
atomicMin   |               |              |               | X (LD, ST) |
atomicMax   |               |              |               | X (LD, ST) |
atomicInc   |               |              |      X (ST)   |   X (LD)   |
atomicDec   |               |              |      X (ST)   |   X (LD)   |
atomicCAS   |               |    X (LD)    |               |   X (ST)   |
atomicAnd   |      X        |              |               | X (LD, ST) |
atomicOr    |      X        |              |               | X (LD, ST) |
atomicXor   |      X        |    X (LD)    |               |            |

If your ecosystem does not make the same assumptions, then you will need to add the appropriate fences (e.g., CUDA's __threadfence() and __threadfence_block()) to ensure the proper ordering of requests in the memory system.  In the case of the HIP implementation, you may be able to use some OpenCL atomics with the desired orderings, but we left it as is to ensure portability and correctness with future versions of HIP that may not support this feature.

Reprogrammed Atomics:

In addition to the above assumptions about semantics for a given atomic, we have also reprogrammed some of the CUDA atomics to provide certain functionality we needed that CUDA doesn't provide:

- atomicAnd() was reprogrammed to have the same functionality as an atomicInc() but without store release semantics (i.e., atomicInc has store release semantics, atomicAnd does not).  We chose atomicAnd() for this because it was not used in any of our applications.  This change was necessary because atomicInc() sometimes needs store release semantics.
- atomicXor() was reprogrammed to do an atomic load (instead of an atomic RMW).
- atomicOr() was reprogrammed to do an (unpaired) atomic store (instead of an atomic RMW).  We chose atomicOr for the symmetry with atomicXor and because no applications used it.
- atomicExch() was not reprogrammed in the simulator, but we have re-purposed it assuming that the value returned by the atomicExch() is never returned or used in the program.  This allows us to treat atomicExch() as if it were an atomic store.  Thus, the programmer should consider an atomicExch() to be an atomic store.  All of the applications we have encountered thus far already did this.  In the simulator, we account for the read on the timing and functional sides.

Instruction-Centric vs. Data-Centric:

Common programming languages like C++ and OpenCL, which use a data-centric approach.  These languages identify atomic accesses by “tagging” a variable with the atomic qualifier.  These languages use an instruction-centric method for identifying which atomic accesses can/should use relaxed atomics instead of SC atomics; the accesses that can be relaxed have “memory_order_relaxed” appended to their accesses.  Since CUDA does not provide support for the same framework as C++ and OpenCL, we had to make a design decision about how to identify atomic accesses and how to identify which of those atomic accesses can use relaxed atomics vs. SC atomics.  We chose to use an instruction-centric method for identifying atomic vs. non-atomic accesses.  In this method, we designate certain CUDA atomic instructions as being load acquires, store releases, or unpaired (as denoted above).  Moreover, note that CUDA does not have direct support for atomic loads or stores.  HIP does support these, but only with OpenCL commands.

CUDA UVM VERSION
----------------

The CUDA UVM version is based on CUDA SDK 6.0, and uses CUDA's unified virtual memory to avoid making explicit copies of some of the arrays and structures.  Unlike the IISWC '17 version, this version does not make any assumptions about ordering atomics provide.  Nor does it require epilogues.  Instead, it adds the appropriate CUDA fence commands around atomic accesses to ensure the SC-for-DRF ordering is provided.  This version has been tested on a Pascal P100 GPU, but has not been tested as rigorously as the IISWC '17 version.

HIP UVM VERSION
----------------

The HIP UVM version is based on HIP 1.6, and uses HIP's unified virtual memory to avoid making explicit copies of some of the arrays and structures.  Unlike the IISWC '17 version, this version does not make any assumptions about ordering atomics provide.  Nor does it require epilogues.  Instead, it adds the appropriate HIP fence commands around atomic accesses to ensure the SC-for-DRF ordering is provided.  This version has been tested on a Vega 56 GPU, but has not been tested as rigorously as the IISWC '17 version.

CITATION
--------

If you publish work that uses these benchmarks, please cite the following papers:

1.  M. D. Sinclair, J. Alsop, and S. V. Adve, HeteroSync: A Benchmark Suite for Fine-Grained Synchronization on Tightly Coupled GPUs, in the IEEE International Symposium on Workload Characterization (IISWC), October 2017

2.  J. A. Stuart and J. D. Owens, “Efficient Synchronization Primitives for GPUs,” CoRR, vol. abs/1110.4623, 2011

ACKNOWLEDGEMENTS
----------------

This work was supported in part by a Qualcomm Innovation Fellowship for Sinclair, the National Science Foundation under grants CCF 13-02641 and CCF 16-19245, the Center for Future Architectures Research (C-FAR), a Semiconductor Research Corporation program sponsored by MARCO and DARPA, and the Center for Applications Driving Architectures (ADA), one of six centers of JUMP, a Semiconductor Research Corporation program co-sponsored by DARPA.
