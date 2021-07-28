#ifndef __HIPLOCKSBARRIERATOMIC_H__
#define __HIPLOCKSBARRIERATOMIC_H__

#include "hip/hip_runtime.h"
#include "hipLocks.h"

inline __device__ void hipBarrierAtomicSub(unsigned int * globalBarr,
                                            int * done,
                                            // numBarr represents the number of
                                            // WGs going to the barrier
                                            const unsigned int numBarr,
                                            int backoff,
                                            const bool isMasterThread)
{
  __syncthreads();
  if (isMasterThread)
  {
    *done = 0;

    // atomicInc acts as a store release, need TF to enforce ordering
    __threadfence();
    // atomicInc effectively adds 1 to atomic for each WG that's part of the
    // global barrier.
    /*
      HIP currently doesn't generate the correct code for atomicInc's here,
      so replace with an atomicAdd of 1 and assume no wraparound
    */
    //atomicInc(globalBarr, 0x7FFFFFFF);
    atomicAdd(globalBarr, 1);
  }
  __syncthreads();

  while (!*done)
  {
    if (isMasterThread)
    {
      /*
        For the tree barrier we expect only 1 WG from each CU to enter the
        global barrier.  Since we are assuming an equal amount of work for all
        CUs, we can use the # of WGs reaching the barrier for the compare value
        here.  Once the atomic's value == numBarr, then reset the value to 0 and
        proceed because all of the WGs have reached the global barrier.
      */
      if (atomicCAS(globalBarr, numBarr, 0) == 0) {
        // atomicCAS acts as a load acquire, need TF to enforce ordering
        __threadfence();
        *done = 1;
      }
      else { // increase backoff to avoid repeatedly hammering global barrier
        // (capped) exponential backoff
        backoff = (((backoff << 1) + 1) & (MAX_BACKOFF-1));
      }
    }
    __syncthreads();

    // do exponential backoff to reduce the number of times we pound the global
    // barrier
    if (!*done) {
      sleepFunc(backoff);
      __syncthreads();
    }
  }
}

inline __device__ void hipBarrierAtomic(unsigned int * barrierBuffers,
                                         // numBarr represents the number of
                                         // WGs going to the barrier
                                         const unsigned int numBarr,
                                         const bool isMasterThread)
{
  unsigned int * atomic1 = barrierBuffers;
  unsigned int * atomic2 = atomic1 + 1;
  __shared__ int done1, done2;
  __shared__ int backoff;

  if (isMasterThread) {
    backoff = 1;
  }
  __syncthreads();

  hipBarrierAtomicSub(atomic1, &done1, numBarr, backoff, isMasterThread);
  // second barrier is necessary to provide a facesimile for a sense-reversing
  // barrier
  hipBarrierAtomicSub(atomic2, &done2, numBarr, backoff, isMasterThread);
}

// does local barrier amongst all of the WGs on an CU
inline __device__ void hipBarrierAtomicSubLocal(unsigned int * perCUBarr,
                                                 int * done,
                                                 const unsigned int numWGs_thisCU,
                                                 const bool isMasterThread)
{
  __syncthreads();
  if (isMasterThread)
  {
    *done = 0;
    // atomicInc acts as a store release, need TF to enforce ordering locally
    __threadfence_block();
    /*
      atomicInc effectively adds 1 to atomic for each WG that's part of the
      barrier.  For the local barrier, this requires using the per-CU
      locations.
    */
    /*
      HIP currently doesn't generate the correct code for atomicInc's here,
      so replace with an atomicAdd of 1 and assume no wraparound
    */
    //atomicInc(perCUBarr, 0x7FFFFFFF);
    atomicAdd(perCUBarr, 1);
  }
  __syncthreads();

  while (!*done)
  {
    if (isMasterThread)
    {
      /*
        Once all of the WGs on this CU have incremented the value at atomic,
        then the value (for the local barrier) should be equal to the # of WGs
        on this CU.  Once that is true, then we want to reset the atomic to 0
        and proceed because all of the WGs on this CU have reached the local
        barrier.
      */
      if (atomicCAS(perCUBarr, numWGs_thisCU, 0) == 0) {
        // atomicCAS acts as a load acquire, need TF to enforce ordering
        // locally
        __threadfence_block();
        *done = 1;
      }
    }
    __syncthreads();
  }
}

// does local barrier amongst all of the WGs on an CU
inline __device__ void hipBarrierAtomicLocal(unsigned int * perCUBarrierBuffers,
                                              const unsigned int cuID,
                                              const unsigned int numWGs_thisCU,
                                              const bool isMasterThread,
                                              const int MAX_BLOCKS)
{
  // each CU has MAX_BLOCKS locations in barrierBuffers, so my CU's locations
  // start at barrierBuffers[cuID*MAX_BLOCKS]
  unsigned int * atomic1 = perCUBarrierBuffers + (cuID * MAX_BLOCKS);
  unsigned int * atomic2 = atomic1 + 1;
  __shared__ int done1, done2;

  hipBarrierAtomicSubLocal(atomic1, &done1, numWGs_thisCU, isMasterThread);
  // second barrier is necessary to approproximate a sense-reversing barrier
  hipBarrierAtomicSubLocal(atomic2, &done2, numWGs_thisCU, isMasterThread);
}

/*
  Helper function for joining the barrier with the atomic tree barrier.
*/
__attribute__((always_inline)) __device__ void joinBarrier_helper(unsigned int * barrierBuffers,
                                                                  unsigned int * perCUBarrierBuffers,
                                                                  const unsigned int numBlocksAtBarr,
                                                                  const int cuID,
                                                                  const int perCU_blockID,
                                                                  const int numWGs_perCU,
                                                                  const bool isMasterThread,
                                                                  const int MAX_BLOCKS) {
  if (numWGs_perCU > 1) {
    hipBarrierAtomicLocal(perCUBarrierBuffers, cuID, numWGs_perCU,
                           isMasterThread, MAX_BLOCKS);

    // only 1 WG per CU needs to do the global barrier since we synchronized
    // the WGs locally first
    if (perCU_blockID == 0) {
      hipBarrierAtomic(barrierBuffers, numBlocksAtBarr, isMasterThread);
    }

    // all WGs on this CU do a local barrier to ensure global barrier is
    // reached
    hipBarrierAtomicLocal(perCUBarrierBuffers, cuID, numWGs_perCU,
                           isMasterThread, MAX_BLOCKS);
  } else { // if only 1 WG on the CU, no need for the local barriers
    hipBarrierAtomic(barrierBuffers, numBlocksAtBarr, isMasterThread);
  }
}

#endif
