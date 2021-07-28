#ifndef __HIPLOCKSBARRIERFAST_H__
#define __HIPLOCKSBARRIERFAST_H__

#include "hip/hip_runtime.h"
#include "hipLocks.h"

/*
  Helper function to set the passed in inVars flag to 1 (signifies that this WG
  has joined the barrier).
 */
inline __device__ void setMyInFlag(unsigned int * inVars,
                                   const unsigned int threadID,
                                   const unsigned int blockID) {
  if (threadID == 0)
  {
    // atomicExch acts as a store release, need TF to enforce ordering
    __threadfence();
    atomicExch((unsigned int *)(inVars + blockID), 1);
  }
  __syncthreads();
}

/*
  Helper function for the main WG of this group to spin, checking to see if
  all other WGs joining this barrier have joined or not.
 */
inline __device__ void spinOnInFlags(unsigned int * inVars,
                                     const int threadID,
                                     const int numThreads,
                                     const int numBlocks) {
  // local variables
  int done3 = 1;

  // "main" WG loops, checking if everyone else has joined the barrier.
  do
  {
    done3 = 1;

    /*
      Each thread in the main WG accesses a subset of the blocks, checking
      if they have joined the barrier yet or not.
    */
    for (int i = threadID; i < numBlocks; i += numThreads)
    {
      if (atomicAdd(&(inVars[i]), 0) != 1) {
        // acts as a load acquire, need TF to enforce ordering
        __threadfence();

        done3 = 0;
        // if one of them isn't ready, don't bother checking the others (just
        // increases traffic)
        break;
      }
    }
  } while (!done3);
  /*
    When all the necessary WGs have joined the barrier, the threads will
    reconverge here -- this avoids unnecessary atomic accesses for threads
    whose assigned WGs have already joined the barrier.
  */
  __syncthreads();
}

/*
  Helper function for the main WG of this group to spin, checking to see if
  all other WGs joining this barrier have joined or not.
*/
inline __device__ void spinOnInFlags_local(unsigned int * inVars,
                                           const int threadID,
                                           const int numThreads,
                                           const int numBlocks) {
  // local variables
  int done3 = 1;

  // "main" WG loops, checking if everyone else has joined the barrier.
  do
  {
    done3 = 1;

    /*
      Each thread in the main WG accesses a subset of the blocks, checking
      if they have joined the barrier yet or not.
    */
    for (int i = threadID; i < numBlocks; i += numThreads)
    {
      if (atomicAdd(&(inVars[i]), 0) != 1) {
        // acts as a load acquire, need TF to enforce ordering locally
        __threadfence_block();

        done3 = 0;
        // if one of them isn't ready, don't bother checking the others (just
        // increases traffic)
        break;
      }
    }
  } while (!done3);
  /*
    When all the necessary WGs have joined the barrier, the threads will
    reconverge here -- this avoids unnecessary atomic accesses for threads
    whose assigned WGs have already joined the barrier.
  */
  __syncthreads();
}

/*
  Helper function for main WG to set the outVars flags for all WGs at this
  barrier to notify them that everyone has joined the barrier and they can
  proceed.
*/
inline __device__ void setOutFlags(unsigned int * inVars,
                                   unsigned int * outVars,
                                   const int threadID,
                                   const int numThreads,
                                   const int numBlocks) {
  for (int i = threadID; i < numBlocks; i += numThreads)
  {
    atomicExch(&(inVars[i]), 0);
    atomicExch(&(outVars[i]), 1);
  }
  __syncthreads();
  // outVars acts as a store release, need TF to enforce ordering
  __threadfence();
}

/*
  Helper function for main WG to set the outVars flags for all WGs at this
  barrier to notify them that everyone has joined the barrier and they can
  proceed.
*/
inline __device__ void setOutFlags_local(unsigned int * inVars,
                                         unsigned int * outVars,
                                         const int threadID,
                                         const int numThreads,
                                         const int numBlocks) {
  for (int i = threadID; i < numBlocks; i += numThreads)
  {
    atomicExch(&(inVars[i]), 0);
    atomicExch(&(outVars[i]), 1);
  }
  __syncthreads();
  // outVars acts as a store release, need TF to enforce ordering locally
  __threadfence_block();
}

/*
  Helper function for each WG to spin waiting for its outVars flag to be set
  by the main WG.  When it is set, then this WG can safely exit the barrier.
*/
inline __device__ void spinOnMyOutFlag(unsigned int * inVars,
                                       unsigned int * outVars,
                                       const int blockID,
                                       const int threadID) {
  if (threadID == 0)
  {
    while (atomicAdd(&(outVars[blockID]), 0) != 1) { ; }

    atomicExch(&(inVars[blockID]), 0);
    atomicExch(&(outVars[blockID]), 0);
    // these stores act as a store release, need TF to enforce ordering
    __threadfence();
  }
  __syncthreads();
}

/*
  Helper function for each WG to spin waiting for its outVars flag to be set
  by the main WG.  When it is set, then this WG can safely exit the barrier.
*/
inline __device__ void spinOnMyOutFlag_local(unsigned int * inVars,
                                             unsigned int * outVars,
                                             const int blockID,
                                             const int threadID) {
  if (threadID == 0)
  {
    while (atomicAdd(&(outVars[blockID]), 0) != 1) { ; }

    atomicExch(&(inVars[blockID]), 0);
    atomicExch(&(outVars[blockID]), 0);
    // these stores act as a store release, need TF to enforce ordering locally
    __threadfence_block();
  }
  __syncthreads();
}

__device__ void hipBarrier(unsigned int * barrierBuffers,
                           const int arrayStride,
                           const unsigned int numBlocksAtBarr)
{
  // local variables
  const int threadID = hipThreadIdx_x;
  const int blockID = hipBlockIdx_x;
  const int numThreads = hipBlockDim_x;
  // ** NOTE: setting numBlocks like this only works if the first WG on
  // each CU joins the global barrier
  const int numBlocks = numBlocksAtBarr;
  unsigned int * const inVars  = barrierBuffers;
  unsigned int * const outVars = barrierBuffers + arrayStride;

  /*
    Thread 0 from each WG sets its 'private' flag in the in array to 1 to
    signify that it has joined the barrier.
  */
  setMyInFlag(inVars, threadID, blockID);

  // WG 0 is the "main" WG for the global barrier
  if (blockID == 0)
  {
    // "main" WG loops, checking if everyone else has joined the barrier.
    spinOnInFlags(inVars, threadID, numThreads, numBlocks);

    /*
      Once all the WGs arrive at the barrier, the main WG resets them to
      notify everyone else that they can move forward beyond the barrier --
      again each thread in the main WG takes a subset of the necessary WGs
      and sets their in flag to 0 and out flag to 1.
    */
    setOutFlags(inVars, outVars, threadID, numThreads, numBlocks);
  }

  /*
    All WGs (including the main one) spin, checking to see if the main one
    set their out location yet -- if it did, then they can move ahead
    because the barrier is done.
  */
  spinOnMyOutFlag(inVars, outVars, blockID, threadID);
}

// same algorithm but per-CU synchronization
__device__ void hipBarrierLocal(// for global barrier
                                unsigned int * barrierBuffers,
                                const unsigned int numBlocksAtBarr,
                                const int arrayStride,
                                // for local barrier
                                unsigned int * perCUBarrierBuffers,
                                const unsigned int cuID,
                                const unsigned int numWGs_perCU,
                                const unsigned int perCU_blockID,
                                const bool isLocalGlobalBarr,
                                const int MAX_BLOCKS)
{
  // local variables
  const int threadID = hipThreadIdx_x;
  const int numThreads = hipBlockDim_x;
  const int numBlocks = numWGs_perCU;
  /*
    Each CU has MAX_BLOCKS*2 locations in perCUBarrierBuffers, so my CU's
    inVars locations start at perCUBarrierBuffers[cuID*2*MAX_BLOCKS] and my
    CU's outVars locations start at
    perCUBarrierBuffers[cuID*2*MAX_BLOCKS + MAX_BLOCKS].
  */
  unsigned int * const inVars  = perCUBarrierBuffers + (MAX_BLOCKS * cuID * 2);
  unsigned int * const outVars = perCUBarrierBuffers + ((MAX_BLOCKS * cuID * 2) + MAX_BLOCKS);

  /*
    Thread 0 from each WG sets its 'private' flag in the in array to 1 to
    signify that it has joined the barrier.
  */
  setMyInFlag(inVars, threadID, perCU_blockID);

  // first WG on this CU is the "main" WG for the local barrier
  if (perCU_blockID == 0)
  {
    // "main" WG loops, checking if everyone else has joined the barrier.
    spinOnInFlags_local(inVars, threadID, numThreads, numBlocks);

    /*
      If we are calling the global tree barrier from within the local tree
      barrier, call it here.  Now that all of the WGs on this CU have joined
      the local barrier, WG 0 on this CU joins the global barrier.
    */
    if (isLocalGlobalBarr) {
      hipBarrier(barrierBuffers, arrayStride, numBlocksAtBarr);
    }

    /*
      Once all the WGs arrive at the barrier, the main WG resets their inVar
      and sets their outVar to notify everyone else that they can move
      forward beyond the barrier -- each thread in the main WG takes a subset
      of the necessary WGs and sets their in flag to 0 and out flag to 1.
    */
    setOutFlags_local(inVars, outVars, threadID, numThreads, numBlocks);
  }

  /*
    All WGs (including the main one) spin, checking to see if the main WG
    set their out location yet -- if it did, then they can move ahead
    because the barrier is done.
  */
  spinOnMyOutFlag_local(inVars, outVars, perCU_blockID, threadID);
}

/*
  Decentralized tree barrier that has 1 WG per CU join the global decentralized
  barrier in the middle, then sets the out flags of the others on this CU to 1
  after returning.  This avoids the need for a second local barrier after the
  global barrier.
*/
__device__ void hipBarrierLocalGlobal(// for global barrier
                                      unsigned int * barrierBuffers,
                                      const unsigned int numBlocksAtBarr,
                                      const int arrayStride,
                                      // for local barrier
                                      unsigned int * perCUBarrierBuffers,
                                      const unsigned int cuID,
                                      const unsigned int numWGs_perCU,
                                      const unsigned int perCU_blockID,
                                      const int MAX_BLOCKS)
{
  // will call global barrier within it
  hipBarrierLocal(barrierBuffers, numBlocksAtBarr, arrayStride,
                  perCUBarrierBuffers, cuID, numWGs_perCU, perCU_blockID,
                  true, MAX_BLOCKS);
}

/*
  Helper function for joining the barrier with the 'lock-free' tree barrier.
*/
__device__ void joinLFBarrier_helper(unsigned int * barrierBuffers,
                                     unsigned int * perCUBarrierBuffers,
                                     const unsigned int numBlocksAtBarr,
                                     const int cuID,
                                     const int perCU_blockID,
                                     const int numWGs_perCU,
                                     const int arrayStride,
                                     const int MAX_BLOCKS) {
  if (numWGs_perCU > 1) {
    hipBarrierLocalGlobal(barrierBuffers, numBlocksAtBarr, arrayStride,
                          perCUBarrierBuffers, cuID, numWGs_perCU,
                          perCU_blockID, MAX_BLOCKS);
  } else { // if only 1 WG on the CU, no need for the local barriers
    hipBarrier(barrierBuffers, arrayStride, numBlocksAtBarr);
  }
}

#endif
