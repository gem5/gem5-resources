#ifndef __HIPLOCKSMUTEXFA_H__
#define __HIPLOCKSMUTEXFA_H__

#include "hip/hip_runtime.h"
#include "hipLocks.h"

inline __host__ hipError_t hipMutexCreateFA(hipMutex_t * const handle,
                                            const int mutexNumber)
{
  *handle = mutexNumber;
  return hipSuccess;
}

inline __device__ void hipMutexFALock(const hipMutex_t mutex,
                                      unsigned int * mutexBufferHeads,
                                      unsigned int * mutexBufferTails,
                                      const int NUM_CU)
{
  const bool isMasterThread = (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 &&
                               hipThreadIdx_z == 0);
  __shared__ unsigned int myTicketNum;
  __shared__ bool haveLock;
  const unsigned int maxTurnNum = 1000000000;

  unsigned int * ticketNumber = mutexBufferHeads + (mutex * NUM_CU);
  unsigned int * turnNumber =
      (unsigned int * )mutexBufferTails + (mutex * NUM_CU);

  __syncthreads();
  if (isMasterThread)
  {
    // load below provides ordering, no TF needed
    myTicketNum = atomicInc(ticketNumber, maxTurnNum);
    haveLock = false;
  }
  __syncthreads();
  while (!haveLock)
  {
    if (isMasterThread)
    {
      unsigned int currTicketNum = atomicAdd(turnNumber, 0);

      // it's my turn, I get the lock now
      if (currTicketNum == myTicketNum) {
        // above acts as a load acquire, so need TF to enforce ordering
        __threadfence();
        haveLock = true;
      }
    }
    __syncthreads();
  }
}

inline __device__ void hipMutexFAUnlock(const hipMutex_t mutex,
                                        unsigned int * mutexBufferTails,
                                        const int NUM_CU)
{
  const bool isMasterThread = (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 &&
                               hipThreadIdx_z == 0);
  const unsigned int maxTurnNum = 1000000000;
  unsigned int * turnNumber = mutexBufferTails + (mutex * NUM_CU);

  __syncthreads();
  if (isMasterThread) {
    // atomicInc acts as a store release, need TF to enforce ordering
    __threadfence();
    /*
      HIP currently doesn't generate the correct code for atomicInc's here,
      so replace with an atomicAdd of 1 and assume no wraparound
    */
    //atomicInc(turnNumber, maxTurnNum);
    atomicAdd(turnNumber, 1);
  }
  __syncthreads();
}

// same algorithm but uses per-CU lock
inline __device__ void hipMutexFALockLocal(const hipMutex_t mutex,
                                           const unsigned int cuID,
                                           unsigned int * mutexBufferHeads,
                                           unsigned int * mutexBufferTails,
                                           const int NUM_CU)
{
  // local variables
  const bool isMasterThread = (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 &&
                               hipThreadIdx_z == 0);
  __shared__ unsigned int myTicketNum;
  __shared__ bool haveLock;
  const unsigned int maxTurnNum = 100000000;

  unsigned int * ticketNumber = mutexBufferHeads + ((mutex * NUM_CU) +
                                                          cuID);
  unsigned int * turnNumber =
      (unsigned int *)mutexBufferTails + ((mutex * NUM_CU) + cuID);

  __syncthreads();
  if (isMasterThread)
  {
    myTicketNum = atomicInc(ticketNumber, maxTurnNum);
    haveLock = false;
  }
  __syncthreads();
  while (!haveLock)
  {
    if (isMasterThread)
    {
      unsigned int currTicketNum = atomicAdd(turnNumber, 0);

      // it's my turn, I get the lock now
      if (currTicketNum == myTicketNum) {
        // above acts as a load acquire, so need TF to enforce ordering locally
        __threadfence_block();
        haveLock = true;
      }
    }
    __syncthreads();
  }
}

// same algorithm but uses per-CU lock
inline __device__ void hipMutexFAUnlockLocal(const hipMutex_t mutex,
                                             const unsigned int cuID,
                                             unsigned int * mutexBufferTails,
                                             const int NUM_CU)
{
  const bool isMasterThread = (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 &&
                               hipThreadIdx_z == 0);
  const unsigned int maxTurnNum = 100000000;

  unsigned int * turnNumber = mutexBufferTails + ((mutex * NUM_CU) + cuID);

  __syncthreads();
  if (isMasterThread) {
    // atomicInc acts as a store release, need TF to enforce ordering locally
    __threadfence_block();
    /*
      HIP currently doesn't generate the correct code for atomicInc's here,
      so replace with an atomicAdd of 1 and assume no wraparound
    */
    //atomicInc(turnNumber, maxTurnNum);
    atomicAdd(turnNumber, 1);
  }
  __syncthreads();
}

#endif
