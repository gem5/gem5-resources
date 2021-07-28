#ifndef __HIPLOCKSMUTEXSPIN_H__
#define __HIPLOCKSMUTEXSPIN_H__

#include "hip/hip_runtime.h"

inline __host__ hipError_t hipMutexCreateSpin(hipMutex_t * const handle,
                                              const int mutexNumber)
{
  *handle = mutexNumber;
  return hipSuccess;
}

// This is the brain dead algorithm. Just spin on an atomic until you get the
// lock.
__device__ void hipMutexSpinLock(const hipMutex_t mutex,
                                 unsigned int * mutexBufferHeads,
                                 const int NUM_CU)
{
  __shared__ int done;
  const bool isMasterThread = ((hipThreadIdx_x == 0) && (hipThreadIdx_y == 0) &&
                               (hipThreadIdx_z == 0));
  if (isMasterThread) { done = 0; }
  __syncthreads();

  while (!done)
  {
    __syncthreads();
    if (isMasterThread)
    {
      if (atomicCAS(mutexBufferHeads + (mutex * NUM_CU), 0, 1) == 0) {
        // atomicCAS acts as a load acquire, need TF to enforce ordering
        __threadfence();
        done = 1;
      }
    }
    __syncthreads();
  }
}

__device__ void hipMutexSpinUnlock(const hipMutex_t mutex,
                                   unsigned int * mutexBufferHeads,
                                   const int NUM_CU)
{
  __syncthreads();
  if (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 && hipThreadIdx_z == 0)
  {
    // atomicExch acts as a store release, need TF to enforce ordering
    __threadfence();
    atomicExch(mutexBufferHeads + (mutex * NUM_CU), 0);
  }
  __syncthreads();
}

// same algorithm but uses local TF instead because data is local
__device__ void hipMutexSpinLockLocal(const hipMutex_t mutex,
                                      const unsigned int cuID,
                                      unsigned int * mutexBufferHeads,
                                      const int NUM_CU)
{
  __shared__ int done;
  const bool isMasterThread = ((hipThreadIdx_x == 0) && (hipThreadIdx_y == 0) &&
                               (hipThreadIdx_z == 0));
  if (isMasterThread) { done = 0; }
  __syncthreads();

  while (!done)
  {
    __syncthreads();
    if (isMasterThread)
    {
      if (atomicCAS(mutexBufferHeads + ((mutex * NUM_CU) + cuID), 0, 1) == 0)
      {
        // atomicCAS acts as a load acquire, need TF to enforce ordering locally
        __threadfence_block();
        done = 1;
      }
    }
    __syncthreads();
  }
}

// same algorithm but uses local TF instead because data is local
__device__ void hipMutexSpinUnlockLocal(const hipMutex_t mutex,
                                        const unsigned int cuID,
                                        unsigned int * mutexBufferHeads,
                                        const int NUM_CU)
{
  __syncthreads();
  if ((hipThreadIdx_x == 0) && (hipThreadIdx_y == 0) && (hipThreadIdx_z == 0))
  {
    // atomicExch acts as a store release, need TF to enforce ordering locally
    __threadfence_block();
    // mutex math allows us to access the appropriate per-CU spin mutex location
    atomicExch(mutexBufferHeads + ((mutex * NUM_CU) + cuID), 0);
  }
  __syncthreads();
}

#endif
