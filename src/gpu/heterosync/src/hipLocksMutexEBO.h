#ifndef __HIPLOCKSMUTEXEBO_H__
#define __HIPLOCKSMUTEXEBO_H__

#include "hip/hip_runtime.h"
#include "hipLocks.h"

inline __host__ hipError_t hipMutexCreateEBO(hipMutex_t * const handle,
                                             const int mutexNumber)
{
  *handle = mutexNumber;
  return hipSuccess;
}

inline __device__ void hipMutexEBOLock(const hipMutex_t mutex,
                                       unsigned int * mutexBufferHeads,
                                       const int NUM_CU)
{
  // local variables
  __shared__ int done, backoff;
  const bool isMasterThread = (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 &&
                               hipThreadIdx_z == 0);
  unsigned int * mutexHeadPtr = NULL;

  if (isMasterThread)
  {
    backoff = 1;
    done = 0;
    mutexHeadPtr = (mutexBufferHeads + (mutex * NUM_CU));
  }
  __syncthreads();
  while (!done)
  {
    __syncthreads();
    if (isMasterThread)
    {
      // try to acquire the lock
      if (atomicCAS(mutexHeadPtr, 0, 1) == 0) {
        // atomicCAS acts as a load acquire, need TF to enforce ordering
        __threadfence();
        done = 1;
      }
      else
      {
        // if we failed in acquiring the lock, wait for a little while before
        // trying again
        sleepFunc(backoff);
        // (capped) exponential backoff
        backoff = (((backoff << 1) + 1) & (MAX_BACKOFF-1));
      }
    }
    __syncthreads();
  }
}

inline __device__ void hipMutexEBOUnlock(const hipMutex_t mutex,
                                         unsigned int * mutexBufferHeads,
                                         const int NUM_CU)
{
  __syncthreads();
  if (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 && hipThreadIdx_z == 0) {
    // atomicExch acts as a store release, need TF to enforce ordering
    __threadfence();
    atomicExch(mutexBufferHeads + (mutex * NUM_CU), 0); // release the lock
  }
  __syncthreads();
}

// same locking algorithm but with local scope
inline __device__ void hipMutexEBOLockLocal(const hipMutex_t mutex,
                                            const unsigned int cuID,
                                            unsigned int * mutexBufferHeads,
                                            const int NUM_CU)
{
  // local variables
  __shared__ int done, backoff;
  const bool isMasterThread = (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 &&
                               hipThreadIdx_z == 0);
  unsigned int * mutexHeadPtr = NULL;

  if (isMasterThread)
  {
    backoff = 1;
    done = 0;
    mutexHeadPtr = (mutexBufferHeads + ((mutex * NUM_CU) + cuID));
  }
  __syncthreads();
  while (!done)
  {
    __syncthreads();
    if (isMasterThread)
    {
      // try to acquire the lock
      if (atomicCAS(mutexHeadPtr, 0, 1) == 0) {
        // atomicCAS acts as a load acquire, need TF to enforce ordering locally
        __threadfence_block();
        done = 1;
      }
      else
      {
        // if we failed in acquiring the lock, wait for a little while before
        // trying again
        sleepFunc(backoff);
        // (capped) exponential backoff
        backoff = (((backoff << 1) + 1) & (MAX_BACKOFF-1));
      }
    }
    __syncthreads();
  }
}

// same unlock algorithm but with local scope
inline __device__ void hipMutexEBOUnlockLocal(const hipMutex_t mutex,
                                              const unsigned int cuID,
                                              unsigned int * mutexBufferHeads,
                                              const int NUM_CU)
{
  __syncthreads();
  if (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 && hipThreadIdx_z == 0) {
    // atomicExch acts as a store release, need TF to enforce ordering locally
    __threadfence_block();
    atomicExch(mutexBufferHeads + ((mutex * NUM_CU) + cuID), 0); // release the lock
  }
  __syncthreads();
}

#endif
