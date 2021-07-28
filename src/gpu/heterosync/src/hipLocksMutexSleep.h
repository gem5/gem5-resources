#ifndef __HIPLOCKMUTEXSLEEP_H__
#define __HIPLOCKMUTEXSLEEP_H__

#include "hip/hip_runtime.h"
#include "hipLocks.h"

inline __host__ hipError_t hipMutexCreateSleep(hipMutex_t * const handle,
                                               const int mutexNumber)
{
  *handle = mutexNumber;
  return hipSuccess;
}

/*
  Instead of constantly pounding an atomic to try and lock the mutex, we simply
  put ourselves into a ring buffer. Then we check our location in the ring
  buffer to see if it's been set to 1 -- when it has, it is our turn.  When
  we're done, unset our location and set the next location to 1.

  locks the mutex. must be called by the entire WG.
*/
__device__ unsigned int hipMutexSleepLock(const hipMutex_t mutex,
                                          int * mutexBuffers,
                                          unsigned int * mutexBufferTails,
                                          const int maxRingBufferSize,
                                          const int arrayStride,
                                          const int NUM_CU)
{
  __syncthreads();

  // local variables
  const bool isMasterThread = (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 &&
                               hipThreadIdx_z == 0);

  unsigned int * const ringBufferTailPtr = mutexBufferTails + (mutex * NUM_CU);
  int * const ringBuffer = (int *)mutexBuffers + (mutex * NUM_CU) * arrayStride;

  __shared__ unsigned int myRingBufferLoc;
  __shared__ bool haveLock;
  __shared__ int backoff;

  // this is a fire-and-forget atomic.
  if (isMasterThread)
  {
    /*
      Don't need store release semantics -- the atomicAdd below determines
      the happens-before ordering here.
    */
    /*
      HIP currently doesn't generate the correct code for atomicInc's,
      so replace with an atomicAdd of 1 and assume no wraparound
    */
    //myRingBufferLoc = atomicInc(ringBufferTailPtr, maxRingBufferSize);
    myRingBufferLoc = atomicAdd(ringBufferTailPtr, 1);

    haveLock = false; // initially we don't have the lock
    backoff = 1;
  }
  __syncthreads();

  //  Two possibilities
  //    Mutex is unlocked
  //    Mutex is locked
  while (!haveLock)
  {
    __syncthreads();
    if (isMasterThread)
    {
      // spin waiting for our location in the ring buffer to == 1.
      if (atomicAdd(&ringBuffer[myRingBufferLoc], 0) == 1)
      {
        // atomicAdd (load) acts as a load acquire, need TF to enforce ordering
        __threadfence();

        // When our location in the ring buffer == 1, we have the lock
        haveLock = true;
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

  return myRingBufferLoc;
}

// to unlock, simply increment the ring buffer's head pointer.
__device__ void hipMutexSleepUnlock(const hipMutex_t mutex,
                                    int * mutexBuffers,
                                    unsigned int myBufferLoc,
                                    const int maxRingBufferSize,
                                    const int arrayStride,
                                    const int NUM_CU)
{
  __syncthreads();

  const bool isMasterThread = (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 &&
                               hipThreadIdx_z == 0);
  int * ringBuffer = (int * )mutexBuffers + (mutex * NUM_CU) * arrayStride;
  // next location is 0 if we're the last location in the buffer (wraparound)
  const unsigned int nextBufferLoc = ((myBufferLoc >= maxRingBufferSize) ? 0 :
                                      myBufferLoc + 1);

  if (isMasterThread)
  {
    // set my ring buffer location to -1
    atomicExch((int *)(ringBuffer + myBufferLoc), -1);

    // set the next location in the ring buffer to 1 so that next WG in line
    // can get the lock now
    atomicExch((int *)ringBuffer + nextBufferLoc, 1);

    // atomicExch acts as a store release, need TF to enforce ordering
    __threadfence();
  }
  __syncthreads();
}

// same algorithm but uses per-CU lock
__device__ unsigned int hipMutexSleepLockLocal(const hipMutex_t mutex,
                                               const unsigned int cuID,
                                               int * mutexBuffers,
                                               unsigned int * mutexBufferTails,
                                               const int maxRingBufferSize,
                                               const int arrayStride,
                                               const int NUM_CU)
{
  __syncthreads();

  // local variables
  const bool isMasterThread = (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 &&
                               hipThreadIdx_z == 0);
  unsigned int * const ringBufferTailPtr = mutexBufferTails + ((mutex * NUM_CU) +
                                                               cuID);
  int * const ringBuffer = (int * )mutexBuffers +
    ((mutex * NUM_CU) + cuID) * arrayStride;

  __shared__ unsigned int myRingBufferLoc;
  __shared__ bool haveLock;

  // this is a fire-and-forget atomic.
  if (isMasterThread)
  {
    /*
      HIP currently doesn't generate the correct code for atomicInc's here,
      so replace with an atomicAdd of 1 and assume no wraparound
    */
    //myRingBufferLoc = atomicInc(ringBufferTailPtr, maxRingBufferSize);
    myRingBufferLoc = atomicAdd(ringBufferTailPtr, 1);

    haveLock = false; // initially we don't have the lock
  }
  __syncthreads();

  //  Two possibilities
  //    Mutex is unlocked
  //    Mutex is locked
  while (!haveLock)
  {
    __syncthreads();
    if (isMasterThread)
    {
      // spin waiting for our location in the ring buffer to == 1.
      if (atomicAdd(&ringBuffer[myRingBufferLoc], 0) == 1)
      {
        // atomicAdd (load) acts as a load acquire, need TF to enforce ordering locally
        __threadfence_block();

        // When our location in the ring buffer == 1, we have the lock
        haveLock = true;
      }
    }
    __syncthreads();
  }

  return myRingBufferLoc;
}

// to unlock, simply increment the ring buffer's head pointer -- same algorithm
// but uses per-CU lock.
__device__ void hipMutexSleepUnlockLocal(const hipMutex_t mutex,
                                         const unsigned int cuID,
                                         int * mutexBuffers,
                                         unsigned int myBufferLoc,
                                         const int maxRingBufferSize,
                                         const int arrayStride,
                                         const int NUM_CU)
{
  __syncthreads();

  const bool isMasterThread = (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 &&
                               hipThreadIdx_z == 0);
  int * ringBuffer = (int * )mutexBuffers + ((mutex * NUM_CU) + cuID) *
                     arrayStride;
  // next location is 0 if we're the last location in the buffer (wraparound)
  const unsigned int nextBufferLoc = ((myBufferLoc >= maxRingBufferSize) ? 0 :
                                      myBufferLoc + 1);

  if (isMasterThread)
  {
    // set my ring buffer location to -1
    atomicExch((int *)(ringBuffer + myBufferLoc), -1);

    // set the next location in the ring buffer to 1 so that next WG in line
    // can get the lock now
    atomicExch((int *)ringBuffer + nextBufferLoc, 1);

    // atomicExch acts as a store release, need TF to enforce ordering locally
    __threadfence_block();
  }
  __syncthreads();
}

#endif
