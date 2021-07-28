#ifndef __HIPSEMAPHOREEBO_H__
#define __HIPSEMAPHOREEBO_H__

#include "hip/hip_runtime.h"
#include "hipLocks.h"

inline __host__ hipError_t hipSemaphoreCreateEBO(hipSemaphore_t * const handle,
                                                 const int semaphoreNumber,
                                                 const unsigned int count,
                                                 const int NUM_CU)
{
  // Here we set the initial value to be count+1, this allows us to do an
  // atomicExch(sem, 0) and basically use the semaphore value as both a
  // lock and a semaphore.
  unsigned int initialValue = (count + 1), zero = 0;
  *handle = semaphoreNumber;
  for (int id = 0; id < NUM_CU; ++id) { // need to set these values for all CUs
    hipMemcpy(&(cpuLockData->semaphoreBuffers[((semaphoreNumber * 4 * NUM_CU) + (id * 4))]), &initialValue, sizeof(initialValue), hipMemcpyHostToDevice);
    hipMemcpy(&(cpuLockData->semaphoreBuffers[((semaphoreNumber * 4 * NUM_CU) + (id * 4)) + 1]), &zero, sizeof(zero), hipMemcpyHostToDevice);
    hipMemcpy(&(cpuLockData->semaphoreBuffers[((semaphoreNumber * 4 * NUM_CU) + (id * 4)) + 2]), &zero, sizeof(zero), hipMemcpyHostToDevice);
    hipMemcpy(&(cpuLockData->semaphoreBuffers[((semaphoreNumber * 4 * NUM_CU) + (id * 4)) + 3]), &initialValue, sizeof(initialValue), hipMemcpyHostToDevice);
  }
  return hipSuccess;
}

inline __device__ bool hipSemaphoreEBOTryWait(const hipSemaphore_t sem,
                                              const bool isWriter,
                                              const unsigned int maxSemCount,
                                              unsigned int * semaphoreBuffers,
                                              const int NUM_CU)
{
  const bool isMasterThread = (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 &&
                               hipThreadIdx_z == 0);
  /*
    Each sem has NUM_CU * 4 locations in the buffer.  Of these locations, each
    CU uses 4 of them (current count, head, tail, max count).  For the global
    semaphore all CUs use semaphoreBuffers[sem * 4 * NUM_CU].
  */
  unsigned int * const currCount = semaphoreBuffers + (sem * 4 * NUM_CU);
  unsigned int * const lock = currCount + 1;
  /*
    Reuse the tail for the "writers are waiting" flag since tail is unused.

    For now just use to indicate that at least 1 writer is waiting instead of
    a count to make sure that readers aren't totally starved out until all the
    writers are done.
  */
  unsigned int * const writerWaiting = currCount + 2;
  __shared__ bool acq1, acq2;

  __syncthreads();
  if (isMasterThread)
  {
    acq1 = false;
    // try to acquire the sem head "lock"
    if (atomicCAS(lock, 0, 1) == 0) {
      // atomicCAS acts as a load acquire, need TF to enforce ordering
      __threadfence();
      acq1 = true;
    }
  }
  __syncthreads();

  if (!acq1) { return false; } // return if we couldn't acquire the lock
  if (isMasterThread)
  {
    acq2 = false;
    /*
      NOTE: currCount is only accessed by 1 WG at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    unsigned int currSemCount = currCount[0];

    if (isWriter) {
      // writer needs the count to be == maxSemCount to enter the critical
      // section (otherwise there are readers in the critical section)
      if (currSemCount == maxSemCount) { acq2 = true; }
    } else {
      // if there is a writer waiting, readers aren't allowed to enter the
      // critical section
      if (writerWaiting[0] == 0) {
        // readers need count > 1 to enter critical section (otherwise semaphore
        // is full)
        if (currSemCount > 1) { acq2 = true; }
      }
    }
  }
  __syncthreads();

  if (!acq2) // release the sem head "lock" since the semaphore was full
  {
    // writers set a flag to note that they are waiting so more readers don't
    // join after the writer started waiting
    if (isWriter) { writerWaiting[0] = 1; /* if already 1, just reset to 1 */ }

    if (isMasterThread) {
      // atomicExch acts as a store release, need TF to enforce ordering
      __threadfence();
      atomicExch(lock, 0);
    }
    __syncthreads();
    return false;
  }
  __syncthreads();

  if (isMasterThread) {
    /*
      NOTE: currCount is only accessed by 1 WG at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    if (isWriter) {
      /*
        writer decrements the current count of the semaphore by the max to
        ensure that no one else can enter the critical section while it's
        writing.
      */
      currCount[0] -= maxSemCount;

      // writers also need to unset the "writer is waiting" flag
      writerWaiting[0] = 0;
    } else {
      /*
        readers decrement the current count of the semaphore by 1 so other
        readers can also read the data (but not the writers since they needs
        the entire CS).
      */
      --currCount[0]; //atomicSub(currCount, 1);
    }

    // atomicExch acts as a store release, need TF to enforce ordering
    __threadfence();
    // now that we've updated the semaphore count can release the lock
    atomicExch(lock, 0);
  }
  __syncthreads();

  return true;
}

inline __device__ void hipSemaphoreEBOWait(const hipSemaphore_t sem,
                                           const bool isWriter,
                                           const unsigned int maxSemCount,
                                           unsigned int * semaphoreBuffers,
                                           const int NUM_CU)
{
  __shared__ int backoff;
  const bool isMasterThread = (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 &&
                               hipThreadIdx_z == 0);

  if (isMasterThread)
  {
    backoff = 1;
  }
  __syncthreads();

  while (!hipSemaphoreEBOTryWait(sem, isWriter, maxSemCount, semaphoreBuffers, NUM_CU))
  {
    __syncthreads();
    if (isMasterThread)
    {
      // if we failed to enter the semaphore, wait for a little while before
      // trying again
      sleepFunc(backoff);
      /*
        for writers increse backoff a lot because failing means readers are in
        the CS currently -- most important for non-unique because all WGs on
        all CUs are going for the same semaphore.
      */
      if (isWriter) {
        // (capped) exponential backoff
        backoff = (((backoff << 1) + 1) & (MAX_BACKOFF-1));
      }
      else { backoff += 5; /* small, linear backoff increase for readers */ }
    }
    __syncthreads();
  }
  __syncthreads();
}

inline __device__ void hipSemaphoreEBOPost(const hipSemaphore_t sem,
                                           const bool isWriter,
                                           const unsigned int maxSemCount,
                                           unsigned int * semaphoreBuffers,
                                           const int NUM_CU)
{
  const bool isMasterThread = (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 &&
                               hipThreadIdx_z == 0);
  /*
    Each sem has NUM_CU * 4 locations in the buffer.  Of these locations, each
    CU uses 4 of them (current count, head, tail, max count).  For the global
    semaphore use semaphoreBuffers[sem * 4 * NUM_CU].
  */
  unsigned int * const currCount = semaphoreBuffers + (sem * 4 * NUM_CU);
  unsigned int * const lock = currCount + 1;
  __shared__ bool acquired;

  if (isMasterThread) { acquired = false; }
  __syncthreads();

  while (!acquired)
  {
    __syncthreads();
    if (isMasterThread)
    {
      // try to acquire sem head "lock"
      if (atomicCAS(lock, 0, 1) == 0) {
        // atomicCAS acts as a load acquire, need TF to enforce ordering
        __threadfence();
        acquired = true;
      } else {
        acquired = false;
      }
    }
    __syncthreads();
  }

  if (isMasterThread) {
    /*
      NOTE: currCount is only accessed by 1 WG at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    if (isWriter) {
      // writers add the max value to the semaphore to allow the readers to
      // start accessing the critical section.
      currCount[0] += maxSemCount;
    } else {
      ++currCount[0]; // readers add 1 to the semaphore
    }

    // atomicExch acts as a store release, need TF to enforce ordering
    __threadfence();
    // now that we've updated the semaphore count can release the lock
    atomicExch(lock, 0);
  }
  __syncthreads();
}

// same wait algorithm but with local scope and per-CU synchronization
inline __device__ bool hipSemaphoreEBOTryWaitLocal(const hipSemaphore_t sem,
                                                   const unsigned int cuID,
                                                   const bool isWriter,
                                                   const unsigned int maxSemCount,
                                                   unsigned int * semaphoreBuffers,
                                                   const int NUM_CU)
{
  const bool isMasterThread = (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 &&
                               hipThreadIdx_z == 0);
  /*
    Each sem has NUM_CU * 4 locations in the buffer.  Of these locations, each
    CU gets 4 of them (current count, head, tail, max count).  So CU 0 starts
    at semaphoreBuffers[sem * 4 * NUM_CU].
  */
  unsigned int * const currCount = semaphoreBuffers + ((sem * 4 * NUM_CU) +
                                                       (cuID * 4));
  unsigned int * const lock = currCount + 1;
  /*
    Reuse the tail for the "writers are waiting" flag since tail is unused.

    For now just use to indicate that at least 1 writer is waiting instead of
    a count to make sure that readers aren't totally starved out until all the
    writers are done.
  */
  unsigned int * const writerWaiting = currCount + 2;
  __shared__ bool acq1, acq2;

  __syncthreads();
  if (isMasterThread)
  {
    acq1 = false;
    // try to acquire the sem head "lock"
    if (atomicCAS(lock, 0, 1) == 0) {
      // atomicCAS acts as a load acquire, need TF to enforce ordering locally
      __threadfence_block();
      acq1 = true;
    }
  }
  __syncthreads();

  if (!acq1) { return false; } // return if we couldn't acquire the lock
  if (isMasterThread)
  {
    acq2 = false;
    /*
      NOTE: currCount is only accessed by 1 WG at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    unsigned int currSemCount = currCount[0];

    if (isWriter) {
      // writer needs the count to be == maxSemCount to enter the critical
      // section (otherwise there are readers in the critical section)
      if (currSemCount == maxSemCount) { acq2 = true; }
    } else {
      // if there is a writer waiting, readers aren't allowed to enter the
      // critical section
      if (writerWaiting[0] == 0) {
        // readers need count > 1 to enter critical section (otherwise semaphore
        // is full)
        if (currSemCount > 1) { acq2 = true; }
      }
    }
  }
  __syncthreads();

  if (!acq2) // release the sem head "lock" since the semaphore was full
  {
    // writers set a flag to note that they are waiting so more readers don't
    // join after the writer started waiting
    if (isWriter) { writerWaiting[0] = 1; /* if already 1, just reset to 1 */ }

    if (isMasterThread) {
      // atomicExch acts as a store release, need TF to enforce ordering locally
      __threadfence_block();
      atomicExch(lock, 0);
    }
    __syncthreads();
    return false;
  }
  __syncthreads();

  if (isMasterThread) {
    /*
      NOTE: currCount is only accessed by 1 WG at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    if (isWriter) {
      /*
        writer decrements the current count of the semaphore by the max to
        ensure that no one else can enter the critical section while it's
        writing.
      */
      currCount[0] -= maxSemCount;

      // writers also need to unset the "writer is waiting" flag
      writerWaiting[0] = 0;
    } else {
      /*
        readers decrement the current count of the semaphore by 1 so other
        readers can also read the data (but not the writers since they needs
        the entire CS).
      */
      --currCount[0];
    }

    // atomicExch acts as a store release, need TF to enforce ordering locally
    __threadfence_block();
    // now that we've updated the semaphore count can release the lock
    atomicExch(lock, 0);
  }
  __syncthreads();

  return true;
}

// same algorithm but with local scope
inline __device__ void hipSemaphoreEBOWaitLocal(const hipSemaphore_t sem,
                                                const unsigned int cuID,
                                                const bool isWriter,
                                                const unsigned int maxSemCount,
                                                unsigned int * semaphoreBuffers,
                                                const int NUM_CU)
{
  __shared__ int backoff;
  const bool isMasterThread = (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 &&
                               hipThreadIdx_z == 0);

  if (isMasterThread)
  {
    backoff = 1;
  }
  __syncthreads();

  while (!hipSemaphoreEBOTryWaitLocal(sem, cuID, isWriter, maxSemCount, semaphoreBuffers, NUM_CU))
  {
    __syncthreads();
    if (isMasterThread)
    {
      /*
        if we failed to enter the semaphore, wait for a little while before
        trying again
      */
      sleepFunc(backoff);
      // (capped) exponential backoff
      backoff = (((backoff << 1) + 1) & (MAX_BACKOFF-1));
    }
    __syncthreads();
  }
  __syncthreads();
}

inline __device__ void hipSemaphoreEBOPostLocal(const hipSemaphore_t sem,
                                                const unsigned int cuID,
                                                const bool isWriter,
                                                const unsigned int maxSemCount,
                                                unsigned int * semaphoreBuffers,
                                                const int NUM_CU)
{
  const bool isMasterThread = (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 &&
                               hipThreadIdx_z == 0);
  // Each sem has NUM_CU * 4 locations in the buffer.  Of these locations, each
  // CU gets 4 of them.  So CU 0 starts at semaphoreBuffers[sem * 4 * NUM_CU].
  unsigned int * const currCount = semaphoreBuffers + ((sem * 4 * NUM_CU) +
                                                       (cuID * 4));
  unsigned int * const lock = currCount + 1;
  __shared__ bool acquired;

  if (isMasterThread) { acquired = false; }
  __syncthreads();

  while (!acquired)
  {
    __syncthreads();
    if (isMasterThread)
    {
      // try to acquire sem head "lock"
      if (atomicCAS(lock, 0, 1) == 0) {
        // atomicCAS acts as a load acquire, need TF to enforce ordering locally
        __threadfence_block();
        acquired = true;
      } else {
        acquired = false;
      }
    }
    __syncthreads();
  }

  if (isMasterThread) {
    /*
      NOTE: currCount is only accessed by 1 WGs at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    if (isWriter) {
      // writers add the max value to the semaphore to allow the readers to
      // start accessing the critical section.
      currCount[0] += maxSemCount;
    } else {
      ++currCount[0]; // readers add 1 to the semaphore
    }

    // atomicExch acts as a store release, need TF to enforce ordering locally
    __threadfence_block();
    // now that we've updated the semaphore count can release the lock
    atomicExch(lock, 0);
  }
  __syncthreads();
}

#endif // #ifndef __HIPSEMAPHOREEBO_H__
