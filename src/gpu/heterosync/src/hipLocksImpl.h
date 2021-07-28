#include "hipLocks.h"

hipError_t hipLocksInit(const int maxWGsPerKernel, const int numMutexes,
                        const int numSemaphores, const bool pageAlign,
                        const int NUM_CU, const int NUM_REPEATS,
                        const int NUM_ITERS)
{
  hipError_t hipErr = hipGetLastError();
  checkError(hipErr, "Start hipLocksInit");

  hipHostMalloc(&cpuLockData, sizeof(hipLockData_t));

  if (maxWGsPerKernel <= 0)       return hipErrorInitializationError;
  if (numMutexes <= 0)            return hipErrorInitializationError;
  if (numSemaphores <= 0)         return hipErrorInitializationError;

  // initialize some of the lock data's values
  /*
    Since HIP doesn't generate the correct code for atomicInc's, this
    means wraparound is not handled properly.  However, since in the current
    version each subsequent kernel launch starts in the ring buffer where
    the last kernel left off, this eventually leads to wraparound.  Increase
    buffer size to prevent wraparound and hide this.
  */
  cpuLockData->maxBufferSize          = maxWGsPerKernel * NUM_REPEATS * NUM_ITERS;
  cpuLockData->arrayStride            = (cpuLockData->maxBufferSize + NUM_CU) /
                                          NUM_WORDS_PER_CACHELINE * NUM_WORDS_PER_CACHELINE;
  cpuLockData->mutexCount             = numMutexes;
  cpuLockData->semaphoreCount         = numSemaphores;

  hipHostMalloc(&cpuLockData->barrierBuffers,   sizeof(unsigned int) * cpuLockData->arrayStride * 2);

  hipHostMalloc(&cpuLockData->mutexBuffers,     sizeof(int) * cpuLockData->arrayStride * cpuLockData->mutexCount);
  hipHostMalloc(&cpuLockData->mutexBufferHeads, sizeof(unsigned int) * cpuLockData->mutexCount);
  hipHostMalloc(&cpuLockData->mutexBufferTails, sizeof(unsigned int) * cpuLockData->mutexCount);

  hipHostMalloc(&cpuLockData->semaphoreBuffers, sizeof(unsigned int) * 4 * cpuLockData->semaphoreCount);

  hipErr = hipGetLastError();
  checkError(hipErr, "Before memsets");

  hipDeviceSynchronize();

  hipMemset(cpuLockData->barrierBuffers, 0,
            sizeof(unsigned int) * cpuLockData->arrayStride * 2);

  hipMemset(cpuLockData->mutexBufferHeads, 0,
            sizeof(unsigned int) * cpuLockData->mutexCount);
  hipMemset(cpuLockData->mutexBufferTails, 0,
            sizeof(unsigned int) * cpuLockData->mutexCount);

  /*
    initialize mutexBuffers to appropriate values

    set the first location for each CU to 1 so that the ring buffer can be
    used by the first WG right away (otherwise livelock because no locations
    ever == 1)

    for all other locations initialize to -1 so WGs for these locations
    don't think it's their turn right away

    since hipMemset sets everything in bytes, initialize all to 0 first
  */
  hipMemset(&(cpuLockData->mutexBuffers[0]), 0,
            cpuLockData->arrayStride * cpuLockData->mutexCount * sizeof(int));
  for (int i = 0; i < (cpuLockData->arrayStride * cpuLockData->mutexCount);
       i += cpuLockData->arrayStride) {
    hipMemset(&(cpuLockData->mutexBuffers[i]), 0x0001, 1);
    hipMemset(&(cpuLockData->mutexBuffers[i + 1]), -1,
              (cpuLockData->arrayStride - 1) * sizeof(int));
  }

  hipMemset(cpuLockData->semaphoreBuffers, 0,
            sizeof(unsigned int) * cpuLockData->semaphoreCount * 4);

  hipDeviceSynchronize();

  return hipSuccess;
}

hipError_t hipLocksDestroy()
{
  if (cpuLockData == NULL) { return hipErrorInitializationError; }
  hipHostFree(cpuLockData->mutexBuffers);
  hipHostFree(cpuLockData->mutexBufferHeads);
  hipHostFree(cpuLockData->mutexBufferTails);

  hipHostFree(cpuLockData->semaphoreBuffers);

  hipHostFree(cpuLockData);

  return hipSuccess;
}
