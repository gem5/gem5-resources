#include <cstdio>
#include <string>
#include <assert.h>
#include <math.h>
#include "hip/hip_runtime.h"
#include "hip_error.h"

#define MAX_BACKOFF             1024
#define NUM_WIS_PER_WG 64
#define MAD_MUL 1.1f
#define MAD_ADD 0.25f
#define NUM_WORDS_PER_CACHELINE 16
#define NUM_WIS_PER_QUARTERWAVE 16

// separate .h files
#include "hipLocksBarrier.h"
#include "hipLocksImpl.h"
#include "hipLocksMutex.h"
#include "hipLocksSemaphore.h"
// program globals
const int NUM_REPEATS = 10;
int NUM_LDST = 0;
int numWGs = 0;
// number of CUs our GPU has
int NUM_CU = 0;
int MAX_WGS = 0;

bool pageAlign = false;

/*
  Helper function to do data accesses for golden checking code.
*/
void accessData_golden(float * storageGolden, int currLoc, int numStorageLocs)
{
  /*
    If this location isn't the first location accessed by a
    thread, update it -- each half-warp accesses (NUM_LDST + 1) cache
    lines.
  */
  if (currLoc % (NUM_WIS_PER_QUARTERWAVE * (NUM_LDST + 1)) >=
      NUM_WORDS_PER_CACHELINE)
  {
    assert((currLoc - NUM_WORDS_PER_CACHELINE) >= 0);
    assert(currLoc < numStorageLocs);
    // each location is dependent on the location accessed at the
    // same word on the previous cache line
    storageGolden[currLoc] =
      ((storageGolden[currLoc -
                      NUM_WORDS_PER_CACHELINE]/* * MAD_MUL*/) /*+ MAD_ADD*/);
  }
}

/*
  Shared function that does the critical section data accesses for the barrier
  and mutex kernels.

  NOTE: kernels that access data differently (e.g., semaphores) should not call
  this function.
*/
inline __device__ void accessData(float * storage, int threadBaseLoc,
                                  int threadOffset, int NUM_LDST)
{
  // local variables
  int readLoc = 0, writeLoc = 0;

  for (int n = NUM_LDST-1; n >= 0; --n) {
    writeLoc = ((threadBaseLoc + n + 1) * NUM_WORDS_PER_CACHELINE) +
               threadOffset;
    readLoc = ((threadBaseLoc + n) * NUM_WORDS_PER_CACHELINE) + threadOffset;
    storage[writeLoc] = ((storage[readLoc]/* * MAD_MUL*/) /*+ MAD_ADD*/);
  }
}

/*
  Helper function for semaphore writers.  Although semaphore writers are
  iterating over all locations accessed on a given CU, the logic for this
  varies and is done outside this helper, so can just call accessData().
*/
inline __device__ void accessData_semWr(float * storage, int threadBaseLoc,
                                        int threadOffset, int NUM_LDST)
{
  accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);
}


/*
  Helper function for semaphore readers.
*/
inline __device__ void accessData_semRd(float * storage,
                                        volatile float * dummyArray,
                                        int threadBaseLoc,
                                        int threadOffset, int NUM_LDST)
{
  for (int n = NUM_LDST-1; n >= 0; --n) {
    dummyArray[hipThreadIdx_x] +=
      storage[((threadBaseLoc + n) * NUM_WORDS_PER_CACHELINE) +
              threadOffset];
    __syncthreads();
  }
}

// performs a tree barrier.  Each WG on an CU accesses unique data then joins a
// local barrier.  1 of the WGs from each CU then joins the global barrier
__global__ void kernelAtomicTreeBarrierUniq(float * storage,
                                            hipLockData_t * gpuLockData,
                                            unsigned int * perCUBarrierBuffers,
                                            const int ITERATIONS,
                                            const int NUM_LDST,
                                            const int NUM_CU,
                                            const int MAX_WGS)
{
  // local variables
  // thread 0 is master thread
  const bool isMasterThread = ((hipThreadIdx_x == 0) && (hipThreadIdx_y == 0) &&
                               (hipThreadIdx_z == 0));
  // represents the number of WGs going to the barrier (max NUM_CU, hipGridDim_x if
  // fewer WGs than CUs).
  const unsigned int numWGsAtBarr = ((hipGridDim_x < NUM_CU) ? hipGridDim_x :
                                        NUM_CU);
  const int cuID = (hipBlockIdx_x % numWGsAtBarr); // mod by # CUs to get CU ID
  // all work groups on the same CU access unique locations because the
  // barrier can't ensure DRF between WGs
  int currWGID = hipBlockIdx_x;
  int tid = ((currWGID * hipBlockDim_x) + hipThreadIdx_x);
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  int threadOffset = (hipThreadIdx_x % NUM_WORDS_PER_CACHELINE);
  // determine if I'm WG 0 on my CU
  const int perCU_WGID = (hipBlockIdx_x / numWGsAtBarr);
  // given the hipGridDim_x, we can figure out how many WGs are on our CU -- assume
  // all CUs have an identical number of WGs
  int numWGs_perCU = (int)ceil((float)hipGridDim_x / numWGsAtBarr);

  for (int i = 0; i < ITERATIONS; ++i)
  {
    accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);

    joinBarrier_helper(gpuLockData->barrierBuffers, perCUBarrierBuffers,
                       numWGsAtBarr, cuID, perCU_WGID, numWGs_perCU,
                       isMasterThread, MAX_WGS);

    // get new thread ID by trading amongst WGs -- + 1 WG ID to shift to next
    // CUs data
    currWGID = ((currWGID + 1) % hipGridDim_x);
    tid = ((currWGID * hipBlockDim_x) + hipThreadIdx_x);
    threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  }
}

// like the tree barrier but also has WGs exchange work locally before joining
// the global barrier
__global__ void kernelAtomicTreeBarrierUniqLocalExch(float * storage,
                                                     hipLockData_t * gpuLockData,
                                                     unsigned int * perCUBarrierBuffers,
                                                     const int ITERATIONS,
                                                     const int NUM_LDST,
                                                     const int NUM_CU,
                                                     const int MAX_WGS)
{
  // local variables
  // thread 0 is master thread
  const bool isMasterThread = ((hipThreadIdx_x == 0) && (hipThreadIdx_y == 0) &&
                               (hipThreadIdx_z == 0));
  // represents the number of WGs going to the barrier (max NUM_CU, hipGridDim_x if
  // fewer WGs than CUs).
  const unsigned int numWGsAtBarr = ((hipGridDim_x < NUM_CU) ? hipGridDim_x :
                                        NUM_CU);
  const int cuID = (hipBlockIdx_x % numWGsAtBarr); // mod by # CUs to get CU ID
  // all work groups on the same CU access unique locations because the
  // barrier can't ensure DRF between WGs
  int currWGID = hipBlockIdx_x;
  int tid = ((currWGID * hipBlockDim_x) + hipThreadIdx_x);
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  int threadOffset = (hipThreadIdx_x % NUM_WORDS_PER_CACHELINE);
  // determine if I'm WG 0 on my CU
  const int perCU_WGID = (hipBlockIdx_x / numWGsAtBarr);
  // given the hipGridDim_x, we can figure out how many WGs are on our CU -- assume
  // all CUs have an identical number of WGs
  int numWGs_perCU = (int)ceil((float)hipGridDim_x / numWGsAtBarr);

  for (int i = 0; i < ITERATIONS; ++i)
  {
    accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);

    // all WGs on this CU do a local barrier (if > 1 WG)
    if (numWGs_perCU > 1) {
      hipBarrierAtomicLocal(perCUBarrierBuffers, cuID, numWGs_perCU,
                             isMasterThread, MAX_WGS);

      // exchange data within the WGs on this CU, then do some more computations
      currWGID = ((currWGID + numWGsAtBarr) % hipGridDim_x);
      tid = ((currWGID * hipBlockDim_x) + hipThreadIdx_x);
      threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));

      accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);
    }

    joinBarrier_helper(gpuLockData->barrierBuffers, perCUBarrierBuffers,
                       numWGsAtBarr, cuID, perCU_WGID, numWGs_perCU,
                       isMasterThread, MAX_WGS);

    // get new thread ID by trading amongst WGs -- + 1 WG ID to shift to next
    // CUs data
    currWGID = ((currWGID + 1) % hipGridDim_x);
    tid = ((currWGID * hipBlockDim_x) + hipThreadIdx_x);
    threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  }
}

// performs a tree barrier like above but with a lock-free barrier
__global__ void kernelFBSTreeBarrierUniq(float * storage,
                                         hipLockData_t * gpuLockData,
                                         unsigned int * perCUBarrierBuffers,
                                         const int ITERATIONS,
                                         const int NUM_LDST,
                                         const int NUM_CU,
                                         const int MAX_WGS)
{
  // local variables
  // represents the number of WGs going to the barrier (max NUM_CU, hipGridDim_x if
  // fewer WGs than CUs).
  const unsigned int numWGsAtBarr = ((hipGridDim_x < NUM_CU) ? hipGridDim_x :
                                        NUM_CU);
  const int cuID = (hipBlockIdx_x % numWGsAtBarr); // mod by # CUs to get CU ID
  // all work groups on the same CU access unique locations because the
  // barrier can't ensure DRF between WGs
  int currWGID = hipBlockIdx_x;
  int tid = ((currWGID * hipBlockDim_x) + hipThreadIdx_x);
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (hipThreadIdx_x % NUM_WORDS_PER_CACHELINE);
  // determine if I'm WG 0 on my CU
  const int perCU_WGID = (hipBlockIdx_x / numWGsAtBarr);
  // given the hipGridDim_x, we can figure out how many WGs are on our CU -- assume
  // all CUs have an identical number of WGs
  int numWGs_perCU = (int)ceil((float)hipGridDim_x/numWGsAtBarr);

  for (int i = 0; i < ITERATIONS; ++i)
  {
    accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);

    joinLFBarrier_helper(gpuLockData->barrierBuffers, perCUBarrierBuffers,
                         numWGsAtBarr, cuID, perCU_WGID, numWGs_perCU,
                         gpuLockData->arrayStride, MAX_WGS);

    // get new thread ID by trading amongst WGs -- + 1 WG ID to shift to next
    // CUs data
    currWGID = ((currWGID + 1) % hipGridDim_x);
    tid = ((currWGID * hipBlockDim_x) + hipThreadIdx_x);
    threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  }
}

// performs a tree barrier like above but with a lock-free barrier and has WGs
// exchange work locally before joining the global barrier
__global__ void kernelFBSTreeBarrierUniqLocalExch(float * storage,
                                                  hipLockData_t * gpuLockData,
                                                  unsigned int * perCUBarrierBuffers,
                                                  const int ITERATIONS,
                                                  const int NUM_LDST,
                                                  const int NUM_CU,
                                                  const int MAX_WGS)
{
  // local variables
  // represents the number of WGs going to the barrier (max NUM_CU, hipGridDim_x if
  // fewer WGs than CUs).
  const unsigned int numWGsAtBarr = ((hipGridDim_x < NUM_CU) ? hipGridDim_x : NUM_CU);
  const int cuID = (hipBlockIdx_x % numWGsAtBarr); // mod by # CUs to get CU ID
  // all work groups on the same CU access unique locations because the
  // barrier can't ensure DRF between WGs
  int currWGID = hipBlockIdx_x;
  int tid = ((currWGID * hipBlockDim_x) + hipThreadIdx_x);
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (hipThreadIdx_x % NUM_WORDS_PER_CACHELINE);
  // determine if I'm WG 0 on my CU
  const int perCU_WGID = (hipBlockIdx_x / numWGsAtBarr);
  // given the hipGridDim_x, we can figure out how many WGs are on our CU -- assume
  // all CUs have an identical number of WGs
  int numWGs_perCU = (int)ceil((float)hipGridDim_x/numWGsAtBarr);

  for (int i = 0; i < ITERATIONS; ++i)
  {
    accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);

    // all WGs on this CU do a local barrier (if > 1 WG per CU)
    if (numWGs_perCU > 1) {
      hipBarrierLocal(gpuLockData->barrierBuffers, numWGsAtBarr,
                       gpuLockData->arrayStride, perCUBarrierBuffers, cuID, numWGs_perCU,
                       perCU_WGID, false, MAX_WGS);

      // exchange data within the WGs on this CU and do some more computations
      currWGID = ((currWGID + numWGsAtBarr) % hipGridDim_x);
      tid = ((currWGID * hipBlockDim_x) + hipThreadIdx_x);
      threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));

      accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);
    }

    joinLFBarrier_helper(gpuLockData->barrierBuffers, perCUBarrierBuffers,
                         numWGsAtBarr, cuID, perCU_WGID, numWGs_perCU,
                         gpuLockData->arrayStride, MAX_WGS);

    // get new thread ID by trading amongst WGs -- + 1 WG ID to shift to next
    // CUs data
    currWGID = ((currWGID + 1) % hipGridDim_x);
    tid = ((currWGID * hipBlockDim_x) + hipThreadIdx_x);
    threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  }
}

__global__ void kernelSleepingMutex(hipMutex_t mutex, float * storage,
                                    hipLockData_t * gpuLockData,
                                    const int ITERATIONS, const int NUM_LDST,
                                    const int NUM_CU)
{
  // local variables
  // all work groups access the same locations (rely on release to get
  // ownership in time)
  const int tid = hipThreadIdx_x;
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  const int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (hipThreadIdx_x % NUM_WORDS_PER_CACHELINE);
  __shared__ int myRingBufferLoc; // tracks my WGs location in the ring buffer

  if (hipThreadIdx_x == 0) {
    myRingBufferLoc = -1; // initially I have no location
  }
  __syncthreads();

  for (int i = 0; i < ITERATIONS; ++i)
  {
    myRingBufferLoc = hipMutexSleepLock(mutex, gpuLockData->mutexBuffers,
                                        gpuLockData->mutexBufferTails, gpuLockData->maxBufferSize,
                                        gpuLockData->arrayStride, NUM_CU);

    accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);

    hipMutexSleepUnlock(mutex, gpuLockData->mutexBuffers, myRingBufferLoc,
                         gpuLockData->maxBufferSize, gpuLockData->arrayStride, NUM_CU);
  }
}

__global__ void kernelSleepingMutexUniq(hipMutex_t mutex, float * storage,
                                        hipLockData_t * gpuLockData,
                                        const int ITERATIONS,
                                        const int NUM_LDST, const int NUM_CU)
{
  // local variables
  const int cuID = (hipBlockIdx_x % NUM_CU); // mod by # CUs to get CU ID
  // all work groups on the same CU access the same locations
  const int tid = ((cuID * hipBlockDim_x) + hipThreadIdx_x);
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  const int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (hipThreadIdx_x % NUM_WORDS_PER_CACHELINE);
  __shared__ int myRingBufferLoc; // tracks my WGs location in the ring buffer

  if (hipThreadIdx_x == 0) {
    myRingBufferLoc = -1; // initially I have no location
  }
  __syncthreads();

  for (int i = 0; i < ITERATIONS; ++i)
  {
    myRingBufferLoc = hipMutexSleepLockLocal(mutex, cuID, gpuLockData->mutexBuffers,
                                              gpuLockData->mutexBufferTails, gpuLockData->maxBufferSize,
                                              gpuLockData->arrayStride, NUM_CU);

    accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);

    hipMutexSleepUnlockLocal(mutex, cuID, gpuLockData->mutexBuffers, myRingBufferLoc,
                              gpuLockData->maxBufferSize, gpuLockData->arrayStride,
                              NUM_CU);
  }
}

__global__ void kernelFetchAndAddMutex(hipMutex_t mutex, float * storage,
                                       hipLockData_t * gpuLockData,
                                       const int ITERATIONS,
                                       const int NUM_LDST, const int NUM_CU)
{
  // local variables
  // all work groups access the same locations (rely on release to get
  // ownership in time)
  const int tid = hipThreadIdx_x;
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  const int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (hipThreadIdx_x % NUM_WORDS_PER_CACHELINE);

  for (int i = 0; i < ITERATIONS; ++i)
  {
    hipMutexFALock(mutex, gpuLockData->mutexBufferHeads, gpuLockData->mutexBufferTails,
                    NUM_CU);

    accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);

    hipMutexFAUnlock(mutex, gpuLockData->mutexBufferTails, NUM_CU);
  }
}

__global__ void kernelFetchAndAddMutexUniq(hipMutex_t mutex, float * storage,
                                           hipLockData_t * gpuLockData,
                                           const int ITERATIONS,
                                           const int NUM_LDST,
                                           const int NUM_CU)
{
  // local variables
  const int cuID = (hipBlockIdx_x % NUM_CU); // mod by # CUs to get CU ID
  // all work groups on the same CU access the same locations
  const int tid = ((cuID * hipBlockDim_x) + hipThreadIdx_x);
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  const int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (hipThreadIdx_x % NUM_WORDS_PER_CACHELINE);

  for (int i = 0; i < ITERATIONS; ++i)
  {
    hipMutexFALockLocal(mutex, cuID, gpuLockData->mutexBufferHeads,
                         gpuLockData->mutexBufferTails, NUM_CU);

    accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);

    hipMutexFAUnlockLocal(mutex, cuID, gpuLockData->mutexBufferTails, NUM_CU);
  }
}

__global__ void kernelSpinLockMutex(hipMutex_t mutex, float * storage,
                                    hipLockData_t * gpuLockData,
                                    const int ITERATIONS, const int NUM_LDST,
                                    const int NUM_CU)
{
  // local variables
  // all work groups access the same locations (rely on release to get
  // ownership in time)
  const int tid = hipThreadIdx_x;
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  const int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (hipThreadIdx_x % NUM_WORDS_PER_CACHELINE);

  for (int i = 0; i < ITERATIONS; ++i)
  {
    hipMutexSpinLock(mutex, gpuLockData->mutexBufferHeads, NUM_CU);

    accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);

    hipMutexSpinUnlock(mutex, gpuLockData->mutexBufferHeads, NUM_CU);
  }
}

__global__ void kernelSpinLockMutexUniq(hipMutex_t mutex, float * storage,
                                        hipLockData_t * gpuLockData,
                                        const int ITERATIONS,
                                        const int NUM_LDST, const int NUM_CU)
{
  // local variables
  const int cuID = (hipBlockIdx_x % NUM_CU); // mod by # CUs to get CU ID
  // all work groups on the same CU access the same locations
  const int tid = ((cuID * hipBlockDim_x) + hipThreadIdx_x);
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  const int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (hipThreadIdx_x % NUM_WORDS_PER_CACHELINE);

  for (int i = 0; i < ITERATIONS; ++i)
  {
    hipMutexSpinLockLocal(mutex, cuID, gpuLockData->mutexBufferHeads, NUM_CU);

    accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);

    hipMutexSpinUnlockLocal(mutex, cuID, gpuLockData->mutexBufferHeads, NUM_CU);
  }
}

__global__ void kernelEBOMutex(hipMutex_t mutex, float * storage,
                               hipLockData_t * gpuLockData,
                               const int ITERATIONS, const int NUM_LDST,
                               const int NUM_CU)
{
  // local variables
  // all work groups access the same locations (rely on release to get
  // ownership in time)
  const int tid = hipThreadIdx_x;
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  const int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (hipThreadIdx_x % NUM_WORDS_PER_CACHELINE);

  for (int i = 0; i < ITERATIONS; ++i)
  {
    hipMutexEBOLock(mutex, gpuLockData->mutexBufferHeads, NUM_CU);

    accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);

    hipMutexEBOUnlock(mutex, gpuLockData->mutexBufferHeads, NUM_CU);
  }
}

__global__ void kernelEBOMutexUniq(hipMutex_t mutex, float * storage,
                                   hipLockData_t * gpuLockData,
                                   const int ITERATIONS, const int NUM_LDST,
                                   const int NUM_CU)
{
  // local variables
  const int cuID = (hipBlockIdx_x % NUM_CU); // mod by # CUs to get CU ID
  // all work groups on the same CU access the same locations
  const int tid = ((cuID * hipBlockDim_x) + hipThreadIdx_x);
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  const int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (hipThreadIdx_x % NUM_WORDS_PER_CACHELINE);

  for (int i = 0; i < ITERATIONS; ++i)
  {
    hipMutexEBOLockLocal(mutex, cuID, gpuLockData->mutexBufferHeads, NUM_CU);

    accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);

    hipMutexEBOUnlockLocal(mutex, cuID, gpuLockData->mutexBufferHeads, NUM_CU);
  }
}

// All WGs on all CUs access the same data with 1 writer per CU (and N-1)
// readers per CU.
__global__ void kernelSpinLockSemaphore(hipSemaphore_t sem,
                                        float * storage,
                                        hipLockData_t * gpuLockData,
                                        const unsigned int numStorageLocs,
                                        const int ITERATIONS,
                                        const int NUM_LDST,
                                        const int NUM_CU)
{
  // local variables
  const unsigned int maxSemCount =
    gpuLockData->semaphoreBuffers[((sem * 4 * NUM_CU) + (0 * 4))];
  const int cuID = (hipBlockIdx_x % NUM_CU); // mod by # CUs to get CU ID
  // If there are fewer WGs than # CUs, need to take into account for various
  // math below.  If WGs >= NUM_CU, use NUM_CU.
  const unsigned int numCU = ((hipGridDim_x < NUM_CU) ? hipGridDim_x : NUM_CU);
  // given the hipGridDim_x, we can figure out how many WGs are on our CU -- assume
  // all CUs have an identical number of WGs
  int numWGs_perCU = (int)ceil((float)hipGridDim_x / numCU);
  // number of threads on each WG
  //const int numThrs_perCU = (hipBlockDim_x * numWGs_perCU);
  const int perCU_WGID = (hipBlockIdx_x / numCU);
  // rotate which WG is the writer
  const bool isWriter = (perCU_WGID == (cuID % numWGs_perCU));

  // all work groups on the same CU access unique locations except the writer,
  // which writes all of the locations that all of the WGs access
  //int currWGID = hipBlockIdx_x;
  // the (reader) WGs on each CU access unique locations but those same
  // locations are accessed by the reader WGs on all CUs
  int tid = ((perCU_WGID * hipBlockDim_x) + hipThreadIdx_x);
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (hipThreadIdx_x % NUM_WORDS_PER_CACHELINE);

  // dummy array to hold the loads done in the readers
  __shared__ volatile float dummyArray[NUM_WIS_PER_WG];

  for (int i = 0; i < ITERATIONS; ++i)
  {
    /*
      NOTE: There is a race here for entering the critical section.  Most
      importantly, it means that the at least one of the readers could win and
      thus the readers will read before the writer has had a chance to write
      the data.
    */
    hipSemaphoreSpinWait(sem, isWriter, maxSemCount,
                          gpuLockData->semaphoreBuffers, NUM_CU);

    // writer writes all the data that the WGs on this CU access
    if (isWriter) {
      for (int j = 0; j < numWGs_perCU; ++j) {
        /*
          Update the writer's "location" so it writes to the locations that the
          readers will access (due to RR scheduling the next WG on this CU is
          numCU WGs away).  Use loop counter because the non-unique version
          writes the same locations on all CUs.
        */
        tid = ((j * hipBlockDim_x) + hipThreadIdx_x);
        threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));

        accessData_semWr(storage, threadBaseLoc, threadOffset, NUM_LDST);
      }
      // reset locations
      tid = ((perCU_WGID * hipBlockDim_x) + hipThreadIdx_x);
      threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
    }
    // rest of WGs on this CU read the data written by each CU's writer WG
    else {
      accessData_semRd(storage, dummyArray, threadBaseLoc, threadOffset,
                       NUM_LDST);
    }
    hipSemaphoreSpinPost(sem, isWriter, maxSemCount,
                          gpuLockData->semaphoreBuffers, NUM_CU);
  }
}

__global__ void kernelSpinLockSemaphoreUniq(hipSemaphore_t sem,
                                            float * storage,
                                            hipLockData_t * gpuLockData,
                                            const int ITERATIONS,
                                            const int NUM_LDST,
                                            const int NUM_CU)
{
  // local variables
  const unsigned int maxSemCount =
      gpuLockData->semaphoreBuffers[((sem * 4 * NUM_CU) + (0 * 4))];
  // If there are fewer WGs than # CUs, need to take into account for various
  // math below.  If WGs >= NUM_CU, use NUM_CU.
  const unsigned int numCU = ((hipGridDim_x < NUM_CU) ? hipGridDim_x : NUM_CU);
  const int cuID = (hipBlockIdx_x % numCU); // mod by # CUs to get CU ID
  // given the hipGridDim_x, we can figure out how many WGs are on our CU -- assume
  // all CUs have an identical number of WGs
  int numWGs_perCU = (int)ceil((float)hipGridDim_x / numCU);
  const int perCU_WGID = (hipBlockIdx_x / numCU);
  // rotate which WG is the writer
  const bool isWriter = (perCU_WGID == (cuID % numWGs_perCU));

  // all work groups on the same CU access unique locations except the writer,
  // which writes all of the locations that all of the WGs access
  int currWGID = hipBlockIdx_x;
  int tid = ((currWGID * hipBlockDim_x) + hipThreadIdx_x);
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (hipThreadIdx_x % NUM_WORDS_PER_CACHELINE);
  // dummy array to hold the loads done in the readers
  __shared__ volatile float dummyArray[NUM_WIS_PER_WG];

  for (int i = 0; i < ITERATIONS; ++i)
  {
    /*
      NOTE: There is a race here for entering the critical section.  Most
      importantly, it means that the at least one of the readers could win and
      thus the readers will read before the writer has had a chance to write
      the data.
    */
    hipSemaphoreSpinWaitLocal(sem, cuID, isWriter, maxSemCount,
                               gpuLockData->semaphoreBuffers, NUM_CU);

    // writer WG writes all the data that the WGs on this CU access
    if (isWriter) {
      for (int j = 0; j < numWGs_perCU; ++j) {
        accessData_semWr(storage, threadBaseLoc, threadOffset, NUM_LDST);

        /*
          update the writer's "location" so it writes to the locations that the
          readers will access (due to RR scheduling the next WG on this CU is
          numCU WGs away and < hipGridDim_x).

          NOTE: First location writer writes to is its own location(s).  If the
          writer is not CU 0 on this CU, it may require wrapping around to CUs
          with smaller WG IDs.
        */
        currWGID = (currWGID + numCU) % hipGridDim_x;
        tid = ((currWGID * hipBlockDim_x) + hipThreadIdx_x);
        threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
      }
      // reset locations
      currWGID = hipBlockIdx_x;
      tid = ((currWGID * hipBlockDim_x) + hipThreadIdx_x);
      threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
    }
    // rest of WGs on this CU read the data written by each CU's writer WG
    else {
      accessData_semRd(storage, dummyArray, threadBaseLoc, threadOffset,
                       NUM_LDST);
    }
    hipSemaphoreSpinPostLocal(sem, cuID, isWriter, maxSemCount,
                               gpuLockData->semaphoreBuffers, NUM_CU);
  }
}

// All WGs on all CUs access the same data with 1 writer per CU (and N-1)
// readers per CU.
__global__ void kernelEBOSemaphore(hipSemaphore_t sem, float * storage,
                                   hipLockData_t * gpuLockData,
                                   const unsigned int numStorageLocs,
                                   const int ITERATIONS, const int NUM_LDST,
                                   const int NUM_CU)
{
  // local variables
  const unsigned int maxSemCount =
      gpuLockData->semaphoreBuffers[((sem * 4 * NUM_CU) + (0 * 4))];
  // If there are fewer WGs than # CUs, need to take into account for various
  // math below.  If WGs >= NUM_CU, use NUM_CU.
  const unsigned int numCU = ((hipGridDim_x < NUM_CU) ? hipGridDim_x : NUM_CU);
  const int cuID = (hipBlockIdx_x % NUM_CU); // mod by # CUs to get CU ID
  // given the hipGridDim_x, we can figure out how many WGs are on our CU -- assume
  // all CUs have an identical number of WGs
  int numWGs_perCU = (int)ceil((float)hipGridDim_x / numCU);
  // number of threads on each WG
  //const int numThrs_perCU = (hipBlockDim_x * numWGs_perCU);
  const int perCU_WGID = (hipBlockIdx_x / numCU);
  // rotate which WG is the writer
  const bool isWriter = (perCU_WGID == (cuID % numWGs_perCU));

  // all work groups on the same CU access unique locations except the writer,
  // which writes all of the locations that all of the WGs access
  //int currWGID = hipBlockIdx_x;
  // the (reader) WGs on each CU access unique locations but those same
  // locations are accessed by the reader WGs on all CUs
  int tid = ((perCU_WGID * hipBlockDim_x) + hipThreadIdx_x);
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (hipThreadIdx_x % NUM_WORDS_PER_CACHELINE);

  // dummy array to hold the loads done in the readers
  __shared__ volatile float dummyArray[NUM_WIS_PER_WG];

  for (int i = 0; i < ITERATIONS; ++i)
  {
    /*
      NOTE: There is a race here for entering the critical section.  Most
      importantly, it means that the at least one of the readers could win and
      thus the readers will read before the writer has had a chance to write
      the data.
    */
   hipSemaphoreEBOWait(sem, isWriter, maxSemCount,
                        gpuLockData->semaphoreBuffers, NUM_CU);

    // writer WG writes all the data that the WGs on this CU access
    if (isWriter) {
      for (int j = 0; j < numWGs_perCU; ++j) {
        /*
          Update the writer's "location" so it writes to the locations that the
          readers will access (due to RR scheduling the next WG on this CU is
          numCU WGs away).  Use loop counter because the non-unique version
          writes the same locations on all CUs.
        */
        tid = ((j * hipBlockDim_x) + hipThreadIdx_x);
        threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));

        accessData_semWr(storage, threadBaseLoc, threadOffset, NUM_LDST);
      }
      // reset locations
      tid = ((perCU_WGID * hipBlockDim_x) + hipThreadIdx_x);
      threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
    }
    // rest of WGs on this CU read the data written by each CU's writer WG
    else {
      accessData_semRd(storage, dummyArray, threadBaseLoc, threadOffset,
                       NUM_LDST);
    }
    hipSemaphoreEBOPost(sem, isWriter, maxSemCount,
                         gpuLockData->semaphoreBuffers, NUM_CU);
  }
}

__global__ void kernelEBOSemaphoreUniq(hipSemaphore_t sem, float * storage,
                                       hipLockData_t * gpuLockData,
                                       const int ITERATIONS,
                                       const int NUM_LDST,
                                       const int NUM_CU)
{
  // local variables
  const unsigned int maxSemCount =
      gpuLockData->semaphoreBuffers[((sem * 4 * NUM_CU) + (0 * 4))];
  // If there are fewer WGs than # CUs, need to take into account for various
  // math below.  If WGs >= NUM_CU, use NUM_CU.
  const unsigned int numCU = ((hipGridDim_x < NUM_CU) ? hipGridDim_x : NUM_CU);
  const int cuID = (hipBlockIdx_x % numCU); // mod by # CUs to get CU ID
  // given the hipGridDim_x, we can figure out how many WGs are on our CU -- assume
  // all CUs have an identical number of WGs
  int numWGs_perCU = (int)ceil((float)hipGridDim_x / numCU);
  const int perCU_WGID = (hipBlockIdx_x / numCU);
  // rotate which WG is the writer
  const bool isWriter = (perCU_WGID == (cuID % numWGs_perCU));

  // all work groups on the same CU access unique locations except the writer,
  // which writes all of the locations that all of the WGs access
  int currWGID = hipBlockIdx_x;
  int tid = ((currWGID * hipBlockDim_x) + hipThreadIdx_x);
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (hipThreadIdx_x % NUM_WORDS_PER_CACHELINE);
  // dummy array to hold the loads done in the readers
  __shared__ volatile float dummyArray[NUM_WIS_PER_WG];

  for (int i = 0; i < ITERATIONS; ++i)
  {
    /*
      NOTE: There is a race here for entering the critical section.  Most
      importantly, it means that the at least one of the readers could win and
      thus the readers will read before the writer has had a chance to write
      the data.
    */
    hipSemaphoreEBOWaitLocal(sem, cuID, isWriter, maxSemCount,
                              gpuLockData->semaphoreBuffers, NUM_CU);

    // writer WG writes all the data that the WGs on this CU access
    if (isWriter) {
      for (int j = 0; j < numWGs_perCU; ++j) {
        accessData_semWr(storage, threadBaseLoc, threadOffset, NUM_LDST);

        /*
          update the writer's "location" so it writes to the locations that the
          readers will access (due to RR scheduling the next WG on this CU is
          numCU WGs away and < hipGridDim_x).

          NOTE: First location writer writes to is its own location(s).  If the
          writer is not CU 0 on this CU, it may require wrapping around to CUs
          with smaller WG IDs.
        */
        currWGID = (currWGID + numCU) % hipGridDim_x;
        tid = ((currWGID * hipBlockDim_x) + hipThreadIdx_x);
        threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
      }
      // reset locations
      currWGID = hipBlockIdx_x;
      tid = ((currWGID * hipBlockDim_x) + hipThreadIdx_x);
      threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
    }
    // rest of WGs on this CU read the data written by each CU's writer WG
    else {
      accessData_semRd(storage, dummyArray, threadBaseLoc, threadOffset,
                       NUM_LDST);
    }
    hipSemaphoreEBOPostLocal(sem, cuID, isWriter, maxSemCount,
                              gpuLockData->semaphoreBuffers, NUM_CU);
  }
}

void invokeAtomicTreeBarrier(float * storage_d, unsigned int * perCUBarriers_d,
                             int numIters)
{
  // local variable
  const int WGs = numWGs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelAtomicTreeBarrierUniq), dim3(WGs), dim3(NUM_WIS_PER_WG), 0, 0,
        storage_d, cpuLockData, perCUBarriers_d, numIters, NUM_LDST, NUM_CU,
        MAX_WGS);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    hipError_t hipErr = hipDeviceSynchronize();
    checkError(hipErr, "hipDeviceSynchronize (kernelAtomicTreeBarrierUniq)");
  }
}

void invokeAtomicTreeBarrierLocalExch(float * storage_d,
                                      unsigned int * perCUBarriers_d,
                                      int numIters)
{
  // local variable
  const int WGs = numWGs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelAtomicTreeBarrierUniqLocalExch), dim3(WGs), dim3(NUM_WIS_PER_WG), 0, 0,
        storage_d, cpuLockData, perCUBarriers_d, numIters, NUM_LDST, NUM_CU,
        MAX_WGS);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    hipError_t hipErr = hipDeviceSynchronize();
    checkError(hipErr,
               "hipDeviceSynchronize (kernelAtomicTreeBarrierUniqLockExch)");
  }
}

void invokeFBSTreeBarrier(float * storage_d, unsigned int * perCUBarriers_d,
                          int numIters)
{
  // local variable
  const int WGs = numWGs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelFBSTreeBarrierUniq), dim3(WGs), dim3(NUM_WIS_PER_WG), 0, 0,
        storage_d, cpuLockData, perCUBarriers_d, numIters, NUM_LDST, NUM_CU,
        MAX_WGS);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    hipError_t hipErr = hipDeviceSynchronize();
    checkError(hipErr, "hipDeviceSynchronize (kernelFBSTreeBarrierUniq)");
  }
}

void invokeFBSTreeBarrierLocalExch(float * storage_d,
                                   unsigned int * perCUBarriers_d,
                                   int numIters)
{
  // local variable
  const int WGs = numWGs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelFBSTreeBarrierUniqLocalExch), dim3(WGs), dim3(NUM_WIS_PER_WG), 0, 0,
        storage_d, cpuLockData, perCUBarriers_d, numIters, NUM_LDST, NUM_CU,
        MAX_WGS);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    hipError_t hipErr = hipDeviceSynchronize();
    checkError(hipErr, "hipDeviceSynchronize (kernelFBSTreeBarrierUniqLocalExch)");
  }
}

void invokeSpinLockMutex(hipMutex_t mutex, float * storage_d, int numIters)
{
  // local variable
  const int WGs = numWGs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelSpinLockMutex), dim3(1, 1, 1), dim3(NUM_WIS_PER_WG), 0, 0,
        mutex, storage_d, cpuLockData, numIters, NUM_LDST, NUM_CU);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    hipError_t hipErr = hipDeviceSynchronize();
    checkError(hipErr, "hipDeviceSynchronize (kernelSpinLockMutex)");
  }
}

void invokeSpinLockMutex_uniq(hipMutex_t mutex, float * storage_d,
                              int numIters)
{
  // local variable
  const int WGs = numWGs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelSpinLockMutexUniq), dim3(WGs), dim3(NUM_WIS_PER_WG), 0, 0,
        mutex, storage_d, cpuLockData, numIters, NUM_LDST, NUM_CU);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    hipError_t hipErr = hipDeviceSynchronize();
    checkError(hipErr, "hipDeviceSynchronize (kernelSpinLockMutexUniq)");
  }
}

void invokeEBOMutex(hipMutex_t mutex, float * storage_d, int numIters)
{
  // local variable
  const int WGs = numWGs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelEBOMutex), dim3(WGs), dim3(NUM_WIS_PER_WG), 0, 0,
        mutex, storage_d, cpuLockData, numIters, NUM_LDST, NUM_CU);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    hipError_t hipErr = hipDeviceSynchronize();
    checkError(hipErr, "hipDeviceSynchronize (kernelEBOMutex)");
  }
}

void invokeEBOMutex_uniq(hipMutex_t mutex, float * storage_d, int numIters)
{
  // local variable
  const int WGs = numWGs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelEBOMutexUniq), dim3(WGs), dim3(NUM_WIS_PER_WG), 0, 0,
        mutex, storage_d, cpuLockData, numIters, NUM_LDST, NUM_CU);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    hipError_t hipErr = hipDeviceSynchronize();
    checkError(hipErr, "hipDeviceSynchronize (kernelEBOMutexUniq)");
  }
}

void invokeSleepingMutex(hipMutex_t mutex, float * storage_d, int numIters)
{
  // local variable
  const int WGs = numWGs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelSleepingMutex), dim3(WGs), dim3(NUM_WIS_PER_WG), 0, 0,
                       mutex, storage_d, cpuLockData, numIters, NUM_LDST, NUM_CU);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    hipError_t hipErr = hipDeviceSynchronize();
    checkError(hipErr, "hipDeviceSynchronize (kernelSleepingMutex)");
  }
}

void invokeSleepingMutex_uniq(hipMutex_t mutex, float * storage_d,
                              int numIters)
{
  // local variable
  const int WGs = numWGs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelSleepingMutexUniq), dim3(WGs), dim3(NUM_WIS_PER_WG), 0, 0,
        mutex, storage_d, cpuLockData, numIters, NUM_LDST, NUM_CU);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    hipError_t hipErr = hipDeviceSynchronize();
    checkError(hipErr, "hipDeviceSynchronize (kernelSleepingMutexUniq)");
  }
}

void invokeFetchAndAddMutex(hipMutex_t mutex, float * storage_d, int numIters)
{
  // local variable
  const int WGs = numWGs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelFetchAndAddMutex), dim3(WGs), dim3(NUM_WIS_PER_WG), 0, 0,
        mutex, storage_d, cpuLockData, numIters, NUM_LDST, NUM_CU);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    hipError_t hipErr = hipDeviceSynchronize();
    checkError(hipErr, "hipDeviceSynchronize (kernelFetchAndAddMutex)");
  }
}

void invokeFetchAndAddMutex_uniq(hipMutex_t mutex, float * storage_d, int numIters)
{
  // local variable
  const int WGs = numWGs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelFetchAndAddMutexUniq), dim3(WGs), dim3(NUM_WIS_PER_WG), 0, 0,
        mutex, storage_d, cpuLockData, numIters, NUM_LDST, NUM_CU);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    hipError_t hipErr = hipDeviceSynchronize();
    checkError(hipErr, "hipDeviceSynchronize (kernelFetchAndAddMutexUniq)");
  }
}

void invokeSpinLockSemaphore(hipSemaphore_t sem, float * storage_d,
                             const int maxVal,
                             int numIters, int numStorageLocs)
{
  // local variable
  const int WGs = numWGs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelSpinLockSemaphore), dim3(WGs), dim3(NUM_WIS_PER_WG), 0, 0,
        sem, storage_d, cpuLockData, numStorageLocs, numIters, NUM_LDST,
        NUM_CU);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    hipError_t hipErr = hipDeviceSynchronize();
    checkError(hipErr, "hipDeviceSynchronize (kernelSpinLockSemaphore)");
  }
}

void invokeSpinLockSemaphore_uniq(hipSemaphore_t sem, float * storage_d,
                                  const int maxVal, int numIters)
{
  // local variable
  const int WGs = numWGs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelSpinLockSemaphoreUniq), dim3(WGs), dim3(NUM_WIS_PER_WG), 0, 0,
        sem, storage_d, cpuLockData, numIters, NUM_LDST, NUM_CU);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    hipError_t hipErr = hipDeviceSynchronize();
    checkError(hipErr, "hipDeviceSynchronize (kernelSpinLockSemaphoreUniq)");
  }
}

void invokeEBOSemaphore(hipSemaphore_t sem, float * storage_d, const int maxVal,
                        int numIters, int numStorageLocs)
{
  // local variable
  const int WGs = numWGs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelEBOSemaphore), dim3(WGs), dim3(NUM_WIS_PER_WG), 0, 0,
        sem, storage_d, cpuLockData, numStorageLocs, numIters, NUM_LDST,
        NUM_CU);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    hipError_t hipErr = hipDeviceSynchronize();
    checkError(hipErr, "hipDeviceSynchronize (kernelEBOSemaphore)");
  }
}

void invokeEBOSemaphore_uniq(hipSemaphore_t sem, float * storage_d,
                             const int maxVal, int numIters)
{
  // local variable
  const int WGs = numWGs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelEBOSemaphoreUniq), dim3(WGs), dim3(NUM_WIS_PER_WG), 0, 0,
        sem, storage_d, cpuLockData, numIters, NUM_LDST, NUM_CU);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    hipError_t hipErr = hipDeviceSynchronize();
    checkError(hipErr, "hipDeviceSynchronize (kernelEBOSemaphoreUniq)");
  }
}

int main(int argc, char ** argv)
{
  if (argc != 5) {
    fprintf(stderr, "./allSyncPrims-1kernel <syncPrim> <numLdSt> <numWGs> "
            "<numCSIters>\n");
    fprintf(stderr, "where:\n");
    fprintf(stderr, "\t<syncPrim>: a string that represents which synchronization primitive to run.\n"
            "\t\tatomicTreeBarrUniq - Atomic Tree Barrier\n"
            "\t\tatomicTreeBarrUniqLocalExch - Atomic Tree Barrier with local exchange\n"
            "\t\tlfTreeBarrUniq - Lock-Free Tree Barrier\n"
            "\t\tlfTreeBarrUniqLocalExch - Lock-Free Tree Barrier with local exchange\n"
            "\t\tspinMutex - Spin Lock Mutex\n"
            "\t\tspinMutexEBO - Spin Lock Mutex with Backoff\n"
            "\t\tsleepMutex - Sleep Mutex\n"
            "\t\tfaMutex - Fetch-and-Add Mutex\n"
            "\t\tspinSem1 - Spin Semaphore (Max: 1)\n"
            "\t\tspinSem2 - Spin Semaphore (Max: 2)\n"
            "\t\tspinSem10 - Spin Semaphore (Max: 10)\n"
            "\t\tspinSem120 - Spin Semaphore (Max: 120)\n"
            "\t\tspinSemEBO1 - Spin Semaphore with Backoff (Max: 1)\n"
            "\t\tspinSemEBO2 - Spin Semaphore with Backoff (Max: 2)\n"
            "\t\tspinSemEBO10 - Spin Semaphore with Backoff (Max: 10)\n"
            "\t\tspinSemEBO120 - Spin Semaphore with Backoff (Max: 120)\n"
            "\t\tspinMutexUniq - Spin Lock Mutex -- accesses to unique locations per WG\n"
            "\t\tspinMutexEBOUniq - Spin Lock Mutex with Backoff -- accesses to unique locations per WG\n"
            "\t\tsleepMutexUniq - Sleep Mutex -- accesses to unique locations per WG\n"
            "\t\tfaMutexUniq - Fetch-and-Add Mutex -- accesses to unique locations per WG\n"
            "\t\tspinSemUniq1 - Spin Semaphore (Max: 1) -- accesses to unique locations per WG\n"
            "\t\tspinSemUniq2 - Spin Semaphore (Max: 2) -- accesses to unique locations per WG\n"
            "\t\tspinSemUniq10 - Spin Semaphore (Max: 10) -- accesses to unique locations per WG\n"
            "\t\tspinSemUniq120 - Spin Semaphore (Max: 120) -- accesses to unique locations per WG\n"
            "\t\tspinSemEBOUniq1 - Spin Semaphore with Backoff (Max: 1) -- accesses to unique locations per WG\n"
            "\t\tspinSemEBOUniq2 - Spin Semaphore with Backoff (Max: 2) -- accesses to unique locations per WG\n"
            "\t\tspinSemEBOUniq10 - Spin Semaphore with Backoff (Max: 10) -- accesses to unique locations per WG\n"
            "\t\tspinSemEBOUniq120 - Spin Semaphore with Backoff (Max: 120) -- accesses to unique locations per WG\n");
    fprintf(stderr, "\t<numLdSt>: the # of LDs and STs to do for each thread "
            "in the critical section.\n");
    fprintf(stderr, "\t<numWGs>: the # of WGs to execute (want to be "
            "divisible by the number of CUs).\n");
    fprintf(stderr, "\t<numCSIters>: number of iterations of the critical "
            "section.\n");
    exit(-1);
  }

  // boilerplate code to identify compute capability, # CU/CUM/CUX, etc.
  int deviceCount;
  hipGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    fprintf(stderr, "There is no device supporting HIP\n");
    exit(-1);
  }

  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, 0);
  fprintf(stdout, "GPU Compute Capability: %d.%d\n", deviceProp.major,
          deviceProp.minor);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
    fprintf(stderr, "There is no HIP capable device\n");
    exit(-1);
  }

  NUM_CU = deviceProp.multiProcessorCount;
  const int maxWGPerCU = deviceProp.maxThreadsPerBlock/NUM_WIS_PER_WG;
  //assert(maxWGPerCU * NUM_WIS_PER_WG <=
  //       deviceProp.maxThreadsPerMultiProcessor);
  MAX_WGS = maxWGPerCU * NUM_CU;

  fprintf(stdout, "# CU: %d, Max Thrs/WG: %d, Max WG/CU: %d, Max # WG: %d\n",
          NUM_CU, deviceProp.maxThreadsPerBlock, maxWGPerCU, MAX_WGS);

  hipError_t hipErr = hipGetLastError();
  checkError(hipErr, "Begin");

  // parse input args
  const char * syncPrim_str = argv[1];
  NUM_LDST = atoi(argv[2]);
  numWGs = atoi(argv[3]);
  assert(numWGs <= MAX_WGS);
  const int NUM_ITERS = atoi(argv[4]);
  const int numWGs_perCU = (int)ceil((float)numWGs / NUM_CU);
  assert(numWGs_perCU > 0);

  unsigned int syncPrim = 9999;
  // set the syncPrim variable to the appropriate value based on the inputted
  // string for the microbenchmark
  if (strcmp(syncPrim_str, "atomicTreeBarrUniq") == 0) { syncPrim = 0; }
  else if (strcmp(syncPrim_str, "atomicTreeBarrUniqLocalExch") == 0) {
    syncPrim = 1;
  }
  else if (strcmp(syncPrim_str, "lfTreeBarrUniq") == 0) { syncPrim = 2; }
  else if (strcmp(syncPrim_str, "lfTreeBarrUniqLocalExch") == 0) {
    syncPrim = 3;
  }
  else if (strcmp(syncPrim_str, "spinMutex") == 0) { syncPrim = 4; }
  else if (strcmp(syncPrim_str, "spinMutexEBO") == 0) { syncPrim = 5; }
  else if (strcmp(syncPrim_str, "sleepMutex") == 0) { syncPrim = 6; }
  else if (strcmp(syncPrim_str, "faMutex") == 0) { syncPrim = 7; }
  else if (strcmp(syncPrim_str, "spinSem1") == 0) { syncPrim = 8; }
  else if (strcmp(syncPrim_str, "spinSem2") == 0) { syncPrim = 9; }
  else if (strcmp(syncPrim_str, "spinSem10") == 0) { syncPrim = 10; }
  else if (strcmp(syncPrim_str, "spinSem120") == 0) { syncPrim = 11; }
  else if (strcmp(syncPrim_str, "spinSemEBO1") == 0) { syncPrim = 12; }
  else if (strcmp(syncPrim_str, "spinSemEBO2") == 0) { syncPrim = 13; }
  else if (strcmp(syncPrim_str, "spinSemEBO10") == 0) { syncPrim = 14; }
  else if (strcmp(syncPrim_str, "spinSemEBO120") == 0) { syncPrim = 15; }
  // cases 16-19 reserved
  else if (strcmp(syncPrim_str, "spinMutexUniq") == 0) { syncPrim = 20; }
  else if (strcmp(syncPrim_str, "spinMutexEBOUniq") == 0) { syncPrim = 21; }
  else if (strcmp(syncPrim_str, "sleepMutexUniq") == 0) { syncPrim = 22; }
  else if (strcmp(syncPrim_str, "faMutexUniq") == 0) { syncPrim = 23; }
  else if (strcmp(syncPrim_str, "spinSemUniq1") == 0) { syncPrim = 24; }
  else if (strcmp(syncPrim_str, "spinSemUniq2") == 0) { syncPrim = 25; }
  else if (strcmp(syncPrim_str, "spinSemUniq10") == 0) { syncPrim = 26; }
  else if (strcmp(syncPrim_str, "spinSemUniq120") == 0) { syncPrim = 27; }
  else if (strcmp(syncPrim_str, "spinSemEBOUniq1") == 0) { syncPrim = 28; }
  else if (strcmp(syncPrim_str, "spinSemEBOUniq2") == 0) { syncPrim = 29; }
  else if (strcmp(syncPrim_str, "spinSemEBOUniq10") == 0) { syncPrim = 30; }
  else if (strcmp(syncPrim_str, "spinSemEBOUniq120") == 0) { syncPrim = 31; }
  // cases 32-36 reserved
  else
  {
    fprintf(stderr, "ERROR: Unknown synchronization primitive: %s\n",
            syncPrim_str);
    exit(-1);
  }

  // multiply number of mutexes, semaphores by NUM_CU to
  // allow per-core locks
  hipLocksInit(MAX_WGS, 8 * NUM_CU, 24 * NUM_CU, pageAlign, NUM_CU, NUM_REPEATS, NUM_ITERS);

  hipErr = hipGetLastError();
  checkError(hipErr, "After hipLocksInit");

  /*
    The barriers need a per-CU barrier that is not part of the global synch
    structure.  In terms of size, for the lock-free barrier there are 2 arrays
    in here -- inVars and outVars.  Each needs to be sized to hold the maximum
    number of WGs/CU and each CU needs an array.

    The atomic barrier per-CU synchronization fits inside the lock-free size
    requirements so we can reuse the same locations.
  */
  unsigned int * perCUBarriers;
  hipHostMalloc(&perCUBarriers, sizeof(unsigned int) * (NUM_CU * MAX_WGS * 2));

  int numLocsMult = 0;
  // barriers and unique semaphores have numWGs WGs accessing unique locations
  if ((syncPrim < 4) ||
      ((syncPrim >= 24) && (syncPrim <= 35))) { numLocsMult = numWGs; }
  // The non-unique mutex microbenchmarks, all WGs access the same locations so
  // multiplier is 1
  else if ((syncPrim >= 4) && (syncPrim <= 7)) { numLocsMult = 1; }
  // The non-unique semaphores have 1 writer and numWGs_perCU - 1 readers per CU
  // so the multiplier is numWGs_perCU
  else if ((syncPrim >= 8) && (syncPrim <= 19)) { numLocsMult = numWGs_perCU; }
  // For the unique mutex microbenchmarks and condition variable, all WGs on
  // same CU access same data so multiplier is NUM_CU.
  else if (((syncPrim >= 20) && (syncPrim <= 23)) ||
           (syncPrim == 36)) { numLocsMult = ((numWGs < NUM_CU) ?
                                              numWGs : NUM_CU); }
  else { // should never reach here
    fprintf(stderr, "ERROR: Unknown syncPrim: %u\n", syncPrim);
    exit(-1);
  }

  // each thread in a WG accesses NUM_LDST locations but accesses
  // per WI are offset so that each subsequent access is dependent
  // on the previous one -- thus need an extra access per WI.
  int numUniqLocsAccPerWG = (NUM_WIS_PER_WG * (NUM_LDST + 1));
  assert(numUniqLocsAccPerWG > 0);
  int numStorageLocs = (numLocsMult * numUniqLocsAccPerWG);
  assert(numStorageLocs > 0);
  float * storage;
  hipHostMalloc(&storage, sizeof(float) * numStorageLocs);

  fprintf(stdout, "# WGs: %d, # Ld/St: %d, # Locs Mult: %d, # Uniq Locs/WG: %d, # Storage Locs: %d\n", numWGs, NUM_LDST, numLocsMult, numUniqLocsAccPerWG, numStorageLocs);

  // initialize storage
  for (int i = 0; i < numStorageLocs; ++i) { storage[i] = i; }
  // initialize per-CU barriers to 0's
  for (int i = 0; i < (NUM_CU * MAX_WGS * 2); ++i) { perCUBarriers[i] = 0; }

  // lock variables
  hipMutex_t spinMutex, faMutex, sleepMutex, eboMutex;
  hipMutex_t spinMutex_uniq, faMutex_uniq, sleepMutex_uniq, eboMutex_uniq;
  hipSemaphore_t spinSem1, eboSem1,
                  spinSem2, eboSem2,
                  spinSem10, eboSem10,
                  spinSem120, eboSem120;
  hipSemaphore_t spinSem1_uniq, eboSem1_uniq,
                  spinSem2_uniq, eboSem2_uniq,
                  spinSem10_uniq, eboSem10_uniq,
                  spinSem120_uniq, eboSem120_uniq;
  switch (syncPrim) {
    case 0: // atomic tree barrier doesn't require any special fields to be
            // created
      printf("atomic_tree_barrier_%03d\n", NUM_ITERS); fflush(stdout);
      break;
    case 1: // atomic tree barrier with local exchange doesn't require any
            // special fields to be created
      printf("atomic_tree_barrier_localExch_%03d\n", NUM_ITERS); fflush(stdout);
      break;
    case 2: // lock-free tree barrier doesn't require any special fields to be
            // created
      printf("fbs_tree_barrier_%03d\n", NUM_ITERS); fflush(stdout);
      break;
    case 3: // lock-free barrier with local exchange doesn't require any
            // special fields to be created
      printf("fbs_tree_barrier_localExch_%03d\n", NUM_ITERS); fflush(stdout);
      break;
    case 4:
      printf("spin_lock_mutex_%03d\n", NUM_ITERS); fflush(stdout);
      hipMutexCreateSpin     (&spinMutex,          0);
      break;
    case 5:
      printf("ebo_mutex_%03d\n", NUM_ITERS); fflush(stdout);
      hipMutexCreateEBO      (&eboMutex,           1);
      break;
    case 6:
      printf("sleeping_mutex_%03d\n", NUM_ITERS); fflush(stdout);
      hipMutexCreateSleep    (&sleepMutex,         2);
      break;
    case 7:
      printf("fetchadd_mutex_%03d\n", NUM_ITERS); fflush(stdout);
      hipMutexCreateFA       (&faMutex,            3);
      break;
    case 8:
      printf("spin_lock_sem_%03d_%03d\n", 1, NUM_ITERS); fflush(stdout);
      hipSemaphoreCreateSpin (&spinSem1,      0,   1, NUM_CU);
      break;
    case 9:
      printf("spin_lock_sem_%03d_%03d\n", 2, NUM_ITERS); fflush(stdout);
      hipSemaphoreCreateSpin (&spinSem2,      1,   2, NUM_CU);
      break;
    case 10:
      printf("spin_lock_sem_%03d_%03d\n", 10, NUM_ITERS); fflush(stdout);
      hipSemaphoreCreateSpin (&spinSem10,      2,   10, NUM_CU);
      break;
    case 11:
      printf("spin_lock_sem_%03d_%03d\n", 2, NUM_ITERS); fflush(stdout);
      hipSemaphoreCreateSpin (&spinSem120,      3,   120, NUM_CU);
      break;
    case 12:
      printf("ebo_sem_%03d_%03d\n", 1, NUM_ITERS); fflush(stdout);
      hipSemaphoreCreateEBO  (&eboSem1,       4,   1, NUM_CU);
      break;
    case 13:
      printf("ebo_sem_%03d_%03d\n", 2, NUM_ITERS); fflush(stdout);
      hipSemaphoreCreateEBO  (&eboSem2,       5,   2, NUM_CU);
      break;
    case 14:
      printf("ebo_sem_%03d_%03d\n", 10, NUM_ITERS); fflush(stdout);
      hipSemaphoreCreateEBO  (&eboSem10,       6,   10, NUM_CU);
      break;
    case 15:
      printf("ebo_sem_%03d_%03d\n", 120, NUM_ITERS); fflush(stdout);
      hipSemaphoreCreateEBO  (&eboSem120,       7,   120, NUM_CU);
      break;
    // cases 16-19 reserved
    case 16:
      break;
    case 17:
      break;
    case 18:
      break;
    case 19:
      break;
    case 20:
      printf("spin_lock_mutex_uniq_%03d\n", NUM_ITERS); fflush(stdout);
      hipMutexCreateSpin     (&spinMutex_uniq,          4);
      break;
    case 21:
      printf("ebo_mutex_uniq_%03d\n", NUM_ITERS); fflush(stdout);
      hipMutexCreateEBO      (&eboMutex_uniq,           5);
      break;
    case 22:
      printf("sleeping_mutex_uniq_%03d\n", NUM_ITERS); fflush(stdout);
      hipMutexCreateSleep    (&sleepMutex_uniq,         6);
      break;
    case 23:
      printf("fetchadd_mutex_uniq_%03d\n", NUM_ITERS); fflush(stdout);
      hipMutexCreateFA       (&faMutex_uniq,            7);
      break;
    case 24:
      printf("spin_lock_sem_uniq_%03d_%03d\n", 1, NUM_ITERS); fflush(stdout);
      hipSemaphoreCreateSpin (&spinSem1_uniq,      12,   1, NUM_CU);
      break;
    case 25:
      printf("spin_lock_sem_uniq_%03d_%03d\n", 2, NUM_ITERS); fflush(stdout);
      hipSemaphoreCreateSpin (&spinSem2_uniq,      13,   2, NUM_CU);
      break;
    case 26:
      printf("spin_lock_sem_uniq_%03d_%03d\n", 10, NUM_ITERS); fflush(stdout);
      hipSemaphoreCreateSpin (&spinSem10_uniq,      14,   10, NUM_CU);
      break;
    case 27:
      printf("spin_lock_sem_uniq_%03d_%03d\n", 2, NUM_ITERS); fflush(stdout);
      hipSemaphoreCreateSpin (&spinSem120_uniq,      15,   120, NUM_CU);
      break;
    case 28:
      printf("ebo_sem_uniq_%03d_%03d\n", 1, NUM_ITERS); fflush(stdout);
      hipSemaphoreCreateEBO  (&eboSem1_uniq,       16,   1, NUM_CU);
      break;
    case 29:
      printf("ebo_sem_uniq_%03d_%03d\n", 2, NUM_ITERS); fflush(stdout);
      hipSemaphoreCreateEBO  (&eboSem2_uniq,       17,   2, NUM_CU);
      break;
    case 30:
      printf("ebo_sem_uniq_%03d_%03d\n", 10, NUM_ITERS); fflush(stdout);
      hipSemaphoreCreateEBO  (&eboSem10_uniq,       18,   10, NUM_CU);
      break;
    case 31:
      printf("ebo_sem_uniq_%03d_%03d\n", 120, NUM_ITERS); fflush(stdout);
      hipSemaphoreCreateEBO  (&eboSem120_uniq,       19,   120, NUM_CU);
      break;
    // cases 32-36 reserved
    case 32:
      break;
    case 33:
      break;
    case 34:
      break;
    case 35:
      break;
    case 36:
      break;
    default:
      fprintf(stderr, "ERROR: Trying to run synch prim #%u, but only 0-36 are "
              "supported\n", syncPrim);
      exit(-1);
      break;
  }

  // # WGs must be < maxBufferSize or sleep mutex ring buffer won't work
  if ((syncPrim == 6) || (syncPrim == 22)) {
    assert(MAX_WGS <= cpuLockData->maxBufferSize);
  }

  // NOTE: region of interest begins here
  hipDeviceSynchronize();

  switch (syncPrim) {
    case 0: // atomic tree barrier
      invokeAtomicTreeBarrier(storage, perCUBarriers, NUM_ITERS);
      break;
    case 1: // atomic tree barrier with local exchange
      invokeAtomicTreeBarrierLocalExch(storage, perCUBarriers, NUM_ITERS);
      break;
    case 2: // lock-free barrier
      invokeFBSTreeBarrier(storage, perCUBarriers, NUM_ITERS);
      break;
    case 3: // lock-free barrier with local exchange
      invokeFBSTreeBarrierLocalExch(storage, perCUBarriers, NUM_ITERS);
      break;
    case 4: // Spin Lock Mutex
      invokeSpinLockMutex   (spinMutex,  storage, NUM_ITERS);
      break;
    case 5: // Spin Lock Mutex with backoff
      invokeEBOMutex        (eboMutex,   storage, NUM_ITERS);
      break;
    case 6: // Sleeping Mutex
      invokeSleepingMutex   (sleepMutex, storage, NUM_ITERS);
      break;
    case 7: // fetch-and-add mutex
      invokeFetchAndAddMutex(faMutex,    storage, NUM_ITERS);
      break;
    case 8: // spin semaphore (1)
      invokeSpinLockSemaphore(spinSem1,   storage,   1, NUM_ITERS, numStorageLocs);
      break;
    case 9: // spin semaphore (2)
      invokeSpinLockSemaphore(spinSem2,   storage,   2, NUM_ITERS, numStorageLocs);
      break;
    case 10: // spin semaphore (10)
      invokeSpinLockSemaphore(spinSem10,   storage,   10, NUM_ITERS, numStorageLocs);
      break;
    case 11: // spin semaphore (120)
      invokeSpinLockSemaphore(spinSem120,   storage,   120, NUM_ITERS, numStorageLocs);
      break;
    case 12: // spin semaphore with backoff (1)
      invokeEBOSemaphore(eboSem1,   storage,     1, NUM_ITERS, numStorageLocs);
      break;
    case 13: // spin semaphore with backoff (2)
      invokeEBOSemaphore(eboSem2,   storage,     2, NUM_ITERS, numStorageLocs);
      break;
    case 14: // spin semaphore with backoff (10)
      invokeEBOSemaphore(eboSem10,   storage,   10, NUM_ITERS, numStorageLocs);
      break;
    case 15: // spin semaphore with backoff (120)
      invokeEBOSemaphore(eboSem120,   storage, 120, NUM_ITERS, numStorageLocs);
      break;
    // cases 16-19 reserved
    case 16:
      break;
    case 17:
      break;
    case 18:
      break;
    case 19:
      break;
    case 20: // Spin Lock Mutex (uniq)
      invokeSpinLockMutex_uniq   (spinMutex_uniq,  storage, NUM_ITERS);
      break;
    case 21: // Spin Lock Mutex with backoff (uniq)
      invokeEBOMutex_uniq        (eboMutex_uniq,   storage, NUM_ITERS);
      break;
    case 22: // Sleeping Mutex (uniq)
      invokeSleepingMutex_uniq   (sleepMutex_uniq, storage, NUM_ITERS);
      break;
    case 23: // fetch-and-add mutex (uniq)
      invokeFetchAndAddMutex_uniq(faMutex_uniq,    storage, NUM_ITERS);
      break;
    case 24: // spin semaphore (1) (uniq)
      invokeSpinLockSemaphore_uniq(spinSem1_uniq,   storage,   1, NUM_ITERS);
      break;
    case 25: // spin semaphore (2) (uniq)
      invokeSpinLockSemaphore_uniq(spinSem2_uniq,   storage,   2, NUM_ITERS);
      break;
    case 26: // spin semaphore (10) (uniq)
      invokeSpinLockSemaphore_uniq(spinSem10_uniq,   storage,   10, NUM_ITERS);
      break;
    case 27: // spin semaphore (120) (uniq)
      invokeSpinLockSemaphore_uniq(spinSem120_uniq,   storage,   120, NUM_ITERS);
      break;
    case 28: // spin semaphore with backoff (1) (uniq)
      invokeEBOSemaphore_uniq(eboSem1_uniq,   storage,     1, NUM_ITERS);
      break;
    case 29: // spin semaphore with backoff (2) (uniq)
      invokeEBOSemaphore_uniq(eboSem2_uniq,   storage,     2, NUM_ITERS);
      break;
    case 30: // spin semaphore with backoff (10) (uniq)
      invokeEBOSemaphore_uniq(eboSem10_uniq,   storage,   10, NUM_ITERS);
      break;
    case 31: // spin semaphore with backoff (120) (uniq)
      invokeEBOSemaphore_uniq(eboSem120_uniq,   storage, 120, NUM_ITERS);
      break;
    // cases 32-36 reserved
    case 32:
      break;
    case 33:
      break;
    case 34:
      break;
    case 35:
      break;
    case 36:
      break;
    default:
      fprintf(stderr,
              "ERROR: Trying to run synch prim #%u, but only 0-36 are "
              "supported\n",
              syncPrim);
      exit(-1);
      break;
  }

  // NOTE: Can end simulation here if don't care about output checking
  hipDeviceSynchronize();

  // get golden results
  float storageGolden[numStorageLocs];
  int numLocsAccessed = 0, currLoc = 0;
  // initialize
  for (int i = 0; i < numStorageLocs; ++i) { storageGolden[i] = i; }

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    for (int j = 0; j < NUM_ITERS; ++j)
    {
      /*
        The barrier algorithms exchange data across CUs, so we need to perform
        the exchanges in the golden code.

        The barrier algorithms with local exchange exchange data both across
        CUs and across WGs within an CU, so need to perform both in the golden
        code.
      */
      if (syncPrim < 4)
      {
        // Some kernels only access a fraction of the total # of locations,
        // determine how many locations are accessed by each kernel here.
        numLocsAccessed = (numWGs * numUniqLocsAccPerWG);

        // first cache line of words aren't written to
        for (int i = (numLocsAccessed-1); i >= 0; --i)
        {
          // every iteration of the critical section, the location being
          // accessed is shifted by numUniqLocsAccPerWG
          currLoc = (i + (j * numUniqLocsAccPerWG)) % numLocsAccessed;

          accessData_golden(storageGolden, currLoc, numStorageLocs);
        }

        // local exchange versions do additional accesses when there are
        // multiple WGs on a CU
        if ((syncPrim == 1) || (syncPrim == 3))
        {
          if (numWGs_perCU > 1)
          {
            for (int i = (numLocsAccessed-1); i >= 0; --i)
            {
              // advance the current location by the number of unique locations
              // accessed by a CU (mod the number of memory locations accessed)
              currLoc = (i + (numWGs_perCU * numUniqLocsAccPerWG)) %
                        numLocsAccessed;
              // every iteration of the critical section, the location being
              // accessed is also shifted by numUniqLocsAccPerWG
              currLoc = (currLoc + (j * numUniqLocsAccPerWG)) %
                        numLocsAccessed;

              accessData_golden(storageGolden, currLoc, numStorageLocs);
            }
          }
        }
      }
      /*
        In the non-unique mutex microbenchmarks (4-7), all WGs on all CUs access
        the same locations.
      */
      else if ((syncPrim >= 4) && (syncPrim <= 7))
      {
        // need to iterate over the locations for each WG since all WGs
        // access the same locations
        for (int WG = 0; WG < numWGs; ++WG)
        {
          for (int i = (numUniqLocsAccPerWG-1); i >= 0; --i)
          {
            accessData_golden(storageGolden, i, numStorageLocs);
          }
        }
      }
      /*
        In the non-unique semaphore microbenchmarks (8-19), 1 "writer" WG
        per CU writes all the locations accessed by that CU (i.e.,
        numUniqLocsAccPerWG * numWGs_perCU).  Moreover, all writer WGs across
        all CUs access the same locations.
      */
      else if ((syncPrim <= 19) && (syncPrim >= 8))
      {
        int cuID = 0, perCU_wgID = 0;
        const int numCU = ((numWGs < NUM_CU) ? numWGs : NUM_CU);
        bool isWriter = false;

        // need to iterate over the locations for each WG since all WGs
        // access the same locations
        for (int wg = 0; wg < numWGs; ++wg)
        {
          cuID = (wg % numCU);
          perCU_wgID = (wg / numCU);
          // which WG is writer varies per CU
          isWriter = (perCU_wgID == (cuID % numWGs_perCU));

          if (isWriter)
          {
            for (int k = 0; k < numWGs_perCU; ++k)
            {
              // first cache line of words aren't written to
              for (int i = (numUniqLocsAccPerWG-1); i >= 0; --i)
              {
                /*
                  The locations the writer is writing are numUniqLocsAccPerWG
                  apart because the WGs are assigned in round-robin fashion.
                  Thus, need to shift the location accordingly.
                */
                currLoc = (i + (k * numUniqLocsAccPerWG)) % numStorageLocs;
                accessData_golden(storageGolden, currLoc, numStorageLocs);
              }
            }
          }
        }
      }
      /*
        In the unique mutex microbenchmarks (20-23), all WGs on a CU access
        the same data and the data accessed by each CU is unique.
      */
      else if ((syncPrim <= 23) && (syncPrim >= 20))
      {
        // Some kernels only access a fraction of the total # of locations,
        // determine how many locations are accessed by each kernel here.
        numLocsAccessed = (numWGs * numUniqLocsAccPerWG);

        // first cache line of words aren't written to
        for (int i = (numLocsAccessed-1); i >= 0; --i)
        {
          /*
            If this location would be accessed by a WG other than the first
            WG on an CU, wraparound and access the same location as the
            first WG on the CU -- only for the mutexes, for semaphores this
            isn't true.
          */
          currLoc = i % (NUM_CU * numUniqLocsAccPerWG);

          accessData_golden(storageGolden, currLoc, numStorageLocs);
        }
      }
      /*
        In the unique semaphore microbenchmarks (24-35), 1 "writer" WG per
        CU writes all the locations accessed by that CU, but each CU accesses
        unique data.  We model this behavior by accessing all of the data
        accessed by all CUs, since this has the same effect (assuming same
        number of WGs/CU).
      */
      else
      {
        // Some kernels only access a fraction of the total # of locations,
        // determine how many locations are accessed by each kernel here.
        numLocsAccessed = (numWGs * numUniqLocsAccPerWG);

        // first cache line of words aren't written to
        for (int i = (numLocsAccessed-1); i >= 0; --i)
        {
          accessData_golden(storageGolden, i, numStorageLocs);
        }
      }
    }
  }

  fprintf(stdout, "Comparing GPU results to golden results:\n");
  unsigned int numErrors = 0;
  // check the output values
  for (int i = 0; i < numStorageLocs; ++i)
  {
    if (std::abs(storage[i] - storageGolden[i]) > 1E-5)
    {
      fprintf(stderr, "\tERROR: storage[%d] = %f, golden[%d] = %f\n", i,
              storage[i], i, storageGolden[i]);
      ++numErrors;
    }
  }
  if (numErrors > 0)
  {
    fprintf(stderr, "ERROR: %s has %u output errors\n", syncPrim_str,
            numErrors);
    exit(-1);
  }
  else { fprintf(stdout, "PASSED!\n"); }

  // free arrays
  hipLocksDestroy();
  hipHostFree(storage);
  hipHostFree(perCUBarriers);

  return 0;
}
