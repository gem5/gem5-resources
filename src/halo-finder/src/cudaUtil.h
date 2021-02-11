#include "hip/hip_runtime.h"
#pragma once
#include <stdio.h>

#define cudaCheckError() {                                          \
 hipError_t e=hipGetLastError();                                 \
 if(e!=hipSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,hipGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

template <int hipWarpSize, typename T>
__device__ __inline__ T warpReduceSum(T val) {
  if(hipWarpSize>16) val+=__shfl_down(val,16,hipWarpSize);
  if(hipWarpSize>8) val+=__shfl_down(val,8,hipWarpSize);
  if(hipWarpSize>4) val+=__shfl_down(val,4,hipWarpSize);
  if(hipWarpSize>2) val+=__shfl_down(val,2,hipWarpSize);
  if(hipWarpSize>1) val+=__shfl_down(val,1,hipWarpSize);
  return val;
}
template <int hipWarpSize, typename T>
__device__ __inline__ T warpReduceMax(T val) {
  if(hipWarpSize>16) val=max(val,__shfl_down(val,16,hipWarpSize));
  if(hipWarpSize>8)  val=max(val,__shfl_down(val,8,hipWarpSize));
  if(hipWarpSize>4)  val=max(val,__shfl_down(val,4,hipWarpSize));
  if(hipWarpSize>2)  val=max(val,__shfl_down(val,2,hipWarpSize));
  if(hipWarpSize>1)  val=max(val,__shfl_down(val,1,hipWarpSize));
  return val;
}
template <int hipWarpSize, typename T>
__device__ __inline__ T warpReduceMin(T val) {
  if(hipWarpSize>16) val=min(val,__shfl_down(val,16,hipWarpSize));
  if(hipWarpSize>8)  val=min(val,__shfl_down(val,8,hipWarpSize));
  if(hipWarpSize>4)  val=min(val,__shfl_down(val,4,hipWarpSize));
  if(hipWarpSize>2)  val=min(val,__shfl_down(val,2,hipWarpSize));
  if(hipWarpSize>1)  val=min(val,__shfl_down(val,1,hipWarpSize));
  return val;
}

template <typename T>
__device__ __inline__ T blockReduceSum(T val) {
  __shared__ volatile T smem[32];

  val=warpReduceSum<32>(val);
  if(hipThreadIdx_x%32==0) smem[hipThreadIdx_x/32]=val;
  __syncthreads();
  val=0;
  if(hipThreadIdx_x<hipBlockDim_x/32) val=smem[hipThreadIdx_x];
  if(hipThreadIdx_x<32) val=warpReduceSum<32>(val);
  __syncthreads();
  return val;
}

template <typename T>
__device__ __inline__ T blockReduceMax(T val) {
  __shared__ volatile T smem[32];

  val=warpReduceMax<32>(val);
  if(hipThreadIdx_x%32==0) smem[hipThreadIdx_x/32]=val;
  __syncthreads();
  
  if(hipThreadIdx_x<32==0) {
    val=max(smem[0],smem[hipThreadIdx_x]);
    val=warpReduceMax<32>(val);
  }
  __syncthreads();
  return val;
}

template <typename T>
__device__ __inline__ T blockReduceMin(T val) {
  __shared__ volatile T smem[32];

  val=warpReduceMin<32>(val);
  if(hipThreadIdx_x%32==0) smem[hipThreadIdx_x/32]=val;
  __syncthreads();
  
  if(hipThreadIdx_x<32==0) {
    val=min(smem[0],smem[hipThreadIdx_x]);
    val=warpReduceMin<32>(val);
  }
  __syncthreads();
  return val;
}

__device__  __forceinline__
void atomicWarpReduceAndUpdate(POSVEL_T *out, POSVEL_T val) {
  //perform shfl reduction
  val+=__shfl_down(val, 16); 
  val+=__shfl_down(val, 8); 
  val+=__shfl_down(val, 4);
  val+=__shfl_down(val, 2); 
  val+=__shfl_down(val, 1);

  if(hipThreadIdx_x%32==0)
    atomicAdd(out,val);  //atomics are unecessary but they are faster than non-atomics due to a single bus transaction
}

class cudaDeviceSelector {
  public:
  cudaDeviceSelector() {
    char* str;
    int local_rank = 0;
    int numDev=1;

    //No MPI at this time so go by enviornment variables. 
    //This may need to be updated to match your MPI flavor
    if((str = getenv("MV2_COMM_WORLD_LOCAL_RANK")) != NULL) {
      local_rank = atoi(str);
    }
    else if((str = getenv("OMPI_COMM_WORLD_LOCAL_RANK")) != NULL) {
      local_rank = atoi(str);
    }
    else if((str = getenv("SLURM_LOCALID")) != NULL) {
      local_rank = atoi(str);
    }

    //get the number of devices to use
    if((str = getenv("HACC_NUM_CUDA_DEV")) != NULL) {
      numDev=atoi(str);
    }

#if 0

#if 0
    //Use MPS,  need to figure out how to set numDev, perhaps and enviornment varaible?
    char var[100];
    sprintf(var,"/tmp/nvidia-mps_%d",local_rank%numDev);
    setenv("CUDA_MPS_PIPE_DIRECTORY",var,1);
#endif
#else 
    int dev;
    //set via local MPI rank 
    dev = local_rank % numDev;
 
    //we must set this for all threads
	hipSetDevice(dev);
#endif
  }
};

inline void checkCudaPtr(const void* ptr, const char* name) {
  int rank;

  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  int dev;
  hipGetDevice(&dev);

  hipPointerAttribute_t at;

  hipPointerGetAttributes(&at,ptr);

  if(dev!=at.device) {
    printf("%d: Error '%s', dev: %d, at.device: %d\n", rank, name, dev, at.device);
  }
}

  
