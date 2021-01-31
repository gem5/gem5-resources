// The MIT License (MIT)
//
// Copyright (c) 2016 Northeastern University
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef CORE_INCLUDE_COMMON_H_
#define CORE_INCLUDE_COMMON_H_

#include <iostream>

#ifdef NVIDIA_CUDNN
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <curand.h>
#include <cudnn.h>
#include <cublas_v2.h>
#endif

#ifdef AMD_MIOPEN
#include <miopen/miopen.h>
#include <hip/hip_runtime_api.h>
#endif

namespace dnnmark {

#ifdef NVIDIA_CUDNN

#define CUDA_CALL(x) \
do {\
  cudaError_t ret = x;\
  if(ret != cudaSuccess) {\
    std::cout << "CUDA Error at " << __FILE__ << __LINE__ << std::endl;\
    std::cout << cudaGetErrorString(ret) << std::endl;\
    exit(EXIT_FAILURE);\
  }\
} while(0)\

#define CURAND_CALL(x) \
do {\
  if((x) != CURAND_STATUS_SUCCESS) {\
    std::cout << "CURAND Error at " << __FILE__ << __LINE__;\
    exit(EXIT_FAILURE);\
  }\
} while(0)\

#define CUDNN_CALL(x) \
do {\
  cudnnStatus_t ret = x;\
  if(ret != CUDNN_STATUS_SUCCESS) {\
    std::cout << "CUDNN Error at " << __FILE__ << __LINE__;\
    std::cout << cudnnGetErrorString(ret) << std::endl;\
    exit(EXIT_FAILURE);\
  }\
} while(0)\

#define CUBLAS_CALL(x) \
do {\
  if((x) != CUBLAS_STATUS_SUCCESS) {\
    std::cout << "CUDNN Error at " << __FILE__ << __LINE__;\
    exit(EXIT_FAILURE);\
  }\
} while(0)\

#endif

#ifdef AMD_MIOPEN

#define HIP_CALL(x) \
do {\
  hipError_t ret = x;\
  if(ret != hipSuccess) {\
    std::cout << "HIP Error at " << __FILE__ << __LINE__ << std::endl;\
    std::cout << hipGetErrorString(ret) << std::endl;\
    exit(EXIT_FAILURE);\
  }\
} while(0)\

#define MIOPEN_CALL(x) \
do {\
  miopenStatus_t ret = x;\
  if(ret != miopenStatusSuccess) {\
    std::cout << "MIOpen Error: " << ret << " at " << __FILE__ << __LINE__;\
    exit(EXIT_FAILURE);\
  }\
} while(0)\

#define ROCBLAS_CALL(x) \
do {\
  rocblas_status ret = x;\
  if (ret != rocblas_status_success) {\
    std::cout << "ROCBLAS Error: " << ret << " at " << __FILE__ << __LINE__;\
    exit(EXIT_FAILURE);\
  }\
} while(0)\

#endif

#define CONFIG_CHECK(x) \
do {\
  if ((x) != 0) {\
    std::cout << "Parse configuration Error at " << __FILE__ << __LINE__;\
    exit(EXIT_FAILURE);\
  }\
} while(0)\

#ifdef DOUBLE_TEST
#define TestType double
#else
#define TestType float
#endif

// Code courtesy of Caffe
template <typename T>
class DataType;
template <> class DataType<float>  {
 public:
#ifdef NVIDIA_CUDNN
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
#endif
#ifdef AMD_MIOPEN
  static const miopenDataType_t type = miopenFloat;
#endif
  static float oneval, zeroval;
  static const void *one, *zero;
};

#ifdef NVIDIA_CUDNN
template <> class DataType<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
  static double oneval, zeroval;
  static const void *one, *zero;
};
#endif

// Min epsilon for BN
#define BN_MIN_EPSILON 1e-5

// Benchmark running mode
// None: the benchmark haven't been setup
// Standalone: only one layer will be benchmarked
// Composed: multiple layers will be benchmarked, maybe a whole network
enum RunMode {
  NONE = 0,
  STANDALONE,
  COMPOSED
};

// Layer type
enum LayerType {
  CONVOLUTION = 1,
  POOLING,
  LRN,
  ACTIVATION,
  FC,
  SOFTMAX,
  BN,
  DROPOUT,
  BYPASS
};

} // namespace dnnmark

#endif // CORE_INCLUDE_COMMON_H_

