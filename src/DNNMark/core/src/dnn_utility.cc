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

#include "dnn_utility.h"

namespace dnnmark {

Handle::Handle() {
#ifdef NVIDIA_CUDNN
  cudnn_handles_ = new cudnnHandle_t[1];
  CUDNN_CALL(cudnnCreate(&cudnn_handles_[0]));
  blas_handles_ = new cublasHandle_t[1];
  CUBLAS_CALL(cublasCreate(&blas_handles_[0]));
  num_cudnn_handles_ = 1;
  num_blas_handles_ = 1;
#endif
#ifdef AMD_MIOPEN
  miopen_handles_ = new miopenHandle_t[1];
  rocblas_handles_ = new rocblas_handle[1];
  MIOPEN_CALL(miopenCreate(&miopen_handles_[0]));
  ROCBLAS_CALL(rocblas_create_handle(&rocblas_handles_[0]));
  num_miopen_handles_ = 1;
  num_rocblas_handles_ = 1;
#endif
}

Handle::Handle(int num) {
#ifdef NVIDIA_CUDNN
  cudnn_handles_ = new cudnnHandle_t[num];
  for (int i = 0; i < num; i++)
    CUDNN_CALL(cudnnCreate(&cudnn_handles_[i]));
  num_cudnn_handles_ = num;

  blas_handles_ = new cublasHandle_t[num];
  for (int i = 0; i < num; i++)
    CUBLAS_CALL(cublasCreate(&blas_handles_[i]));
  num_blas_handles_ = num;
#endif
#ifdef AMD_MIOPEN
  miopen_handles_ = new miopenHandle_t[num];
  rocblas_handles_ = new rocblas_handle[num];
  for (int i = 0; i < num; i++) {
    MIOPEN_CALL(miopenCreate(&miopen_handles_[i]));
    ROCBLAS_CALL(rocblas_create_handle(&rocblas_handles_[i]));
  }
  num_miopen_handles_ = num;
  num_rocblas_handles_ = num;
#endif
}

Handle::~Handle() {
#ifdef NVIDIA_CUDNN
  for (int i = 0; i < num_cudnn_handles_; i++)
    CUDNN_CALL(cudnnDestroy(cudnn_handles_[i]));
  delete []cudnn_handles_;
  for (int i = 0; i < num_blas_handles_; i++)
    CUBLAS_CALL(cublasDestroy(blas_handles_[i]));
  delete []blas_handles_;
#endif
#ifdef AMD_MIOPEN
  for (int i = 0; i < num_miopen_handles_; i++) {
    MIOPEN_CALL(miopenDestroy(miopen_handles_[i]));
  }
  delete []miopen_handles_;
  for (int i = 0; i < num_rocblas_handles_; i++) {
    ROCBLAS_CALL(rocblas_destroy_handle(rocblas_handles_[i]));
  }
  delete []rocblas_handles_;
#endif
}

#ifdef NVIDIA_CUDNN
cudnnHandle_t Handle::GetCudnn() const { return cudnn_handles_[0]; }
cudnnHandle_t Handle::GetCudnn(int index) const {
  return cudnn_handles_[index];
}
cublasHandle_t Handle::GetBlas() const { return blas_handles_[0]; }
cublasHandle_t Handle::GetBlas(int index) const { return blas_handles_[index]; }
#endif
#ifdef AMD_MIOPEN
miopenHandle_t Handle::GetMIOpen() const { return miopen_handles_[0]; }
miopenHandle_t Handle::GetMIOpen(int index) const {
  return miopen_handles_[index];
}
rocblas_handle Handle::GetBlas() const { return rocblas_handles_[0]; }
rocblas_handle Handle::GetBlas(int index) const {
  return rocblas_handles_[index];
}
#endif

Descriptor::Descriptor()
: set_(false) {}

Descriptor::~Descriptor() {
  set_ = false;
}

bool Descriptor::isSet() { return set_; }

} // namespace dnnmark

