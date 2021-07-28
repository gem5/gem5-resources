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

#ifndef CORE_INCLUDE_DATA_MANAGER_H_
#define CORE_INCLUDE_DATA_MANAGER_H_

#include <memory>
#include <map>
#include <glog/logging.h>

#include "common.h"
#include "data_png.h"

namespace dnnmark {

template <typename T>
class Data {
 private:
  PseudoNumGenerator *png_;
  size_t size_;
  T *gpu_ptr_;
 public:
  Data(size_t size)
  : size_(size) {
    LOG(INFO) << "Create Data chunk of size " << size_;
#ifdef NVIDIA_CUDNN
    CUDA_CALL(cudaMalloc(&gpu_ptr_, size * sizeof(T)));
#endif
#ifdef AMD_MIOPEN
    HIP_CALL(hipHostMalloc(&gpu_ptr_, size * sizeof(T)));
#endif
  }
  ~Data() {
    LOG(INFO) << "Free Data chunk of size " << size_;
    if (size_ > 0) {
#ifdef NVIDIA_CUDNN
      CUDA_CALL(cudaFree(gpu_ptr_));
#endif
#ifdef AMD_MIOPEN
      HIP_CALL(hipHostFree(gpu_ptr_));
#endif
    }
  }
  void Filler() {
    png_ = PseudoNumGenerator::GetInstance();
    png_->GenerateUniformData(gpu_ptr_, size_);
  }
  T *Get() { return gpu_ptr_; }
};


template <typename T>
class DataManager {
 private:
  // Memory pool indexed by chunk id
  std::map<int, std::shared_ptr<Data<T>>> gpu_data_pool_;
  int num_data_chunks_;

  // Constructor
  DataManager()
  : num_data_chunks_(0) {
  }

  // Memory manager instance
  static std::unique_ptr<DataManager<T>> instance_;
 public:
  static DataManager<T> *GetInstance() {
    if (instance_.get())
      return instance_.get();
    instance_.reset(new DataManager());
    return instance_.get();
  }

  ~DataManager() {
    gpu_data_pool_.clear();
  }

  int CreateData(size_t size) {
    int gen_chunk_id = num_data_chunks_;
    num_data_chunks_++;
    gpu_data_pool_.emplace(gen_chunk_id, std::make_shared<Data<T>>(size));
    LOG(INFO) << "Create data with ID: " << gen_chunk_id;
    return gen_chunk_id;
  }

  void RemoveData(int chunk_id) {
    gpu_data_pool_.erase(chunk_id);
  }

  Data<T> *GetData(int chunk_id) {
    return gpu_data_pool_[chunk_id].get();
  }
};

template <typename T>
std::unique_ptr<DataManager<T>> DataManager<T>::instance_ = nullptr;

} // namespace dnnmark

#endif // CORE_INCLUDE_DATA_MANAGER_H_

