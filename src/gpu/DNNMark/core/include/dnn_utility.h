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

#ifndef CORE_INCLUDE_DNN_UTILITY_H_
#define CORE_INCLUDE_DNN_UTILITY_H_

#include <iostream>

#ifdef NVIDIA_CUDNN
#include "cudnn.h"
#endif

#ifdef AMD_MIOPEN
#include <miopen/miopen.h>
#include <rocblas.h>
#endif

#include "common.h"
#include "dnn_param.h"
#include "timer.h"

namespace dnnmark {

class Handle {
#ifdef NVIDIA_CUDNN
 private:
  cudnnHandle_t *cudnn_handles_;
  cublasHandle_t *blas_handles_;
  int num_cudnn_handles_;
  int num_blas_handles_;
 public:
  Handle();
  Handle(int num);
  ~Handle();
  cudnnHandle_t GetCudnn() const;
  cudnnHandle_t GetCudnn(int index) const;
  cublasHandle_t GetBlas() const;
  cublasHandle_t GetBlas(int index) const;
  int num_cudnn() const { return num_cudnn_handles_; }
  int num_blas() const { return num_blas_handles_; }
#endif
#ifdef AMD_MIOPEN
 private:
  miopenHandle_t *miopen_handles_;
  rocblas_handle *rocblas_handles_;
  int num_miopen_handles_;
  int num_rocblas_handles_;
 public:
  Handle();
  Handle(int num);
  ~Handle();
  miopenHandle_t GetMIOpen() const;
  miopenHandle_t GetMIOpen(int index) const;
  rocblas_handle GetBlas() const;
  rocblas_handle GetBlas(int index) const;
  int num_miopen() const { return num_miopen_handles_; }
  int num_blas() const { return num_rocblas_handles_; }
#endif
};

class Descriptor {
 protected:
  bool set_;
 public:
  Descriptor();
  ~Descriptor();
  bool isSet();
};

template <typename T>
class DataTensor : public Descriptor {
 private:
#ifdef NVIDIA_CUDNN
  cudnnTensorDescriptor_t desc_;
#endif
#ifdef AMD_MIOPEN
  miopenTensorDescriptor_t desc_;
#endif

 public:
  DataTensor()
  : Descriptor() {
#ifdef NVIDIA_CUDNN
    CUDNN_CALL(cudnnCreateTensorDescriptor(&desc_));
#endif
#ifdef AMD_MIOPEN
    MIOPEN_CALL(miopenCreateTensorDescriptor(&desc_));
#endif
  }

  ~DataTensor() {
#ifdef NVIDIA_CUDNN
    CUDNN_CALL(cudnnDestroyTensorDescriptor(desc_));
#endif
#ifdef AMD_MIOPEN
    MIOPEN_CALL(miopenDestroyTensorDescriptor(desc_));
#endif
  }

  void Set(int n, int c, int h, int w) {
    if (!set_) {
#ifdef NVIDIA_CUDNN
      CUDNN_CALL(cudnnSetTensor4dDescriptor(desc_,
                                            CUDNN_TENSOR_NCHW,
                                            DataType<T>::type,
                                            n, c, h, w));
#endif
#ifdef AMD_MIOPEN
      MIOPEN_CALL(miopenSet4dTensorDescriptor(desc_,
                                              DataType<T>::type,
                                              n, c, h, w));
#endif
    }
    set_ = true;
  }

#ifdef NVIDIA_CUDNN
  cudnnTensorDescriptor_t Get() const {
#endif
#ifdef AMD_MIOPEN
  miopenTensorDescriptor_t Get() const {
#endif
    if (set_)
      return desc_;
    return nullptr;
  }

};

template <typename T>
class ConvolutionDesc : public Descriptor {
 private:
#ifdef NVIDIA_CUDNN
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
#endif
#ifdef AMD_MIOPEN
  miopenTensorDescriptor_t filter_desc_;
  miopenConvolutionDescriptor_t conv_desc_;
#endif

 public:
  ConvolutionDesc()
  : Descriptor() {
#ifdef NVIDIA_CUDNN
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc_));
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc_));
#endif
#ifdef AMD_MIOPEN
    MIOPEN_CALL(miopenCreateConvolutionDescriptor(&conv_desc_));
    MIOPEN_CALL(miopenCreateTensorDescriptor(&filter_desc_));
#endif
  }

  ~ConvolutionDesc() {
#ifdef NVIDIA_CUDNN
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc_));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc_));
#endif
#ifdef AMD_MIOPEN
    MIOPEN_CALL(miopenDestroyConvolutionDescriptor(conv_desc_));
    MIOPEN_CALL(miopenDestroyTensorDescriptor(filter_desc_));
#endif
  }

  void Set(const ConvolutionParam &param, int num_channel) {
    if (!set_) {
#ifdef NVIDIA_CUDNN
      CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_desc_,
                 param.pad_h_, param.pad_w_,
                 param.stride_u_, param.stride_v_,
                 param.upscale_x_, param.upscale_y_,
                 param.mode_, DataType<T>::type));

      CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc_,
                 DataType<T>::type, CUDNN_TENSOR_NCHW,
                 param.output_num_, num_channel,
                 param.kernel_size_h_, param.kernel_size_w_));
#endif
#ifdef AMD_MIOPEN
      MIOPEN_CALL(miopenInitConvolutionDescriptor(conv_desc_,
                 param.mode_,
                 param.pad_h_, param.pad_w_,
                 param.stride_u_, param.stride_v_,
                 param.upscale_x_, param.upscale_y_));

      MIOPEN_CALL(miopenSet4dTensorDescriptor(filter_desc_,
                 DataType<T>::type,
                 param.output_num_, num_channel,
                 param.kernel_size_h_, param.kernel_size_w_));
#endif
    }
    set_ = true;
  }

#ifdef NVIDIA_CUDNN
  cudnnFilterDescriptor_t GetFilter() const {
#endif
#ifdef AMD_MIOPEN
  miopenTensorDescriptor_t GetFilter() const {
#endif
    if (set_)
      return filter_desc_;
    return nullptr;
  }

#ifdef NVIDIA_CUDNN
  cudnnConvolutionDescriptor_t GetConv() const {
#endif
#ifdef AMD_MIOPEN
  miopenConvolutionDescriptor_t GetConv() const {
#endif
    if (set_)
      return conv_desc_;
    return nullptr;
  }


};

template <typename T>
class PoolingDesc : public Descriptor {
 private:
#ifdef NVIDIA_CUDNN
  cudnnPoolingDescriptor_t pooling_desc_;
#endif
#ifdef AMD_MIOPEN
  miopenPoolingDescriptor_t pooling_desc_;
#endif
 public:
  PoolingDesc()
  : Descriptor() {
#ifdef NVIDIA_CUDNN
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&pooling_desc_));
#endif
#ifdef AMD_MIOPEN
    MIOPEN_CALL(miopenCreatePoolingDescriptor(&pooling_desc_));
#endif
  }

  ~PoolingDesc() {
#ifdef NVIDIA_CUDNN
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(pooling_desc_));
#endif
#ifdef AMD_MIOPEN
    MIOPEN_CALL(miopenDestroyPoolingDescriptor(pooling_desc_));
#endif
  }

  void Set(const PoolingParam &param) {
    if (!set_) {
#ifdef NVIDIA_CUDNN
      CUDNN_CALL(cudnnSetPooling2dDescriptor(pooling_desc_,
                 param.mode_, CUDNN_PROPAGATE_NAN,
                 param.kernel_size_h_, param.kernel_size_w_,
                 param.pad_h_, param.pad_w_,
                 param.stride_h_, param.stride_w_));
#endif
#ifdef AMD_MIOPEN
      MIOPEN_CALL(miopenSet2dPoolingDescriptor(pooling_desc_,
                 param.mode_,
                 param.kernel_size_h_, param.kernel_size_w_,
                 param.pad_h_, param.pad_w_,
                 param.stride_h_, param.stride_w_));
#endif
    }

    set_ = true;
  }

#ifdef NVIDIA_CUDNN
  cudnnPoolingDescriptor_t Get() const {
#endif
#ifdef AMD_MIOPEN
  miopenPoolingDescriptor_t Get() const {
#endif
    if (set_)
      return pooling_desc_;
    return nullptr;
  }

  void GetWorkspaceSize(const DataTensor<T> &y_desc,
                        size_t *workspace_size) {
#ifdef AMD_MIOPEN
    if (set_)
      MIOPEN_CALL(miopenPoolingGetWorkSpaceSize(y_desc.Get(), workspace_size));
    else
      LOG(FATAL) << "Pooling descriptor NOT set";
#endif
  }
};

template <typename T>
class LRNDesc : public Descriptor {
 private:
#ifdef NVIDIA_CUDNN
  cudnnLRNDescriptor_t lrn_desc_;
#endif
#ifdef AMD_MIOPEN
  miopenLRNDescriptor_t lrn_desc_;
#endif
 public:
  LRNDesc()
  : Descriptor() {
#ifdef NVIDIA_CUDNN
    CUDNN_CALL(cudnnCreateLRNDescriptor(&lrn_desc_));
#endif
#ifdef AMD_MIOPEN
    MIOPEN_CALL(miopenCreateLRNDescriptor(&lrn_desc_));
#endif
  }

  ~LRNDesc() {
#ifdef NVIDIA_CUDNN
    CUDNN_CALL(cudnnDestroyLRNDescriptor(lrn_desc_));
#endif
#ifdef AMD_MIOPEN
    MIOPEN_CALL(miopenDestroyLRNDescriptor(lrn_desc_));
#endif
  }

  void Set(const LRNParam &param) {
    if (!set_) {
#ifdef NVIDIA_CUDNN
      CUDNN_CALL(cudnnSetLRNDescriptor(lrn_desc_,
                 param.local_size_,
                 param.alpha_, param.beta_,
                 param.k_));
#endif
#ifdef AMD_MIOPEN
      MIOPEN_CALL(miopenSetLRNDescriptor(lrn_desc_,
                  param.mode_,
                  param.local_size_,
                  param.alpha_, param.beta_,
                  param.k_));
#endif
    }

    set_ = true;
  }

#ifdef NVIDIA_CUDNN
  cudnnLRNDescriptor_t Get() const {
#endif
#ifdef AMD_MIOPEN
  miopenLRNDescriptor_t Get() const {
#endif
    if (set_)
      return lrn_desc_;
    return nullptr;
  }

  void GetWorkspaceSize(const DataTensor<T> &y_desc,
                        size_t *workspace_size) {
#ifdef AMD_MIOPEN
    if (set_)
      MIOPEN_CALL(miopenLRNGetWorkSpaceSize(y_desc.Get(), workspace_size));
    else
      LOG(FATAL) << "LRN descriptor NOT set";
#endif
  }

};

template <typename T>
class ActivationDesc : public Descriptor {
 private:
#ifdef NVIDIA_CUDNN
  cudnnActivationDescriptor_t activation_desc_;
#endif
#ifdef AMD_MIOPEN
  miopenActivationDescriptor_t activation_desc_;
#endif
 public:
  ActivationDesc()
  : Descriptor() {
#ifdef NVIDIA_CUDNN
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activation_desc_));
#endif
#ifdef AMD_MIOPEN
    MIOPEN_CALL(miopenCreateActivationDescriptor(&activation_desc_));
#endif
  }

  ~ActivationDesc() {
#ifdef NVIDIA_CUDNN
    CUDNN_CALL(cudnnDestroyActivationDescriptor(activation_desc_));
#endif
#ifdef AMD_MIOPEN
    MIOPEN_CALL(miopenDestroyActivationDescriptor(activation_desc_));
#endif
  }

  void Set(const ActivationParam &param) {
    if (!set_) {
#ifdef NVIDIA_CUDNN
      CUDNN_CALL(cudnnSetActivationDescriptor(activation_desc_,
                 param.mode_,
                 CUDNN_PROPAGATE_NAN,
                 double(0.0)));
#endif
#ifdef AMD_MIOPEN
      MIOPEN_CALL(miopenSetActivationDescriptor(activation_desc_,
                 param.mode_,
                 param.alpha_,
                 param.beta_,
                 param.power_));
#endif
    }

    set_ = true;
  }

#ifdef NVIDIA_CUDNN
  cudnnActivationDescriptor_t Get() const {
#endif
#ifdef AMD_MIOPEN
  miopenActivationDescriptor_t Get() const {
#endif
    if (set_)
      return activation_desc_;
    return nullptr;
  }

};

template <typename T>
class BypassDesc : public Descriptor {
 private:
#ifdef NVIDIA_CUDNN
  DataDim dim_;
#endif
#ifdef AMD_MIOPEN
  miopenActivationDescriptor_t activation_desc_;
#endif
 public:
  BypassDesc()
  : Descriptor() {
#ifdef AMD_MIOPEN
    MIOPEN_CALL(miopenCreateActivationDescriptor(&activation_desc_));
#endif
  }

  ~BypassDesc() {
#ifdef AMD_MIOPEN
    MIOPEN_CALL(miopenDestroyActivationDescriptor(activation_desc_));
#endif
  }

  void Set(DataDim dim) {
    if (!set_) {
#ifdef NVIDIA_CUDNN
      dim_.n_ = dim.n_;
      dim_.c_ = dim.c_;
      dim_.h_ = dim.h_;
      dim_.w_ = dim.w_;
#endif
#ifdef AMD_MIOPEN
      MIOPEN_CALL(miopenSetActivationDescriptor(activation_desc_,
                 miopenActivationPASTHRU,
                 0, 0, 0));
#endif
    }

    set_ = true;
  }

#ifdef NVIDIA_CUDNN
  DataDim Get() const {
    return dim_;
  }
#endif
#ifdef AMD_MIOPEN
  miopenActivationDescriptor_t Get() const {
    if (set_)
      return activation_desc_;
    return nullptr;
  }
#endif
};

template <typename T>
class DropoutDesc : public Descriptor {
 private:
#ifdef NVIDIA_CUDNN
  cudnnDropoutDescriptor_t dropout_desc_;
#endif
#ifdef AMD_MIOPEN
#endif
 public:
  DropoutDesc()
  : Descriptor() {
#ifdef NVIDIA_CUDNN
    CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc_));
#endif
#ifdef AMD_MIOPEN
#endif
  }

  ~DropoutDesc() {
#ifdef NVIDIA_CUDNN
    CUDNN_CALL(cudnnDestroyDropoutDescriptor(dropout_desc_));
#endif
#ifdef AMD_MIOPEN
#endif
  }

  void SetStatesSize(const Handle &handle, RunMode mode, int idx,
                     size_t *state_size) {
#ifdef NVIDIA_CUDNN
    CUDNN_CALL(cudnnDropoutGetStatesSize(mode ?
                                         handle.GetCudnn(idx):
                                         handle.GetCudnn(),
                                         state_size));
#endif
#ifdef AMD_MIOPEN
#endif
  }

  void SetReserveSpaceSize(DataTensor<T> &bottom_desc,
                           size_t *reserve_space_size) {
#ifdef NVIDIA_CUDNN
    CUDNN_CALL(cudnnDropoutGetReserveSpaceSize(bottom_desc.Get(),
                                               reserve_space_size));
#endif
#ifdef AMD_MIOPEN
#endif
  }

  void Set(const Handle &handle, RunMode mode, int idx,
           const DropoutParam &dropout_param,
           void *states, size_t state_size) {
    if (!set_) {
#ifdef NVIDIA_CUDNN
      if (state_size > 0)
        CUDNN_CALL(cudnnSetDropoutDescriptor(dropout_desc_,
                                       mode == COMPOSED ?
                                       handle.GetCudnn(idx):
                                       handle.GetCudnn(),
                                       dropout_param.dropout_p_,
                                       states,
                                       state_size,
                                       dropout_param.random_seed_));
      else
        LOG(FATAL) << "The size is ZERO";
#endif
#ifdef AMD_MIOPEN
#endif
    }

    set_ = true;
  }

#ifdef NVIDIA_CUDNN
  cudnnDropoutDescriptor_t Get() const {
    return dropout_desc_;
  }
#endif
#ifdef AMD_MIOPEN
#endif
};

template <typename T>
class ConvAlgo {
#ifdef NVIDIA_CUDNN
 private:
  cudnnConvolutionFwdAlgo_t fwd_algo_;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo_;
  std::string bwd_filter_algo_par;

 public:
  ConvAlgo()
  : fwd_algo_(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM),
    bwd_filter_algo_(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0),
    bwd_data_algo_(CUDNN_CONVOLUTION_BWD_DATA_ALGO_0),
    bwd_filter_algo_par("") {}


  cudnnConvolutionBwdDataAlgo_t getDataAlgo(){
    return (bwd_data_algo_);
  }


  void SetFwdAlgo(std::string algo) {
    if (!algo.compare("fft")) {
      fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_FFT;
    } else if (!algo.compare("winograd")) {
      fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
    }
  }
  void SetFwdAlgo(cudnnConvolutionFwdAlgo_t fwd_algo) {
    fwd_algo_ = fwd_algo;
  }
  void SetFwdAlgo(const Handle &handle, RunMode mode, int idx,
                  const DataTensor<T> &bottom_desc,
                  const ConvolutionDesc<T> &conv_desc,
                  const DataTensor<T> &top_desc,
                  cudnnConvolutionFwdPreference_t pref) {
    CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
               mode == COMPOSED ?
               handle.GetCudnn(idx) : handle.GetCudnn(),
               bottom_desc.Get(),
               conv_desc.GetFilter(),
               conv_desc.GetConv(),
               top_desc.Get(),
               pref,
               -1,
               &fwd_algo_));
  }
  void FindFwdAlgo(const Handle &handle, RunMode mode, int idx,
                   const DataTensor<T> &bottom_desc,
                   const ConvolutionDesc<T> &conv_desc,
                   const DataTensor<T> &top_desc) {
    cudnnConvolutionFwdAlgoPerf_t *perf_results;
    int *returned_algo_count;
    CUDNN_CALL(cudnnFindConvolutionForwardAlgorithm(
               mode == COMPOSED ?
               handle.GetCudnn(idx) : handle.GetCudnn(),
               bottom_desc.Get(),
               conv_desc.GetFilter(),
               conv_desc.GetConv(),
               top_desc.Get(),
               1, returned_algo_count,
               perf_results));
    if (*returned_algo_count > 0) {
      fwd_algo_ = perf_results->algo;
    } else {
      fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    }
  }
  void SetBwdFilterAlgo(cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo) {
    bwd_filter_algo_ = bwd_filter_algo;
  }
  void SetBwdFilterAlgo(std::string algo) {
    if (algo.empty()) {
        return;
    }
    if (!algo.compare("fft")) {
      bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT;
    } else if (!algo.compare("winograd")) {
      bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD;
    } else if (stoi(algo) == 0) {
      bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
    } else if (stoi(algo) == 1) {
      bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
    } else if (stoi(algo) == 2) {
      bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT;
    } else if (stoi(algo) == 3) {
      bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3;
    } else if (stoi(algo) == 4) {
      bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD;
    } else if (stoi(algo) == 5) {
      bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED;
    } else if (stoi(algo) == 6) {
      bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING;
    } else if (algo.compare("")) {
      std::cout << "Using algo "<< algo << "\n";
      bwd_filter_algo_par = algo;
    }
    LOG(INFO) << "cuDNN algos: " << CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 << " "
             << CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 << " "
             << "FFT:" << CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT << " "
             << CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 << " "
             << "WIN:" << CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD << " "
             << "WIN_NONFUSED:" << CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED << " "
             << "FFT_TILING:" << CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING << "\n";
    LOG(INFO) << "Set Bwd Filter Algo to " << bwd_filter_algo_ << " with " << algo;
  }
  void SetBwdFilterAlgo(const Handle &handle, RunMode mode, int idx,
                        const DataTensor<T> &bottom_desc,
                        const DataTensor<T> &top_desc,
                        const ConvolutionDesc<T> &conv_desc,
                        cudnnConvolutionBwdFilterPreference_t pref) {
     CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(
                mode == COMPOSED ?
                handle.GetCudnn(idx) : handle.GetCudnn(),
                bottom_desc.Get(),
                top_desc.Get(),
                conv_desc.GetConv(),
                conv_desc.GetFilter(),
                pref,
                -1,
                &bwd_filter_algo_));
  }
  void FindBwdFilterAlgo(const Handle &handle, RunMode mode, int idx,
                         const DataTensor<T> &bottom_desc,
                         const ConvolutionDesc<T> &conv_desc,
                         const DataTensor<T> &top_desc) {
    cudnnConvolutionBwdFilterAlgoPerf_t *perf_results;
    int *returned_algo_count;
    CUDNN_CALL(cudnnFindConvolutionBackwardFilterAlgorithm(
               mode == COMPOSED ?
               handle.GetCudnn(idx) : handle.GetCudnn(),
               bottom_desc.Get(),
               top_desc.Get(),
               conv_desc.GetConv(),
               conv_desc.GetFilter(),
               3, returned_algo_count,
               perf_results));
    std::cout << "cuDNN call returned_algo_count :" << *returned_algo_count <<"\n";
    cudnnConvolutionBwdFilterAlgo_t algo = static_cast<cudnnConvolutionBwdFilterAlgo_t>(perf_results->algo);
    std::cout << "cuDNN call result :" << perf_results->algo <<"\n";
    std::cout << "cuDNN casted result :" << algo <<"\n";
    if (*returned_algo_count > 0) {
      bwd_filter_algo_ = perf_results->algo;
    } else {
      bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
    }
  }

  void FindBwdFilterAlgoEx(const Handle &handle, RunMode mode, int idx,
                         const DataTensor<T> &bottom_desc,
                         const ConvolutionDesc<T> &conv_desc,
                         const DataTensor<T> &top_desc,
                         const void *w,
                         const void *dy,
                         void       *dx,
                         void       *workSpace,
                         size_t     workspace_size)
  {
    cudnnConvolutionBwdFilterAlgoPerf_t *perf_results;
    int *returned_algo_count;
    CUDNN_CALL(cudnnFindConvolutionBackwardFilterAlgorithmEx(
               mode == COMPOSED ?
               handle.GetCudnn(idx) : handle.GetCudnn(),
               bottom_desc.Get(), w,
               top_desc.Get(), dy,
               conv_desc.GetConv(),
               conv_desc.GetFilter(), dx,
               1, returned_algo_count,
               perf_results,
               workSpace, workspace_size));
    if (*returned_algo_count > 0) {
      bwd_filter_algo_ = perf_results->algo;
    } else {
      bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
    }
  }




  void SetBwdDataAlgo(cudnnConvolutionBwdDataAlgo_t bwd_data_algo) {
    bwd_data_algo_ = bwd_data_algo;
  }
  void SetBwdDataAlgo(std::string algo) {
    if (algo.empty()) {
        return;
    }
    if (!algo.compare("fft")) {
      bwd_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT;
    } else if (!algo.compare("winograd")) {
      bwd_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD;
    } else if (!algo.compare("0")) {
      bwd_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    } else if (!algo.compare("1")) {
      bwd_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    } else if (!algo.compare("winograd_nonfused")) {
      bwd_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED;
    } else if (!algo.compare("fft_tiling")) {
      bwd_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING;
    }
  }
  void SetBwdDataAlgo(const Handle &handle, RunMode mode, int idx,
                      const DataTensor<T> &bottom_desc,
                      const DataTensor<T> &top_desc,
                      const ConvolutionDesc<T> &conv_desc,
                      cudnnConvolutionBwdDataPreference_t pref) {
     CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm(
                mode == COMPOSED ?
                handle.GetCudnn(idx) : handle.GetCudnn(),
                conv_desc.GetFilter(),
                top_desc.Get(),
                conv_desc.GetConv(),
                bottom_desc.Get(),
                pref,
                -1,
                &bwd_data_algo_));
  }
  void FindBwdDataAlgo(const Handle &handle, RunMode mode, int idx,
                       const DataTensor<T> &bottom_desc,
                       const ConvolutionDesc<T> &conv_desc,
                       const DataTensor<T> &top_desc) {
    cudnnConvolutionBwdDataAlgoPerf_t *perf_results;
    int *returned_algo_count;
    CUDNN_CALL(cudnnFindConvolutionBackwardDataAlgorithm(
               mode == COMPOSED ?
               handle.GetCudnn(idx) : handle.GetCudnn(),
               conv_desc.GetFilter(),
               top_desc.Get(),
               conv_desc.GetConv(),
               bottom_desc.Get(),
               1, returned_algo_count,
               perf_results));
    if (*returned_algo_count > 0) {
      bwd_data_algo_ = perf_results->algo;
    } else {
      bwd_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    }
  }

  cudnnConvolutionFwdAlgo_t GetFwdAlgo() const {
    return fwd_algo_;
  }
  void GetFwdWorkspaceSize(const Handle &handle, RunMode mode, int idx,
                           const DataTensor<T> &bottom_desc,
                           const DataTensor<T> &top_desc,
                           const ConvolutionDesc<T> &conv_desc,
                           size_t *workspace_size) {
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
               mode == COMPOSED ?
               handle.GetCudnn(idx) : handle.GetCudnn(),
               bottom_desc.Get(),
               conv_desc.GetFilter(),
               conv_desc.GetConv(),
               top_desc.Get(),
               fwd_algo_,
               workspace_size));
  }

  std::string GetBwdFilterAlgoParameter() {
    return bwd_filter_algo_par;
  }


  cudnnConvolutionBwdFilterAlgo_t GetBwdFilterAlgo() const {
    return bwd_filter_algo_;
  }
  void GetBwdFilterWorkspaceSize(const Handle &handle, RunMode mode, int idx,
                                 const DataTensor<T> &bottom_desc,
                                 const DataTensor<T> &top_desc,
                                 const ConvolutionDesc<T> &conv_desc,
                                 size_t *workspace_size) {
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
               mode == COMPOSED ?
               handle.GetCudnn(idx) : handle.GetCudnn(),
               bottom_desc.Get(),
               top_desc.Get(),
               conv_desc.GetConv(),
               conv_desc.GetFilter(),
               bwd_filter_algo_,
               workspace_size));
  }
  cudnnConvolutionBwdDataAlgo_t GetBwdDataAlgo() const {
    return bwd_data_algo_;
  }
  void GetBwdDataWorkspaceSize(const Handle &handle, RunMode mode, int idx,
                               const DataTensor<T> &bottom_desc,
                               const DataTensor<T> &top_desc,
                               const ConvolutionDesc<T> &conv_desc,
                               size_t *workspace_size) {
    CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
               mode == COMPOSED ?
               handle.GetCudnn(idx) : handle.GetCudnn(),
               conv_desc.GetFilter(),
               top_desc.Get(),
               conv_desc.GetConv(),
               bottom_desc.Get(),
               bwd_data_algo_,
               workspace_size));
  }

  // Dictionary for storing best bwd convolution algorithms
  std::map<std::string, int> Algo4DataShape;

  void checkAlgo4DataShape(const cudnnTensorDescriptor_t x,
                                            const cudnnTensorDescriptor_t dy,
                                            const cudnnFilterDescriptor_t dw)
                                             // size_t workspace_in_bytes)
  {

    int n,c,h, w, k, nStride, cStride, hStride, wStride;
    cudnnDataType_t datat;
    cudnnTensorFormat_t format;
    std::cout << "Call to checkAlgo4DataShape \n";
    CUDNN_CALL(cudnnGetTensor4dDescriptor(x,
                                          &datat,
                                          &n, &c, &h, &w,
                                          &nStride, &cStride, &hStride, &wStride));
    std::cout << "x shape: " << n <<" "<< c << " " << h << "x" << w << "\n";
    CUDNN_CALL(cudnnGetTensor4dDescriptor(dy,
                                          &datat,
                                          &n, &c, &h, &w,
                                          &nStride, &cStride, &hStride, &wStride));
    std::cout << "dy shape: " << n <<" "<< c << " " << h << "x" << w << "\n";
    CUDNN_CALL(cudnnGetFilter4dDescriptor(dw,
                                          &datat, &format,
                                          &k, &c, &h, &w));
    std::cout << "dw shape: " << k <<" "<< c << " " << h << "x" << w << "\n";
    // std::string hash = std::to_string(x)+"/"+std::to_string(*dy)+"/"+std::to_string(*dw)+"/"+std::to_string(workspace_in_bytes);
    // std::cout << "datashape hash:" << hash << "x:" << x << "dy:" << y << "w:" << w  "\n";
  }


#endif
#ifdef AMD_MIOPEN
 private:
  miopenConvFwdAlgorithm_t fwd_algo_;
  miopenConvBwdWeightsAlgorithm_t bwd_filter_algo_;
  miopenConvBwdDataAlgorithm_t bwd_data_algo_;

 public:
  ConvAlgo()
  : fwd_algo_(miopenConvolutionFwdAlgoGEMM),
    bwd_filter_algo_(miopenConvolutionBwdWeightsAlgoGEMM),
    bwd_data_algo_(miopenConvolutionBwdDataAlgoGEMM) {}

  void SetFwdAlgo(miopenConvFwdAlgorithm_t fwd_algo) {
    fwd_algo_ = fwd_algo;
  }
  void SetFwdAlgo(std::string algo) {
  }
  void FindFwdAlgo(const Handle &handle, RunMode mode, int idx,
                   const DataTensor<T> &bottom_desc,
                   const ConvolutionDesc<T> &conv_desc,
                   const DataTensor<T> &top_desc,
                   const void *x,
                   const void *w,
                   void *y,
                   void *workspace,
                   size_t workspace_size) {
    fwd_algo_ = miopenConvolutionFwdAlgoGEMM;
#ifdef NOSIM
    miopenConvAlgoPerf_t perf_results;
    int returned_algo_count;
    MIOPEN_CALL(miopenFindConvolutionForwardAlgorithm(
                mode == COMPOSED ?
                handle.GetMIOpen(idx) : handle.GetMIOpen(),
                bottom_desc.Get(), x,
                conv_desc.GetFilter(), w,
                conv_desc.GetConv(),
                top_desc.Get(), y,
                1, &returned_algo_count,
                &perf_results, workspace, workspace_size, false));
    if (returned_algo_count > 0) {
      fwd_algo_ = perf_results.fwd_algo;
    }
#endif
  }
  void SetBwdFilterAlgo(miopenConvBwdWeightsAlgorithm_t bwd_filter_algo) {
    bwd_filter_algo_ = bwd_filter_algo;
  }
  void SetBwdFilterAlgo(std::string algo) {
  }
  void FindBwdFilterAlgo(const Handle &handle, RunMode mode, int idx,
                         const DataTensor<T> &bottom_desc,
                         const ConvolutionDesc<T> &conv_desc,
                         const DataTensor<T> &top_desc,
                         const void *x,
                         const void *dy,
                         void *dw,
                         void *workspace,
                         size_t workspace_size) {
    bwd_filter_algo_ = miopenConvolutionBwdWeightsAlgoGEMM;
#ifdef NOSIM
    miopenConvAlgoPerf_t perf_results;
    int returned_algo_count;
    MIOPEN_CALL(miopenFindConvolutionBackwardWeightsAlgorithm(
                mode == COMPOSED ?
                handle.GetMIOpen(idx) : handle.GetMIOpen(),
                top_desc.Get(), dy,
                bottom_desc.Get(), x,
                conv_desc.GetConv(),
                conv_desc.GetFilter(), dw,
                1, &returned_algo_count,
                &perf_results, workspace, workspace_size, false));
    if (returned_algo_count > 0) {
      bwd_filter_algo_ = perf_results.bwd_weights_algo;
    }
#endif
  }
  void SetBwdDataAlgo(miopenConvBwdDataAlgorithm_t bwd_data_algo) {
    bwd_data_algo_ = bwd_data_algo;
  }
  void SetBwdDataAlgo(std::string algo) {
  }
  void FindBwdDataAlgo(const Handle &handle, RunMode mode, int idx,
                       const DataTensor<T> &bottom_desc,
                       const ConvolutionDesc<T> &conv_desc,
                       const DataTensor<T> &top_desc,
                       const void *dy,
                       const void *w,
                       void *dx,
                       void *workspace,
                       size_t workspace_size) {
    bwd_data_algo_ = miopenConvolutionBwdDataAlgoGEMM;
#ifdef NOSIM
    miopenConvAlgoPerf_t perf_results;
    int returned_algo_count;
    MIOPEN_CALL(miopenFindConvolutionBackwardDataAlgorithm(
                mode == COMPOSED ?
                handle.GetMIOpen(idx) : handle.GetMIOpen(),
                top_desc.Get(), dy,
                conv_desc.GetFilter(), w,
                conv_desc.GetConv(),
                bottom_desc.Get(), dx,
                1, &returned_algo_count,
                &perf_results, workspace, workspace_size, false));
    if (returned_algo_count > 0) {
      bwd_data_algo_ = perf_results.bwd_data_algo;
    }
#endif
  }

  miopenConvFwdAlgorithm_t GetFwdAlgo() const {
    return fwd_algo_;
  }
  void GetFwdWorkspaceSize(const Handle &handle, RunMode mode, int idx,
                           const DataTensor<T> &bottom_desc,
                           const DataTensor<T> &top_desc,
                           const ConvolutionDesc<T> &conv_desc,
                           size_t *workspace_size) {
    MIOPEN_CALL(miopenConvolutionForwardGetWorkSpaceSize(
                mode == COMPOSED ?
                handle.GetMIOpen(idx) : handle.GetMIOpen(),
                conv_desc.GetFilter(),
                bottom_desc.Get(),
                conv_desc.GetConv(),
                top_desc.Get(),
                workspace_size));
  }
  miopenConvBwdWeightsAlgorithm_t GetBwdFilterAlgo() const {
    return bwd_filter_algo_;
  }
  void GetBwdFilterWorkspaceSize(const Handle &handle, RunMode mode, int idx,
                                 const DataTensor<T> &bottom_desc,
                                 const DataTensor<T> &top_desc,
                                 const ConvolutionDesc<T> &conv_desc,
                                 size_t *workspace_size) {
    MIOPEN_CALL(miopenConvolutionBackwardWeightsGetWorkSpaceSize(
                mode == COMPOSED ?
                handle.GetMIOpen(idx) : handle.GetMIOpen(),
                top_desc.Get(),
                bottom_desc.Get(),
                conv_desc.GetConv(),
                conv_desc.GetFilter(),
                workspace_size));
  }
  miopenConvBwdDataAlgorithm_t GetBwdDataAlgo() const {
    return bwd_data_algo_;
  }
  void GetBwdDataWorkspaceSize(const Handle &handle, RunMode mode, int idx,
                               const DataTensor<T> &bottom_desc,
                               const DataTensor<T> &top_desc,
                               const ConvolutionDesc<T> &conv_desc,
                               size_t *workspace_size) {
    MIOPEN_CALL(miopenConvolutionBackwardDataGetWorkSpaceSize(
                mode == COMPOSED ?
                handle.GetMIOpen(idx) : handle.GetMIOpen(),
                top_desc.Get(),
                conv_desc.GetFilter(),
                conv_desc.GetConv(),
                bottom_desc.Get(),
                workspace_size));
  }
#endif
};

// Profiling marker
inline void ProfilerStart(const Handle &handle, RunMode mode, int idx,
                          Timer *timer, const std::string &layer) {
#ifdef NVIDIA_CUDNN
  cudaProfilerStart();
#endif
#ifdef AMD_MIOPEN
  miopenEnableProfiling(mode == COMPOSED ?
                        handle.GetMIOpen(idx) : handle.GetMIOpen(), true);
#endif
  timer->Start(layer + "_" + std::to_string(idx));
}
inline void ProfilerStop(const Handle &handle, RunMode mode, int idx,
                         Timer *timer, const std::string &layer) {
#ifdef NVIDIA_CUDNN
  cudaProfilerStop();
  CUDA_CALL(cudaDeviceSynchronize());
#endif
#ifdef AMD_MIOPEN
  miopenEnableProfiling(mode == COMPOSED ?
                        handle.GetMIOpen(idx) : handle.GetMIOpen(), false);
  HIP_CALL(hipDeviceSynchronize());
#endif
  timer->Stop(layer + "_" + std::to_string(idx));
}

} // namespace dnnmark

#endif // CORE_INCLUDE_DNN_UTILITY_H_
