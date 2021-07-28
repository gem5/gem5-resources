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

#ifndef CORE_INCLUDE_LAYERS_FC_LAYER_H_
#define CORE_INCLUDE_LAYERS_FC_LAYER_H_

#include "dnn_layer.h"

namespace dnnmark {

template <typename T>
class FullyConnectedLayer : public Layer<T> {
  // using declaration for calling member from base class
  using Layer<T>::p_dnnmark_;
  using Layer<T>::layer_id_;
  using Layer<T>::previous_layer_name_;
  using Layer<T>::input_dim_;
  using Layer<T>::output_dim_;
  using Layer<T>::bottom_desc_;
  using Layer<T>::top_desc_;
  using Layer<T>::data_manager_;

  using Layer<T>::num_bottoms_;
  using Layer<T>::bottoms_;
  using Layer<T>::bottom_chunk_ids_;
  using Layer<T>::bottom_diffs_;
  using Layer<T>::bottom_diff_chunk_ids_;

  using Layer<T>::num_tops_;
  using Layer<T>::tops_;
  using Layer<T>::top_chunk_ids_;
  using Layer<T>::top_diffs_;
  using Layer<T>::top_diff_chunk_ids_;

 private:
  FullyConnectedParam fc_param_;

  // Weights demension
  int num_rows_weights_;
  int num_cols_weights_;
  T scale_alpha_;
  T scale_beta_;

  // Layer weights
  Data<T> *weights_;
  int weights_chunk_id_;
  Data<T> *weights_diff_;
  int weights_diff_chunk_id_;

 public:
  FullyConnectedLayer(DNNMark<T> *p_dnnmark)
  : Layer<T>(p_dnnmark),
    fc_param_() {
    Layer<T>::has_learnable_params_ = true;
  }

  FullyConnectedParam *getFullyConnectedParam() { return &fc_param_; }

  void Setup() {
    // Set up indispensable stuff here
    Layer<T>::Setup();

    // Set up fcing related data
    if (input_dim_.n_ != 0 && input_dim_.c_ != 0 &&
        input_dim_.h_ != 0 && input_dim_.w_ != 0) {
      //
      // Standalone mode
      //

      // Compute dimension of output data
      ComputeOutputDim();

      // Set top tensor
      top_desc_.Set(output_dim_.n_,
                    output_dim_.c_,
                    output_dim_.h_,
                    output_dim_.w_);

      // Prepare top data
      int top_size = output_dim_.n_ *
                     output_dim_.c_ *
                     output_dim_.h_ *
                     output_dim_.w_;
      for (int i = 0; i < num_tops_; i++) {
        top_chunk_ids_.push_back(
          data_manager_->CreateData(top_size));
        tops_.push_back(
          data_manager_->GetData(top_chunk_ids_[i]));
        top_diff_chunk_ids_.push_back(
          data_manager_->CreateData(top_size));
        top_diffs_.push_back(
          data_manager_->GetData(top_diff_chunk_ids_[i]));
      }
    }

    // Only one set of weights is considered
    num_rows_weights_ = input_dim_.c_ *
                           input_dim_.h_ *
                           input_dim_.w_;
    num_cols_weights_ = fc_param_.output_num_;
    int weights_size = num_rows_weights_ * num_cols_weights_;
    weights_chunk_id_ = data_manager_->CreateData(weights_size);
    weights_ = data_manager_->GetData(weights_chunk_id_);
    weights_diff_chunk_id_ =
      data_manager_->CreateData(weights_size);
    weights_diff_ = data_manager_->GetData(weights_diff_chunk_id_);

    // Fill the weight data
    weights_->Filler();

    scale_alpha_ = (T)1.0;
    scale_beta_ = (T)0.0;
  }

  void ComputeOutputDim() {
    output_dim_.n_ = input_dim_.n_;
    output_dim_.c_ = fc_param_.output_num_;
    output_dim_.h_ = 1;
    output_dim_.w_ = 1;
  }

  void ForwardPropagation() {
    if (p_dnnmark_->getRunMode() == STANDALONE ||
        !previous_layer_name_.compare("null")) {
      // Fill the bottom data
      for (int i = 0; i < num_bottoms_; i++) {
        bottoms_[i]->Filler();
      }
    }

    // Prepare CuBLAS parameters
    int M = fc_param_.output_num_;
    int N = input_dim_.n_;;
    int K = num_rows_weights_;
    int lda = K;
    int ldb = K;
    int ldc = M;
    bool is_a_transpose = true;
    bool is_b_transpose = false;

    // Fully connected forward computation
    ProfilerStart(*(p_dnnmark_->GetHandle()), p_dnnmark_->getRunMode(),
                  layer_id_, p_dnnmark_->GetTimer(), "FcFwd");
    for (int i = 0; i < num_bottoms_; i++) {
      // Y = T(W) * X
      dnnmarkGEMM(*(p_dnnmark_->GetHandle()),
                  p_dnnmark_->getRunMode(), layer_id_,
                  is_a_transpose, is_b_transpose,
                  M, N, K,
                  &scale_alpha_,
                  weights_->Get(), lda,
                  bottoms_[i]->Get(), ldb,
                  &scale_beta_,
                  tops_[i]->Get(), ldc);
    }
    ProfilerStop(*(p_dnnmark_->GetHandle()), p_dnnmark_->getRunMode(),
                  layer_id_, p_dnnmark_->GetTimer(), "FcFwd");

  }

  void BackwardPropagation() {
    if (p_dnnmark_->getRunMode() == STANDALONE ||
        !previous_layer_name_.compare("null")) {
      // Fill the top diff data
      for (int i = 0; i < num_tops_; i++) {
        top_diffs_[i]->Filler();
      }
      // Fill the bottom data
      for (int i = 0; i < num_bottoms_; i++) {
        bottoms_[i]->Filler();
      }
    }

    // Prepare CuBLAS parameters for calculating d(W)
    int M = num_rows_weights_;
    int N = fc_param_.output_num_;
    int K = input_dim_.n_;
    int lda = M;
    int ldb = N;
    int ldc = M;
    bool is_a_transpose = false;
    bool is_b_transpose = true;

    // Fully connected backward weights computation
    ProfilerStart(*(p_dnnmark_->GetHandle()), p_dnnmark_->getRunMode(),
                  layer_id_, p_dnnmark_->GetTimer(), "FcBwdFilter");
    for (int i = 0; i < num_tops_; i++) {
      // d(W) = X * T(d(Y))
      dnnmarkGEMM(*(p_dnnmark_->GetHandle()),
                  p_dnnmark_->getRunMode(), layer_id_,
                  is_a_transpose, is_b_transpose,
                  M, N, K,
                  &scale_alpha_,
                  bottoms_[i]->Get(), lda,
                  top_diffs_[i]->Get(), ldb,
                  &scale_beta_,
                  weights_diff_->Get(), ldc);
    }
    ProfilerStop(*(p_dnnmark_->GetHandle()), p_dnnmark_->getRunMode(),
                  layer_id_, p_dnnmark_->GetTimer(), "FcBwdFilter");

    M = num_rows_weights_;
    N = input_dim_.n_;
    K = fc_param_.output_num_;
    lda = M;
    ldb = K;
    ldc = M;
    is_a_transpose = false;
    is_b_transpose = false;

    // Fully connected backward data computation
    ProfilerStart(*(p_dnnmark_->GetHandle()), p_dnnmark_->getRunMode(),
                  layer_id_, p_dnnmark_->GetTimer(), "FcBwdData");
    for (int i = 0; i < num_tops_; i++) {
      // d(X) = W * d(Y)
      dnnmarkGEMM(*(p_dnnmark_->GetHandle()),
                  p_dnnmark_->getRunMode(), layer_id_,
                  is_a_transpose, is_b_transpose,
                  M, N, K,
                  &scale_alpha_,
                  weights_->Get(), lda,
                  top_diffs_[i]->Get(), ldb,
                  &scale_beta_,
                  bottom_diffs_[i]->Get(), ldc);
    }
    ProfilerStop(*(p_dnnmark_->GetHandle()), p_dnnmark_->getRunMode(),
                  layer_id_, p_dnnmark_->GetTimer(), "FcBwdData");
  }

};

} // namespace dnnmark

#endif // CORE_INCLUDE_LAYERS_FC_LAYER_H_
