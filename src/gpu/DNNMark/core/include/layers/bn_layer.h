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

#ifndef CORE_INCLUDE_LAYERS_BN_LAYER_H_
#define CORE_INCLUDE_LAYERS_BN_LAYER_H_

#include "dnn_layer.h"

namespace dnnmark {

template <typename T>
class BatchNormLayer : public Layer<T> {
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
  BatchNormParam bn_param_;
  DataTensor<T> bn_specifics_desc_;
  int bn_specifics_size_;
  Data<T> *bn_scale_;
  int bn_scale_chunk_id_;
  Data<T> *bn_scale_diffs_;
  int bn_scale_diffs_chunk_id_;
  Data<T> *bn_bias_;
  int bn_bias_chunk_id_;
  Data<T> *bn_bias_diffs_;
  int bn_bias_diffs_chunk_id_;
  Data<T> *bn_running_mean_;
  int bn_running_mean_chunk_id_;
  Data<T> *bn_running_inv_variance_;
  int bn_running_inv_variance_chunk_id_;
  Data<T> *bn_saved_mean_;
  int bn_saved_mean_chunk_id_;
  Data<T> *bn_saved_inv_variance_;
  int bn_saved_inv_variance_chunk_id_;

  // Work around for MIOpen library
  T alpha_;
  T beta_;

 public:
  BatchNormLayer(DNNMark<T> *p_dnnmark)
  : Layer<T>(p_dnnmark),
    bn_param_() {
      alpha_ = 1.0;
      beta_ = 0.0;
  }

  BatchNormParam *getBatchNormParam() { return &bn_param_; }

  void Setup() {
    // Set up indispensable stuff here
    Layer<T>::Setup();

    // Set up batch normalization related data
    if(bn_param_.epsilon_ < BN_MIN_EPSILON) {
      LOG(FATAL) << "The value of epsilon cannot be less than BN_MIN_EPSILON."
                 << "This value is defined as " << BN_MIN_EPSILON;
    }
    if((BatchNormMode)(bn_param_.mode_) == PerActivation) {
      bn_specifics_desc_.Set(1, input_dim_.c_, input_dim_.h_, input_dim_.w_);
      bn_specifics_size_ = input_dim_.c_ * input_dim_.h_ * input_dim_.w_;
    }
    else if ((BatchNormMode)(bn_param_.mode_) == Spatial) {
      bn_specifics_desc_.Set(1, input_dim_.c_, 1, 1);
      bn_specifics_size_ = input_dim_.c_;
    }

    //Initialize bn_scale_, bn_scale_diffs_, bn_bias_, bn_bias_diffs_, bn_running_mean_, and bn_running_inv_variance_
    bn_scale_chunk_id_ = data_manager_->CreateData(bn_specifics_size_);
    bn_scale_ = data_manager_->GetData(bn_scale_chunk_id_);
    bn_scale_diffs_chunk_id_ = data_manager_->CreateData(bn_specifics_size_);
    bn_scale_diffs_ = data_manager_->GetData(bn_scale_diffs_chunk_id_);
    bn_bias_chunk_id_ = data_manager_->CreateData(bn_specifics_size_);
    bn_bias_ = data_manager_->GetData(bn_bias_chunk_id_);
    bn_bias_diffs_chunk_id_ = data_manager_->CreateData(bn_specifics_size_);
    bn_bias_diffs_ = data_manager_->GetData(bn_bias_diffs_chunk_id_);
    bn_running_mean_chunk_id_ = data_manager_->CreateData(bn_specifics_size_);
    bn_running_mean_ = data_manager_->GetData(bn_running_mean_chunk_id_);
    bn_running_inv_variance_chunk_id_ = data_manager_->CreateData(bn_specifics_size_);
    bn_running_inv_variance_ = data_manager_->GetData(bn_running_inv_variance_chunk_id_);

    bn_scale_->Filler();
    bn_bias_->Filler();
    bn_running_mean_->Filler();
    bn_running_inv_variance_->Filler();

    //All of these tensors use the bn_specifics_ tensor descriptor
    if(bn_param_.save_intermediates_) {
      bn_saved_mean_chunk_id_ = data_manager_->CreateData(bn_specifics_size_);
      bn_saved_mean_ = data_manager_->GetData(bn_saved_mean_chunk_id_);
      bn_saved_inv_variance_chunk_id_ = data_manager_->CreateData(bn_specifics_size_);
      bn_saved_inv_variance_ = data_manager_->GetData(bn_saved_inv_variance_chunk_id_);

      bn_saved_mean_->Filler();
      bn_saved_inv_variance_->Filler();
    }
    else {
      bn_saved_mean_ = nullptr;
      bn_saved_inv_variance_ = nullptr;
    }

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
  }

  void ComputeOutputDim() {
    output_dim_.n_ = input_dim_.n_;
    output_dim_.c_ = input_dim_.c_;
    output_dim_.h_ = input_dim_.h_;
    output_dim_.w_ = input_dim_.w_;
  }

  void ForwardPropagation() {
    if (p_dnnmark_->getRunMode() == STANDALONE ||
        !previous_layer_name_.compare("null")) {
      // Fill the bottom data
      for (int i = 0; i < num_bottoms_; i++) {
        bottoms_[i]->Filler();
      }
    }

    // Batch normalization forward computation
    ProfilerStart(*(p_dnnmark_->GetHandle()), p_dnnmark_->getRunMode(),
                  layer_id_, p_dnnmark_->GetTimer(), "BnFwd");
    for (int i = 0; i < num_bottoms_; i++) {
      dnnmarkBatchNormalizationForwardTraining(
              *(p_dnnmark_->GetHandle()),
              p_dnnmark_->getRunMode(), layer_id_,
              bn_param_,
              //DataType<T>::one,
              //DataType<T>::zero,
              &alpha_,
              &beta_,
              bottom_desc_, bottoms_[i]->Get(),
              top_desc_, tops_[i]->Get(),
              bn_specifics_desc_,
              bn_scale_->Get(),
              bn_bias_->Get(),
              bn_param_.exp_avg_factor_,
              bn_running_mean_->Get(),
              bn_running_inv_variance_->Get(),
              bn_param_.epsilon_,
              bn_saved_mean_->Get(),
              bn_saved_inv_variance_->Get()
              );
    }
    ProfilerStop(*(p_dnnmark_->GetHandle()), p_dnnmark_->getRunMode(),
                  layer_id_, p_dnnmark_->GetTimer(), "BnFwd");
  }

  void BackwardPropagation() {
    if (p_dnnmark_->getRunMode() == STANDALONE ||
        !previous_layer_name_.compare("null")) {
      // Fill the top and top diff data
      for (int i = 0; i < num_tops_; i++) {
        tops_[i]->Filler();
        top_diffs_[i]->Filler();
      }
      // Fill the bottom data
      for (int i = 0; i < num_bottoms_; i++) {
        bottoms_[i]->Filler();
      }
    }

    // Batch normalization backward computation
    ProfilerStart(*(p_dnnmark_->GetHandle()), p_dnnmark_->getRunMode(),
                  layer_id_, p_dnnmark_->GetTimer(), "BnBwd");
    for (int i = 0; i < num_tops_; i++) {
      dnnmarkBatchNormalizationBackward(
              *(p_dnnmark_->GetHandle()),
              p_dnnmark_->getRunMode(), layer_id_,
              bn_param_,
              //DataType<T>::one,
              //DataType<T>::zero,
              //DataType<T>::one,
              //DataType<T>::zero,
              &alpha_,
              &beta_,
              &alpha_,
              &beta_,
              bottom_desc_, bottoms_[i]->Get(), bottom_diffs_[i]->Get(),
              top_desc_, top_diffs_[i]->Get(),
              bn_specifics_desc_,
              bn_scale_->Get(),
              bn_scale_diffs_->Get(),
              bn_bias_diffs_->Get(),
              bn_param_.epsilon_,
              bn_saved_mean_->Get(),
              bn_saved_inv_variance_->Get()
              );
    }
    ProfilerStop(*(p_dnnmark_->GetHandle()), p_dnnmark_->getRunMode(),
                  layer_id_, p_dnnmark_->GetTimer(), "BnBwd");
  }

};

} // namespace dnnmark

#endif // CORE_INCLUDE_LAYERS_BN_LAYER_H_
