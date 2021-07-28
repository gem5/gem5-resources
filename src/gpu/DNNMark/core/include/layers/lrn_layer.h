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

#ifndef CORE_INCLUDE_LAYERS_LRN_LAYER_H_
#define CORE_INCLUDE_LAYERS_LRN_LAYER_H_

#include "dnn_layer.h"

namespace dnnmark {

template <typename T>
class LRNLayer : public Layer<T> {
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
  LRNParam lrn_param_;

  // LRN specific descriptor
  LRNDesc<T> desc_;

  // Workspace
  size_t workspace_size_;
  Data<T> *workspace_;
  int workspace_id_;

 public:
  LRNLayer(DNNMark<T> *p_dnnmark)
  : Layer<T>(p_dnnmark),
    lrn_param_(), desc_() {
    workspace_size_ = 0;
  }

  LRNParam *getLRNParam() { return &lrn_param_; }

  void Setup() {
    // Set up indispensable stuff here
    Layer<T>::Setup();

    // Set lrning related descriptors
    desc_.Set(lrn_param_);

    // Set up lrning related data
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

    // Allocate workspace
    desc_.GetWorkspaceSize(top_desc_, &workspace_size_);
    if (workspace_size_ > 0) {
      workspace_id_ = data_manager_->CreateData(workspace_size_);
      workspace_ = data_manager_->GetData(workspace_id_);
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

    // lrn forward computation
    ProfilerStart(*(p_dnnmark_->GetHandle()), p_dnnmark_->getRunMode(),
                  layer_id_, p_dnnmark_->GetTimer(), "LrnFwd");
    for (int i = 0; i < num_bottoms_; i++) {
      dnnmarkLRNForward(
             *(p_dnnmark_->GetHandle()),
             p_dnnmark_->getRunMode(), layer_id_,
             desc_,
             lrn_param_,
             DataType<T>::one,
             bottom_desc_, bottoms_[i]->Get(),
             DataType<T>::zero,
             top_desc_, tops_[i]->Get(), workspace_);
    }
    ProfilerStop(*(p_dnnmark_->GetHandle()), p_dnnmark_->getRunMode(),
                  layer_id_, p_dnnmark_->GetTimer(), "LrnFwd");

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

    // lrn backward computation
    ProfilerStart(*(p_dnnmark_->GetHandle()), p_dnnmark_->getRunMode(),
                  layer_id_, p_dnnmark_->GetTimer(), "LrnBwd");
    for (int i = 0; i < num_tops_; i++) {
      dnnmarkLRNBackward(
             *(p_dnnmark_->GetHandle()),
             p_dnnmark_->getRunMode(), layer_id_,
             desc_,
             lrn_param_,
             DataType<T>::one,
             top_desc_, tops_[i]->Get(), top_diffs_[i]->Get(),
             DataType<T>::zero,
             bottom_desc_,
             bottoms_[i]->Get(),
             bottom_diffs_[i]->Get(),
             workspace_);
    }
    ProfilerStop(*(p_dnnmark_->GetHandle()), p_dnnmark_->getRunMode(),
                  layer_id_, p_dnnmark_->GetTimer(), "LrnBwd");
  }

};

} // namespace dnnmark

#endif // CORE_INCLUDE_LAYERS_LRN_LAYER_H_
