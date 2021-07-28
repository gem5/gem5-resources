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

#ifndef CORE_INCLUDE_DNN_LAYER_H_
#define CORE_INCLUDE_DNN_LAYER_H_

#include <vector>
#include <glog/logging.h>
#include "common.h"
#include "dnn_param.h"
#include "dnn_utility.h"
#include "data_manager.h"

namespace dnnmark {

// Forward declaration
template <typename T> class DNNMark;

template <typename T>
class Layer {
 protected:
  // DNNMark pointer
  DNNMark<T> *p_dnnmark_;

  bool has_learnable_params_;
  LayerType type_;
  int layer_id_;
  std::string layer_name_;
  std::string previous_layer_name_;
  DataDim input_dim_;
  DataDim output_dim_;
  DataTensor<T> bottom_desc_;
  DataTensor<T> top_desc_;
  DataManager<T> *data_manager_;

  int num_bottoms_;
  // Layer bottom data
  std::vector<Data<T> *> bottoms_;
  std::vector<int> bottom_chunk_ids_;
  std::vector<Data<T> *> bottom_diffs_;
  std::vector<int> bottom_diff_chunk_ids_;

  int num_tops_;
  // Layer top data
  std::vector<Data<T> *> tops_;
  std::vector<int> top_chunk_ids_;
  std::vector<Data<T> *> top_diffs_;
  std::vector<int> top_diff_chunk_ids_;
 public:
  Layer(DNNMark<T> *p_dnnmark)
  : p_dnnmark_(p_dnnmark),
    layer_id_(0), has_learnable_params_(false),
    input_dim_(), bottom_desc_(),
    output_dim_(), top_desc_(),
    num_bottoms_(1), num_tops_(1) {
    data_manager_ = DataManager<T>::GetInstance();
  }
  ~Layer() {
  }
  DataDim *getInputDim() { return &input_dim_; }
  DataDim *getOutputDim() { return &output_dim_; }
  void setLayerName(const char *layer_name) {
    layer_name_.assign(layer_name);
    // Debug info
    LOG(INFO) << "Layer name: " << layer_name_;
  }
  void setPrevLayerName(const char *previous_layer_name) {
    previous_layer_name_.assign(previous_layer_name);
    // Debug info
    LOG(INFO) << "Previous layer: " << previous_layer_name_;
  }
  void setLayerId(int layer_id) { layer_id_ = layer_id; }
  int getLayerId() { return layer_id_; }
  void setLayerType(LayerType type) { type_ = type; }
  LayerType getLayerType() { return type_; }

  // Functions that used to communicate with its successor layer
  int getNumTops() { return num_tops_; }
  int getTopChunkID(int index) { return top_chunk_ids_[index]; }
  int getTopDiffChunkID(int index) { return top_diff_chunk_ids_[index]; }
  int getTopDimN() { return output_dim_.n_; }
  int getTopDimC() { return output_dim_.c_; }
  int getTopDimH() { return output_dim_.h_; }
  int getTopDimW() { return output_dim_.w_; }

  // Base layer setup function
  virtual void Setup() {
    if (input_dim_.n_ != 0 && input_dim_.c_ != 0 &&
        input_dim_.h_ != 0 && input_dim_.w_ != 0) {
      // Debug info
      LOG(INFO) << "Bottom dimension: "
                << "N: " << input_dim_.n_ << " "
                << "C: " << input_dim_.c_ << " "
                << "H: " << input_dim_.h_ << " "
                << "W: " << input_dim_.w_;
      //
      // Standalone mode or the first layer in composed mode
      //
      if (p_dnnmark_->getRunMode() == COMPOSED)
        if (previous_layer_name_.compare("null")!=0) {
          LOG(INFO) << "Problems with "<< layer_name_ << " <- "
                    << previous_layer_name_ << " "
                    << previous_layer_name_.compare("null");
          LOG(FATAL) << "When composed as composed mode, the first layer "
                     << "should set data dimension "
                     << "and have a <null> previous layer";
        }
      // Set bottom tensor
      bottom_desc_.Set(input_dim_.n_,
                       input_dim_.c_,
                       input_dim_.h_,
                       input_dim_.w_);

      // Prepare bottom data
      int bottom_size = input_dim_.n_ *
                        input_dim_.c_ *
                        input_dim_.h_ *
                        input_dim_.w_;
      for (int i = 0; i < num_bottoms_; i++) {
        bottom_chunk_ids_.push_back(
          data_manager_->CreateData(bottom_size));
        bottoms_.push_back(
          data_manager_->GetData(bottom_chunk_ids_[i]));
        bottom_diff_chunk_ids_.push_back(
          data_manager_->CreateData(bottom_size));
        bottom_diffs_.push_back(
          data_manager_->GetData(bottom_diff_chunk_ids_[i]));
      }
    } else {
      //
      // Composed mode
      //
      CHECK_EQ(p_dnnmark_->getRunMode(), COMPOSED);
      if (p_dnnmark_->isLayerExist(previous_layer_name_)) {
        Layer<T> *previous_layer =
          p_dnnmark_->GetLayerByName(previous_layer_name_);
        num_bottoms_ = previous_layer->getNumTops();
        num_tops_ = num_bottoms_;
        input_dim_.n_ = previous_layer->getTopDimN();
        input_dim_.c_ = previous_layer->getTopDimC();
        input_dim_.h_ = previous_layer->getTopDimH();
        input_dim_.w_ = previous_layer->getTopDimW();

        // Debug info
        LOG(INFO) << "Bottom dimension: "
                  << "N: " << input_dim_.n_ << " "
                  << "C: " << input_dim_.c_ << " "
                  << "H: " << input_dim_.h_ << " "
                  << "W: " << input_dim_.w_;

        // Set bottom tensor
        bottom_desc_.Set(input_dim_.n_,
                         input_dim_.c_,
                         input_dim_.h_,
                         input_dim_.w_);
        for (int i = 0; i < num_bottoms_; i++) {
          bottom_chunk_ids_.push_back(
            previous_layer->getTopChunkID(i));
          bottoms_.push_back(
            data_manager_->GetData(bottom_chunk_ids_[i]));
          bottom_diff_chunk_ids_.push_back(
            previous_layer->getTopDiffChunkID(i));
          bottom_diffs_.push_back(
            data_manager_->GetData(bottom_diff_chunk_ids_[i]));
        }
      } else {
        LOG(FATAL) << "Wrong previous layer name!!!";
      }
    }
  }

  virtual void ForwardPropagation() {}
  virtual void BackwardPropagation() {}

};

} // namespace dnnmark

#endif // CORE_INCLUDE_DNN_LAYER_H_
