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

#ifdef NVIDIA_CUDNN
#include "cudnn.h"
#endif
#ifdef AMD_MIOPEN
#include <miopen/miopen.h>
#endif

#include "dnnmark.h"
#include "data_png.h"

namespace dnnmark {

//
// DNNMark class definition
//

template <typename T>
DNNMark<T>::DNNMark(const std::string &mmap_file)
: run_mode_(NONE), handle_(), timer_(), num_layers_added_(0)
{
  PseudoNumGenerator::CreateInstance(mmap_file);
}

template <typename T>
DNNMark<T>::DNNMark(int num_layers, const std::string &mmap_file)
: run_mode_(NONE), handle_(num_layers), timer_(), num_layers_added_(0)
{
  PseudoNumGenerator::CreateInstance(mmap_file);
}
template <typename T>
void DNNMark<T>::SetLayerParams(LayerType layer_type,
                    int current_layer_id,
                    const std::string &var,
                    const std::string &val) {
  DataDim *input_dim;
  ConvolutionParam *conv_param;
  PoolingParam *pool_param;
  LRNParam *lrn_param;
  ActivationParam *activation_param;
  FullyConnectedParam *fc_param;
  SoftmaxParam *softmax_param;
  BatchNormParam *bn_param;
  DropoutParam *dropout_param;
  BypassParam *bypass_param;
  CHECK_GT(num_layers_added_, 0);

  switch(layer_type) {
    case CONVOLUTION: {
      // Obtain the data dimension and parameters variable
      // within specified layer
      input_dim = std::dynamic_pointer_cast<ConvolutionLayer<T>>
                  (layers_map_[current_layer_id])->getInputDim();
      conv_param = std::dynamic_pointer_cast<ConvolutionLayer<T>>
                   (layers_map_[current_layer_id])->getConvParam();

      if(isKeywordExist(var, data_config_keywords))
        break;

      SetupConvParam(var, val, conv_param);
      break;
    } // End of case CONVOLUTION
    case POOLING: {
      // Obtain the data dimension and parameters variable within layer class
      input_dim = std::dynamic_pointer_cast<PoolingLayer<T>>
                  (layers_map_[current_layer_id])->getInputDim();
      pool_param = std::dynamic_pointer_cast<PoolingLayer<T>>
                   (layers_map_[current_layer_id])->getPoolParam();

      if(isKeywordExist(var, data_config_keywords))
        break;

      SetupPoolingParam(var, val, pool_param);
      break;
    } // End of case POOLING
    case LRN: {
      // Obtain the data dimension and parameters variable within layer class
      input_dim = std::dynamic_pointer_cast<LRNLayer<T>>
                  (layers_map_[current_layer_id])->getInputDim();
      lrn_param = std::dynamic_pointer_cast<LRNLayer<T>>
                   (layers_map_[current_layer_id])->getLRNParam();

      if(isKeywordExist(var, data_config_keywords))
        break;

      SetupLrnParam(var, val, lrn_param);
      break;
    } // End of case LRN
    case ACTIVATION: {
      // Obtain the data dimension and parameters variable within layer class
      input_dim = std::dynamic_pointer_cast<ActivationLayer<T>>
                  (layers_map_[current_layer_id])->getInputDim();
      activation_param = std::dynamic_pointer_cast<ActivationLayer<T>>
                   (layers_map_[current_layer_id])->getActivationParam();

      if(isKeywordExist(var, data_config_keywords))
        break;

      // Process all the keywords in config
      SetupActivationParam(var, val, activation_param);
      break;
    } // End of case ACTIVATION
    case FC: {
      // Obtain the data dimension and parameters variable within layer class
      input_dim = std::dynamic_pointer_cast<FullyConnectedLayer<T>>
                  (layers_map_[current_layer_id])->getInputDim();
      fc_param = std::dynamic_pointer_cast<FullyConnectedLayer<T>>
                 (layers_map_[current_layer_id])->getFullyConnectedParam();

      if(isKeywordExist(var, data_config_keywords))
        break;

      SetupFcParam(var, val, fc_param);
      break;
    } // End of case FC
    case SOFTMAX: {
      // Obtain the data dimension and parameters variable within layer class
      input_dim = std::dynamic_pointer_cast<SoftmaxLayer<T>>
                  (layers_map_[current_layer_id])->getInputDim();
      softmax_param = std::dynamic_pointer_cast<SoftmaxLayer<T>>
                 (layers_map_[current_layer_id])->getSoftmaxParam();

      if(isKeywordExist(var, data_config_keywords))
        break;

      SetupSoftmaxParam(var, val, softmax_param);
      break;
    } // End of case SOFTMAX
    case BN: {
      // Obtain the data dimension and parameters variable within layer class
      input_dim = std::dynamic_pointer_cast<BatchNormLayer<T>>
                  (layers_map_[current_layer_id])->getInputDim();
      bn_param = std::dynamic_pointer_cast<BatchNormLayer<T>>
                 (layers_map_[current_layer_id])->getBatchNormParam();

      if(isKeywordExist(var, data_config_keywords))
        break;

      SetupBatchNormParam(var, val, bn_param);
      break;
    } // End of case BN
    case DROPOUT: {
      // Obtain the data dimension and parameters variable within layer class
      input_dim = std::dynamic_pointer_cast<DropoutLayer<T>>
                  (layers_map_[current_layer_id])->getInputDim();
      dropout_param = std::dynamic_pointer_cast<DropoutLayer<T>>
                 (layers_map_[current_layer_id])->getDropoutParam();

      if(isKeywordExist(var, data_config_keywords))
        break;

      SetupDropoutParam(var, val, dropout_param);
      break;
    } // End of case DROPOUT
    case BYPASS: {
      // Obtain the data dimension and parameters variable within layer class
      input_dim = std::dynamic_pointer_cast<BypassLayer<T>>
                  (layers_map_[current_layer_id])->getInputDim();
      bypass_param = std::dynamic_pointer_cast<BypassLayer<T>>
                 (layers_map_[current_layer_id])->getBypassParam();

      if(isKeywordExist(var, data_config_keywords))
        break;

      // Process all the keywords in config
      if(!isKeywordExist(var, bypass_config_keywords)) {
        LOG(FATAL) << var << ": Keywords not exists" << std::endl;
      }
      break;
    } // End of case BYPASS
    default: {
      LOG(WARNING) << "NOT supported layer";
      break;
    } // End of case default

  }

  // Set data configuration at last, since all layers share same parameters
  if(isKeywordExist(var, data_config_keywords)) {
    if (!var.compare("n")) {
      input_dim->n_ = atoi(val.c_str());
    } else if (!var.compare("c")) {
      input_dim->c_ = atoi(val.c_str());
    } else if (!var.compare("h")) {
      input_dim->h_ = atoi(val.c_str());
    } else if (!var.compare("w")) {
      input_dim->w_ = atoi(val.c_str());
    } else if (!var.compare("name")) {
      layers_map_[current_layer_id]->setLayerName(val.c_str());
      name_id_map_[val] = current_layer_id;
    } else if (!var.compare("previous_layer")) {
      layers_map_[current_layer_id]->setPrevLayerName(val.c_str());
    }
  }
}

template <typename T>
void DNNMark<T>::ParseAllConfig(const std::string &config_file) {
  // TODO: use multithread in the future
  // Parse DNNMark specific config
  ParseGeneralConfig(config_file);

  // Parse Layers config
  ParseLayerConfig(config_file);
}

template <typename T>
int DNNMark<T>::ParseGeneralConfig(const std::string &config_file) {
  std::ifstream is;
  is.open(config_file.c_str(), std::ifstream::in);
  LOG(INFO) << "Search and parse general DNNMark configuration";

  // TODO: insert assert regarding run_mode_

  // Parse DNNMark config
  std::string s;
  bool is_general_section = false;
  while (!is.eof()) {
    // Obtain the string in one line
    std::getline(is, s);
    TrimStr(&s);

    // Check the specific configuration section markers
    if (isCommentStr(s) || isEmptyStr(s)) {
      continue;
    } else if (isGeneralSection(s)) {
      is_general_section = true;
      continue;
    } else if (isLayerSection(s)) {
      is_general_section = false;
      break;
    } else if (is_general_section) {
      // Obtain the acutal variable and value
      std::string var;
      std::string val;
      SplitStr(s, &var, &val);

      // Process all the keywords in config
      if(isKeywordExist(var, dnnmark_config_keywords)) {
        if (!var.compare("run_mode")) {
          if (!val.compare("none"))
            run_mode_ = NONE;
          else if(!val.compare("standalone"))
            run_mode_ = STANDALONE;
          else if(!val.compare("composed"))
            run_mode_ = COMPOSED;
          else
            std::cerr << "Unknown run mode" << std::endl;
        }
      } else {
        LOG(FATAL) << var << ": Keywords not exists" << std::endl;
      }
    }
  }

  is.close();
  return 0;
}

template <typename T>
int DNNMark<T>::ParseLayerConfig(const std::string &config_file) {
  std::ifstream is;
  is.open(config_file.c_str(), std::ifstream::in);

  // Parse DNNMark config
  std::string s;
  int current_layer_id;
  LayerType layer_type;
  bool is_layer_section = false;

  LOG(INFO) << "Search and parse layer configuration";
  while (!is.eof()) {
    // Obtain the string in one line
    std::getline(is, s);
    TrimStr(&s);

    // Check the specific configuration section markers
    if (isCommentStr(s) || isEmptyStr(s)){
      continue;
    } else if (isGeneralSection(s)) {
      is_layer_section = false;
    } else if (isLayerSection(s)) {
      is_layer_section = true;
      layer_type = layer_type_map.at(s);
      LOG(INFO) << "Add "
                << s
                << " layer";
      // Create a layer in the main class
      current_layer_id = num_layers_added_;
      if (layer_type == CONVOLUTION)
        layers_map_.emplace(current_layer_id,
          std::make_shared<ConvolutionLayer<T>>(this));
      else if (layer_type == POOLING)
        layers_map_.emplace(current_layer_id,
          std::make_shared<PoolingLayer<T>>(this));
      else if (layer_type == LRN)
        layers_map_.emplace(current_layer_id,
          std::make_shared<LRNLayer<T>>(this));
      else if (layer_type == ACTIVATION)
        layers_map_.emplace(current_layer_id,
          std::make_shared<ActivationLayer<T>>(this));
      else if (layer_type == FC)
        layers_map_.emplace(current_layer_id,
          std::make_shared<FullyConnectedLayer<T>>(this));
      else if (layer_type == SOFTMAX)
        layers_map_.emplace(current_layer_id,
          std::make_shared<SoftmaxLayer<T>>(this));
      else if (layer_type == BN)
        layers_map_.emplace(current_layer_id,
          std::make_shared<BatchNormLayer<T>>(this));
      else if (layer_type == DROPOUT)
        layers_map_.emplace(current_layer_id,
          std::make_shared<DropoutLayer<T>>(this));
      else if (layer_type == BYPASS)
	      layers_map_.emplace(current_layer_id,
	        std::make_shared<BypassLayer<T>>(this));
      layers_map_[current_layer_id]->setLayerId(current_layer_id);
      layers_map_[current_layer_id]->setLayerType(layer_type);
      num_layers_added_++;
      continue;
    } else if (is_layer_section) {
      // Obtain the acutal variable and value
      std::string var;
      std::string val;
      SplitStr(s, &var, &val);

      // Obtain the data dimension and parameters variable within layer class
      SetLayerParams(layer_type,
                     current_layer_id,
                     var, val);
    }
  }

  is.close();
  return 0;
}

template <typename T>
int DNNMark<T>::Initialize() {
  LOG(INFO) << "DNNMark: Initialize...";
  LOG(INFO) << "Running mode: " << run_mode_;
  LOG(INFO) << "Number of Layers: " << layers_map_.size();
  for (auto it = layers_map_.begin(); it != layers_map_.end(); it++) {
    LOG(INFO) << "Layer type: " << it->second->getLayerType();
    if (it->second->getLayerType() == CONVOLUTION) {
      LOG(INFO) << "DNNMark: Setup parameters of Convolution layer";
      std::dynamic_pointer_cast<ConvolutionLayer<T>>(it->second)->Setup();
    }
    if (it->second->getLayerType() == POOLING) {
      LOG(INFO) << "DNNMark: Setup parameters of Pooling layer";
      std::dynamic_pointer_cast<PoolingLayer<T>>(it->second)->Setup();
    }
    if (it->second->getLayerType() == LRN) {
      LOG(INFO) << "DNNMark: Setup parameters of LRN layer";
      std::dynamic_pointer_cast<LRNLayer<T>>(it->second)->Setup();
    }
    if (it->second->getLayerType() == ACTIVATION) {
      LOG(INFO) << "DNNMark: Setup parameters of Activation layer";
      std::dynamic_pointer_cast<ActivationLayer<T>>(it->second)->Setup();
    }
    if (it->second->getLayerType() == FC) {
      LOG(INFO) << "DNNMark: Setup parameters of Fully Connected layer";
      std::dynamic_pointer_cast<FullyConnectedLayer<T>>(it->second)->Setup();
    }
    if (it->second->getLayerType() == SOFTMAX) {
      LOG(INFO) << "DNNMark: Setup parameters of Softmax layer";
      std::dynamic_pointer_cast<SoftmaxLayer<T>>(it->second)->Setup();
    }
    if (it->second->getLayerType() == BN) {
      LOG(INFO) << "DNNMark: Setup parameters of Batch Normalization layer";
      std::dynamic_pointer_cast<BatchNormLayer<T>>(it->second)->Setup();
    }
    if (it->second->getLayerType() == DROPOUT) {
      LOG(INFO) << "DNNMark: Setup parameters of Dropout layer";
      std::dynamic_pointer_cast<DropoutLayer<T>>(it->second)->Setup();
    }
    if (it->second->getLayerType() == BYPASS) {
      LOG(INFO) << "DNNMark: Setup parameters of Bypass layer";
      std::dynamic_pointer_cast<BypassLayer<T>>(it->second)->Setup();
    }
  }
  return 0;
}

template <typename T>
int DNNMark<T>::RunAll() {
  for (auto it = layers_map_.begin(); it != layers_map_.end(); it++) {
    if (it->second->getLayerType() == CONVOLUTION) {
      std::dynamic_pointer_cast<ConvolutionLayer<T>>(it->second)
        ->ForwardPropagation();
      std::dynamic_pointer_cast<ConvolutionLayer<T>>(it->second)
        ->BackwardPropagation();
    }
    if (it->second->getLayerType() == POOLING) {
      std::dynamic_pointer_cast<PoolingLayer<T>>(it->second)
        ->ForwardPropagation();
      std::dynamic_pointer_cast<PoolingLayer<T>>(it->second)
        ->BackwardPropagation();
    }
    if (it->second->getLayerType() == LRN) {
      std::dynamic_pointer_cast<LRNLayer<T>>(it->second)
        ->ForwardPropagation();
      std::dynamic_pointer_cast<LRNLayer<T>>(it->second)
        ->BackwardPropagation();
    }
    if (it->second->getLayerType() == ACTIVATION) {
      std::dynamic_pointer_cast<ActivationLayer<T>>(it->second)
        ->ForwardPropagation();
      std::dynamic_pointer_cast<ActivationLayer<T>>(it->second)
        ->BackwardPropagation();
    }
    if (it->second->getLayerType() == FC) {
      std::dynamic_pointer_cast<FullyConnectedLayer<T>>(it->second)
        ->ForwardPropagation();
      std::dynamic_pointer_cast<FullyConnectedLayer<T>>(it->second)
        ->BackwardPropagation();
    }
    if (it->second->getLayerType() == SOFTMAX) {
      std::dynamic_pointer_cast<SoftmaxLayer<T>>(it->second)
        ->ForwardPropagation();
      std::dynamic_pointer_cast<SoftmaxLayer<T>>(it->second)
        ->BackwardPropagation();
    }
    if (it->second->getLayerType() == BN) {
      std::dynamic_pointer_cast<BatchNormLayer<T>>(it->second)
        ->ForwardPropagation();
      std::dynamic_pointer_cast<BatchNormLayer<T>>(it->second)
        ->BackwardPropagation();
    }
    if (it->second->getLayerType() == DROPOUT) {
      std::dynamic_pointer_cast<DropoutLayer<T>>(it->second)
        ->ForwardPropagation();
      std::dynamic_pointer_cast<DropoutLayer<T>>(it->second)
        ->BackwardPropagation();
    }
    if (it->second->getLayerType() == BYPASS) {
      std::dynamic_pointer_cast<BypassLayer<T>>(it->second)
        ->ForwardPropagation();
      std::dynamic_pointer_cast<BypassLayer<T>>(it->second)
        ->BackwardPropagation();
    }
  }
  return 0;
}

template <typename T>
int DNNMark<T>::Forward() {
  for (auto it = layers_map_.begin(); it != layers_map_.end(); it++) {
    if (it->second->getLayerType() == CONVOLUTION) {
      LOG(INFO) << "DNNMark: Running convolution forward: STARTED";
      std::dynamic_pointer_cast<ConvolutionLayer<T>>(it->second)
        ->ForwardPropagation();
      LOG(INFO) << "DNNMark: Running convolution forward: FINISHED";
    }
    if (it->second->getLayerType() == POOLING) {
      LOG(INFO) << "DNNMark: Running pooling forward: STARTED";
      std::dynamic_pointer_cast<PoolingLayer<T>>(it->second)
        ->ForwardPropagation();
      LOG(INFO) << "DNNMark: Running pooling forward: FINISHED";
    }
    if (it->second->getLayerType() == LRN) {
      LOG(INFO) << "DNNMark: Running LRN forward: STARTED";
      std::dynamic_pointer_cast<LRNLayer<T>>(it->second)
        ->ForwardPropagation();
      LOG(INFO) << "DNNMark: Running LRN forward: FINISHED";
    }
    if (it->second->getLayerType() == ACTIVATION) {
      LOG(INFO) << "DNNMark: Running Activation forward: STARTED";
      std::dynamic_pointer_cast<ActivationLayer<T>>(it->second)
        ->ForwardPropagation();
      LOG(INFO) << "DNNMark: Running Activation forward: FINISHED";
    }
    if (it->second->getLayerType() == FC) {
      LOG(INFO) << "DNNMark: Running FullyConnected forward: STARTED";
      std::dynamic_pointer_cast<FullyConnectedLayer<T>>(it->second)
        ->ForwardPropagation();
      LOG(INFO) << "DNNMark: Running FullyConnected forward: FINISHED";
    }
    if (it->second->getLayerType() == SOFTMAX) {
      LOG(INFO) << "DNNMark: Running Softmax forward: STARTED";
      std::dynamic_pointer_cast<SoftmaxLayer<T>>(it->second)
        ->ForwardPropagation();
      LOG(INFO) << "DNNMark: Running Softmax forward: FINISHED";
    }
    if (it->second->getLayerType() == BN) {
      LOG(INFO) << "DNNMark: Running BatchNormalization forward: STARTED";
      std::dynamic_pointer_cast<BatchNormLayer<T>>(it->second)
        ->ForwardPropagation();
      LOG(INFO) << "DNNMark: Running BatchNormalization forward: FINISHED";
    }
    if (it->second->getLayerType() == DROPOUT) {
      LOG(INFO) << "DNNMark: Running Dropout forward: STARTED";
      std::dynamic_pointer_cast<DropoutLayer<T>>(it->second)
        ->ForwardPropagation();
      LOG(INFO) << "DNNMark: Running Dropout forward: FINISHED";
    }
    if (it->second->getLayerType() == BYPASS) {
      LOG(INFO) << "DNNMark: Running Bypass forward: STARTED";
      std::dynamic_pointer_cast<BypassLayer<T>>(it->second)
        ->ForwardPropagation();
      LOG(INFO) << "DNNMark: Running Bypass forward: FINISHED";
    }
  }
  return 0;
}

template <typename T>
int DNNMark<T>::Backward() {
  for (auto it = layers_map_.rbegin(); it != layers_map_.rend(); it++) {
    if (it->second->getLayerType() == CONVOLUTION) {
      LOG(INFO) << "DNNMark: Running convolution backward: STARTED";
      std::dynamic_pointer_cast<ConvolutionLayer<T>>(it->second)
        ->BackwardPropagation();
      LOG(INFO) << "DNNMark: Running convolution backward: FINISHED";
    }
    if (it->second->getLayerType() == POOLING) {
      LOG(INFO) << "DNNMark: Running pooling backward: STARTED";
      std::dynamic_pointer_cast<PoolingLayer<T>>(it->second)
        ->BackwardPropagation();
      LOG(INFO) << "DNNMark: Running pooling backward: FINISHED";
    }
    if (it->second->getLayerType() == LRN) {
      LOG(INFO) << "DNNMark: Running LRN backward: STARTED";
      std::dynamic_pointer_cast<LRNLayer<T>>(it->second)
        ->BackwardPropagation();
      LOG(INFO) << "DNNMark: Running LRN backward: FINISHED";
    }
    if (it->second->getLayerType() == ACTIVATION) {
      LOG(INFO) << "DNNMark: Running Activation backward: STARTED";
      std::dynamic_pointer_cast<ActivationLayer<T>>(it->second)
        ->BackwardPropagation();
      LOG(INFO) << "DNNMark: Running Activation backward: FINISHED";
    }
    if (it->second->getLayerType() == FC) {
      LOG(INFO) << "DNNMark: Running FullyConnected backward: STARTED";
      std::dynamic_pointer_cast<FullyConnectedLayer<T>>(it->second)
        ->BackwardPropagation();
      LOG(INFO) << "DNNMark: Running FullyConnected backward: FINISHED";
    }
    if (it->second->getLayerType() == SOFTMAX) {
      LOG(INFO) << "DNNMark: Running Softmax backward: STARTED";
      std::dynamic_pointer_cast<SoftmaxLayer<T>>(it->second)
        ->BackwardPropagation();
      LOG(INFO) << "DNNMark: Running Softmax backward: FINISHED";
    }
    if (it->second->getLayerType() == BN) {
      LOG(INFO) << "DNNMark: Running BatchNormalization backward: STARTED";
      std::dynamic_pointer_cast<BatchNormLayer<T>>(it->second)
        ->BackwardPropagation();
      LOG(INFO) << "DNNMark: Running BatchNormalization backward: FINISHED";
    }
    if (it->second->getLayerType() == DROPOUT) {
      LOG(INFO) << "DNNMark: Running Dropout backward: STARTED";
      std::dynamic_pointer_cast<DropoutLayer<T>>(it->second)
        ->BackwardPropagation();
      LOG(INFO) << "DNNMark: Running Dropout backward: FINISHED";
    }
    if (it->second->getLayerType() == BYPASS) {
      LOG(INFO) << "DNNMark: Running Bypass backward: STARTED";
      std::dynamic_pointer_cast<BypassLayer<T>>(it->second)
        ->BackwardPropagation();
      LOG(INFO) << "DNNMark: Running Bypass backward: FINISHED";
    }
  }
  return 0;
}


// Explicit instantiation
template class DNNMark<TestType>;

} // namespace dnnmark

