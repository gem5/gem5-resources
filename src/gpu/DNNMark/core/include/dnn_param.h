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

#ifndef CORE_INCLUDE_DNN_PARAM_H_
#define CORE_INCLUDE_DNN_PARAM_H_

#include <iostream>
#include <string>
#include <glog/logging.h>

#ifdef NVIDIA_CUDNN
#include <cudnn.h>
#endif

#ifdef AMD_MIOPEN
#include <miopen/miopen.h>
#endif

#include "common.h"
#include "dnn_config_keywords.h"

namespace dnnmark {

struct DataDim {
  int n_;
  int c_;
  int h_;
  int w_;

  DataDim()
  : n_(0), c_(0), h_(0), w_(0) {}
};

inline std::ostream &operator<<(std::ostream &os, const DataDim &data_dim) {
  os << std::endl;
  os << "[Data Dim] N: " << data_dim.n_ << std::endl;
  os << "[Data Dim] C: " << data_dim.c_ << std::endl;
  os << "[Data Dim] H: " << data_dim.h_ << std::endl;
  os << "[Data Dim] W: " << data_dim.w_ << std::endl;
  return os;
}

struct ConvolutionParam {
#ifdef NVIDIA_CUDNN
  cudnnConvolutionMode_t mode_;
#endif
#ifdef AMD_MIOPEN
  miopenConvolutionMode_t mode_;
#endif
  bool propagation_;
  int output_num_;
  int pad_h_;
  int pad_w_;
  int stride_u_;
  int stride_v_;
  int upscale_x_;
  int upscale_y_;
  int kernel_size_h_;
  int kernel_size_w_;
  bool algo_set_;
  std::string algo_;
  std::string algod_;
#ifdef NVIDIA_CUDNN
  cudnnConvolutionFwdPreference_t conv_fwd_pref_;
  cudnnConvolutionBwdFilterPreference_t conv_bwd_filter_pref_;
  cudnnConvolutionBwdDataPreference_t conv_bwd_data_pref_;
#endif
#ifdef AMD_MIOPEN
  miopenConvAlgoPerf_t *pref_;
#endif
  ConvolutionParam()
#ifdef NVIDIA_CUDNN
  : mode_(CUDNN_CROSS_CORRELATION),
#endif
#ifdef AMD_MIOPEN
  : mode_(miopenConvolution),
#endif
    output_num_(32),
    pad_h_(2), pad_w_(2),
    stride_u_(1), stride_v_(1),
    upscale_x_(1), upscale_y_(1),
    kernel_size_h_(5), kernel_size_w_(5), propagation_(true),
    algo_set_(false), algo_(""), algod_(""),
#ifdef NVIDIA_CUDNN
    conv_fwd_pref_(CUDNN_CONVOLUTION_FWD_PREFER_FASTEST),
    conv_bwd_filter_pref_(CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST),
    conv_bwd_data_pref_(CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST) {}
#endif
#ifdef AMD_MIOPEN
    pref_(nullptr) {}
#endif

};

inline std::ostream &operator<<(std::ostream &os,
                         const ConvolutionParam &conv_param) {
  os << std::endl;
  os << "[Convolution Param] Output Num: "
     << conv_param.output_num_ << std::endl;
  os << "[Convolution Param] Pad H: "
     << conv_param.pad_h_ << std::endl;
  os << "[Convolution Param] Pad W: "
     << conv_param.pad_w_ << std::endl;
  os << "[Convolution Param] Stride U: "
     << conv_param.stride_u_ << std::endl;
  os << "[Convolution Param] Stride V: "
     << conv_param.stride_v_ << std::endl;
  os << "[Convolution Param] Kernel Size H: "
     << conv_param.kernel_size_h_ << std::endl;
  os << "[Convolution Param] Kernel Size W: "
     << conv_param.kernel_size_w_ << std::endl;

  return os;
}

inline void SetupConvParam(const std::string &var, const std::string &val,
                           ConvolutionParam *conv_param) {
  // Process all the corresponding keywords in config
  if(isKeywordExist(var, conv_config_keywords)) {
    if (!var.compare("conv_mode")) {
      if (!val.compare("convolution"))
#ifdef NVIDIA_CUDNN
        conv_param->mode_ = CUDNN_CONVOLUTION;
#endif
#ifdef AMD_MIOPEN
        conv_param->mode_ = miopenTranspose;
#endif
      else if (!val.compare("cross_correlation"))
#ifdef NVIDIA_CUDNN
        conv_param->mode_ = CUDNN_CROSS_CORRELATION;
#endif
#ifdef AMD_MIOPEN
        conv_param->mode_ = miopenConvolution;
#endif
#ifdef AMD_MIOPEN
      else if (!val.compare("transpose"))
        conv_param->mode_ = miopenTranspose;
#endif
      else
        LOG(FATAL) << "Invalid conv mode: " << val << std::endl;
    } else if (!var.compare("num_output")) {
      conv_param->output_num_ = atoi(val.c_str());
    } else if (!var.compare("kernel_size")) {
      conv_param->kernel_size_h_ = atoi(val.c_str());
      conv_param->kernel_size_w_ = atoi(val.c_str());
    } else if (!var.compare("pad")) {
      conv_param->pad_h_ = atoi(val.c_str());
      conv_param->pad_w_ = atoi(val.c_str());
    } else if (!var.compare("stride")) {
      conv_param->stride_u_ = atoi(val.c_str());
      conv_param->stride_v_ = atoi(val.c_str());
    } else if (!var.compare("kernel_size_h")) {
      conv_param->kernel_size_h_ = atoi(val.c_str());
    } else if (!var.compare("kernel_size_w")) {
      conv_param->kernel_size_w_ = atoi(val.c_str());
    } else if (!var.compare("pad_h")) {
      conv_param->pad_h_ = atoi(val.c_str());
    } else if (!var.compare("pad_w")) {
      conv_param->pad_w_ = atoi(val.c_str());
    } else if (!var.compare("stride_h")) {
      conv_param->stride_u_ = atoi(val.c_str());
    } else if (!var.compare("stride_w")) {
      conv_param->stride_v_ = atoi(val.c_str());
    } else if (!var.compare("propagation")) {
      if (!val.compare("false"))
        conv_param->propagation_ = false;
    } else if (!var.compare("algo")) {
        conv_param->algo_set_ = true;
        conv_param->algo_ = val;
    } else if (!var.compare("algod")) {
        conv_param->algod_ = val;
    } else if (!var.compare("conv_fwd_pref")) {
#ifdef NVIDIA_CUDNN
      if (!val.compare("no_workspace"))
        conv_param->conv_fwd_pref_ = CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
      else if (!val.compare("fastest"))
        conv_param->conv_fwd_pref_ = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
      else if (!val.compare("specify_workspace_limit"))
        conv_param->conv_fwd_pref_ =
          CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT;
#endif
    } else if (!var.compare("conv_bwd_filter_pref")) {
#ifdef NVIDIA_CUDNN
      if (!val.compare("no_workspace"))
        conv_param->conv_bwd_filter_pref_ =
          CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE;
      else if (!val.compare("fastest"))
        conv_param->conv_bwd_filter_pref_ =
          CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST;
      else if (!val.compare("specify_workspace_limit"))
        conv_param->conv_bwd_filter_pref_ =
          CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT;
#endif
    } else if (!var.compare("conv_bwd_data_pref")) {
#ifdef NVIDIA_CUDNN
      if (!val.compare("no_workspace"))
        conv_param->conv_bwd_data_pref_ =
          CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE;
      else if (!val.compare("fastest"))
        conv_param->conv_bwd_data_pref_ =
          CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
      else if (!val.compare("specify_workspace_limit"))
        conv_param->conv_bwd_data_pref_ =
          CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT;
#endif
    }
  } else {
    LOG(FATAL) << var << ": Keywords not exists" << std::endl;
  }
}

struct PoolingParam {
#ifdef NVIDIA_CUDNN
  cudnnPoolingMode_t mode_;
#endif
#ifdef AMD_MIOPEN
  miopenPoolingMode_t mode_;
#endif
  int pad_h_;
  int pad_w_;
  int stride_h_;
  int stride_w_;
  int kernel_size_h_;
  int kernel_size_w_;
  PoolingParam()
#ifdef NVIDIA_CUDNN
  : mode_(CUDNN_POOLING_MAX),
#endif
#ifdef AMD_MIOPEN
  : mode_(miopenPoolingMax),
#endif
    pad_h_(0), pad_w_(0),
    stride_h_(2), stride_w_(2),
    kernel_size_h_(3), kernel_size_w_(3) {}
};

inline std::ostream &operator<<(std::ostream &os,
                         const PoolingParam &pool_param) {
  os << std::endl;
  os << "[Pooling Param] Pad H: "
     << pool_param.pad_h_ << std::endl;
  os << "[Pooling Param] Pad W: "
     << pool_param.pad_w_ << std::endl;
  os << "[Pooling Param] Stride H: "
     << pool_param.stride_h_ << std::endl;
  os << "[Pooling Param] Stride W: "
     << pool_param.stride_w_ << std::endl;
  os << "[Pooling Param] Kernel Size H: "
     << pool_param.kernel_size_h_ << std::endl;
  os << "[Pooling Param] Kernel Size W: "
     << pool_param.kernel_size_w_ << std::endl;

  return os;
}

inline void SetupPoolingParam(const std::string &var, const std::string &val,
                              PoolingParam *pool_param) {
  // Process all the keywords in config
  if(isKeywordExist(var, pool_config_keywords)) {
    if (!var.compare("pool_mode")) {
      if (!val.compare("max"))
#ifdef NVIDIA_CUDNN
        pool_param->mode_ = CUDNN_POOLING_MAX;
#endif
#ifdef AMD_MIOPEN
        pool_param->mode_ = miopenPoolingMax;
#endif
#ifdef NVIDIA_CUDNN
      else if (!val.compare("avg_include_padding"))
        pool_param->mode_ = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
      else if (!val.compare("avg_exclude_padding"))
        pool_param->mode_ = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
#endif
#ifdef AMD_MIOPEN
      else if (!val.compare("avg"))
        pool_param->mode_ = miopenPoolingAverage;
#endif
      else
        LOG(FATAL) << "Invalid pool mode" << std::endl;
    } else if (!var.compare("kernel_size")) {
      pool_param->kernel_size_h_ = atoi(val.c_str());
      pool_param->kernel_size_w_ = atoi(val.c_str());
    } else if (!var.compare("pad")) {
      pool_param->pad_h_ = atoi(val.c_str());
      pool_param->pad_w_ = atoi(val.c_str());
    } else if (!var.compare("stride")) {
      pool_param->stride_h_ = atoi(val.c_str());
      pool_param->stride_w_ = atoi(val.c_str());
    } else if (!var.compare("kernel_size_h")) {
      pool_param->kernel_size_h_ = atoi(val.c_str());
    } else if (!var.compare("kernel_size_w")) {
      pool_param->kernel_size_w_ = atoi(val.c_str());
    } else if (!var.compare("pad_h")) {
      pool_param->pad_h_ = atoi(val.c_str());
    } else if (!var.compare("pad_w")) {
      pool_param->pad_w_ = atoi(val.c_str());
    } else if (!var.compare("stride_h")) {
      pool_param->stride_h_ = atoi(val.c_str());
    } else if (!var.compare("stride_w")) {
      pool_param->stride_w_ = atoi(val.c_str());
    }
  } else {
    LOG(FATAL) << var << ": Keywords not exists" << std::endl;
  }
}

struct LRNParam {
#ifdef NVIDIA_CUDNN
  cudnnLRNMode_t mode_;
#endif
#ifdef AMD_MIOPEN
  miopenLRNMode_t mode_;
#endif
  int local_size_;
  double alpha_;
  double beta_;
  double k_;
  LRNParam()
#ifdef NVIDIA_CUDNN
  : mode_(CUDNN_LRN_CROSS_CHANNEL_DIM1),
#endif
#ifdef AMD_MIOPEN
  : mode_(miopenLRNCrossChannel),
#endif
    local_size_(5),
    alpha_(0.0001), beta_(0.75), k_(2.0) {}
};

inline std::ostream &operator<<(std::ostream &os,
                         const LRNParam &lrn_param) {
  os << std::endl;
  os << "[LRN Param] Local size: "
     << lrn_param.local_size_ << std::endl;
  os << "[LRN Param] Alpha: "
     << lrn_param.alpha_ << std::endl;
  os << "[LRN Param] Beta: "
     << lrn_param.beta_ << std::endl;
  os << "[LRN Param] K: "
     << lrn_param.k_ << std::endl;

  return os;
}

inline void SetupLrnParam(const std::string &var, const std::string &val,
                          LRNParam *lrn_param) {
  // Process all the keywords in config
  if(isKeywordExist(var, lrn_config_keywords)) {
    if (!var.compare("lrn_mode")) {
      if (!val.compare("cross_channel_dim1"))
#ifdef NVIDIA_CUDNN
        lrn_param->mode_ = CUDNN_LRN_CROSS_CHANNEL_DIM1;
#endif
#ifdef AMD_MIOPEN
        lrn_param->mode_ = miopenLRNCrossChannel;
      else if (!val.compare("within_channel"))
        lrn_param->mode_ = miopenLRNWithinChannel;
#endif
      else
        LOG(FATAL) << "Invalid lrn mode" << std::endl;
    } else if (!var.compare("local_size")) {
      lrn_param->local_size_ = atoi(val.c_str());
    } else if (!var.compare("alpha")) {
      lrn_param->alpha_ = atof(val.c_str());
    } else if (!var.compare("beta")) {
      lrn_param->beta_ = atof(val.c_str());
    } else if (!var.compare("k")) {
      lrn_param->k_ = atof(val.c_str());
    }
  } else {
    LOG(FATAL) << var << ": Keywords not exists" << std::endl;
  }
}

struct ActivationParam {
#ifdef NVIDIA_CUDNN
  cudnnActivationMode_t mode_;
  ActivationParam()
  : mode_(CUDNN_ACTIVATION_RELU) {}
#endif
#ifdef AMD_MIOPEN
  miopenActivationMode_t mode_;
  double alpha_;
  double beta_;
  double power_;
  ActivationParam()
  : mode_(miopenActivationRELU),
    alpha_(0.0), beta_(0.0), power_(0.0) {}
#endif
};

inline std::ostream &operator<<(std::ostream &os,
                                const ActivationParam &activation_param) {
  os << std::endl;
  os << "[Activation Param] Mode: "
     << activation_param.mode_ << std::endl;
  return os;
}

inline void SetupActivationParam(const std::string &var, const std::string &val,
                                 ActivationParam *activation_param) {
  // Process all the keywords in config
  if(isKeywordExist(var, activation_config_keywords)) {
    if (!var.compare("activation_mode")) {
#ifdef NVIDIA_CUDNN
      if (!val.compare("sigmoid"))
        activation_param->mode_ = CUDNN_ACTIVATION_SIGMOID;
      else if (!val.compare("relu"))
        activation_param->mode_ = CUDNN_ACTIVATION_RELU;
      else if (!val.compare("tanh"))
        activation_param->mode_ = CUDNN_ACTIVATION_TANH;
      else if (!val.compare("clipped_relu"))
        activation_param->mode_ = CUDNN_ACTIVATION_CLIPPED_RELU;
      else
        LOG(FATAL) << "Invalid activation mode" << std::endl;
#endif
#ifdef AMD_MIOPEN
      if (!val.compare("sigmoid"))
        activation_param->mode_ = miopenActivationLOGISTIC;
      else if (!val.compare("relu"))
        activation_param->mode_ = miopenActivationRELU;
      else if (!val.compare("tanh"))
        activation_param->mode_ = miopenActivationTANH;
      else if (!val.compare("soft_relu"))
        activation_param->mode_ = miopenActivationSOFTRELU;
      else
        LOG(FATAL) << "Invalid activation mode" << std::endl;
#endif
    }
  } else {
    LOG(FATAL) << var << ": Keywords not exists" << std::endl;
  }
}

struct FullyConnectedParam {
  int output_num_;
  FullyConnectedParam()
  : output_num_(4096) {}
};

inline void SetupFcParam(const std::string &var, const std::string &val,
                         FullyConnectedParam *fc_param) {
  // Process all the keywords in config
  if(isKeywordExist(var, fc_config_keywords)) {
    if (!var.compare("num_output")) {
      fc_param->output_num_ = atoi(val.c_str());
    }
  } else {
    LOG(FATAL) << var << ": Keywords not exists" << std::endl;
  }
}

struct SoftmaxParam {
#ifdef NVIDIA_CUDNN
  cudnnSoftmaxAlgorithm_t algo_;
  cudnnSoftmaxMode_t mode_;
  SoftmaxParam()
  : algo_(CUDNN_SOFTMAX_ACCURATE),
    mode_(CUDNN_SOFTMAX_MODE_CHANNEL) {}
#endif
#ifdef AMD_MIOPEN
  SoftmaxParam() {}
#endif
};

inline void SetupSoftmaxParam(const std::string &var, const std::string &val,
                              SoftmaxParam *softmax_param) {
  // Process all the keywords in config
  if(isKeywordExist(var, softmax_config_keywords)) {
    if (!var.compare("softmax_algo")) {
#ifdef NVIDIA_CUDNN
      if (!val.compare("fast"))
        softmax_param->algo_ = CUDNN_SOFTMAX_FAST;
      else if (!val.compare("accurate"))
        softmax_param->algo_ = CUDNN_SOFTMAX_ACCURATE;
      else if (!val.compare("log"))
        softmax_param->algo_ = CUDNN_SOFTMAX_LOG;
#endif
    }
    if (!var.compare("softmax_mode")) {
#ifdef NVIDIA_CUDNN
      if (!val.compare("instance"))
        softmax_param->mode_ = CUDNN_SOFTMAX_MODE_INSTANCE;
      else if (!val.compare("channel"))
        softmax_param->mode_ = CUDNN_SOFTMAX_MODE_CHANNEL;
#endif
    }
  } else {
    LOG(FATAL) << var << ": Keywords not exists" << std::endl;
  }
}

enum BatchNormMode {
#ifdef NVIDIA_CUDNN
  PerActivation = CUDNN_BATCHNORM_PER_ACTIVATION,
  Spatial = CUDNN_BATCHNORM_SPATIAL
#endif
#ifdef AMD_MIOPEN
  PerActivation = miopenBNPerActivation,
  Spatial = miopenBNSpatial
#endif
};

struct BatchNormParam {
#ifdef NVIDIA_CUDNN
  cudnnBatchNormMode_t mode_;
#endif
#ifdef AMD_MIOPEN
  miopenBatchNormMode_t mode_;
#endif
  bool save_intermediates_;
  double exp_avg_factor_;
  double epsilon_;
  BatchNormParam()
#ifdef NVIDIA_CUDNN
  : mode_((cudnnBatchNormMode_t)PerActivation),
#endif
#ifdef AMD_MIOPEN
  : mode_((miopenBatchNormMode_t)PerActivation),
#endif
    save_intermediates_(true),
    exp_avg_factor_(1),
    epsilon_(BN_MIN_EPSILON) {}
};

inline std::ostream &operator<<(std::ostream &os,
                                const BatchNormParam &bn_param) {
  os << std::endl;
  os << "[BatchNorm Param] Mode: "
     << bn_param.mode_ << std::endl;
  os << "[BatchNorm Param] Save Intermediates: "
     << bn_param.save_intermediates_ << std::endl;
  os << "[BatchNorm Param] Exponential Average Factor: "
     << bn_param.exp_avg_factor_ << std::endl;
  os << "[BatchNorm Param] Epsilon: "
     << bn_param.epsilon_ << std::endl;
  return os;
}

inline void SetupBatchNormParam(const std::string &var, const std::string &val,
                                BatchNormParam *bn_param) {
  // Process all the keywords in config
  if(isKeywordExist(var, bn_config_keywords)) {
    if(!var.compare("batchnorm_mode")) {
      if(!val.compare("per_activation"))
#ifdef NVIDIA_CUDNN
        bn_param->mode_ = (cudnnBatchNormMode_t)PerActivation;
#endif
#ifdef AMD_MIOPEN
        bn_param->mode_ = (miopenBatchNormMode_t)PerActivation;
#endif
      else if (!val.compare("spatial"))
#ifdef NVIDIA_CUDNN
        bn_param->mode_ = (cudnnBatchNormMode_t)Spatial;
#endif
#ifdef AMD_MIOPEN
        bn_param->mode_ = (miopenBatchNormMode_t)Spatial;
#endif
    }
    if(!var.compare("save_intermediates")) {
      if(!val.compare("true"))
        bn_param->save_intermediates_ = true;
      else if (!val.compare("false"))
        bn_param->save_intermediates_ = false;
    }
    if(!var.compare("exp_avg_factor")) {
      bn_param->exp_avg_factor_ = atof(val.c_str());
    }
    if(!var.compare("epsilon")) {
      bn_param->epsilon_ = atof(val.c_str());
    }
  } else {
    LOG(FATAL) << var << ": Keywords not exists" << std::endl;
  }
}

struct DropoutParam {
  float dropout_p_;
  unsigned long long random_seed_;
  DropoutParam()
  : dropout_p_(.5),
    random_seed_(0) {}
};

inline std::ostream &operator<<(std::ostream &os,
                                const DropoutParam &dropout_param) {
  os << std::endl;
  os << "[Dropout Param] Dropout Probability: "
     << dropout_param.dropout_p_ << std::endl;
  os << "[Dropout Param] Random Seed: "
     << dropout_param.random_seed_ << std::endl;
  return os;
}

inline void SetupDropoutParam(const std::string &var, const std::string &val,
                              DropoutParam * dropout_param) {
  // Process all the keywords in config
  if(isKeywordExist(var, dropout_config_keywords)) {
    if(!var.compare("dropout_probability")) {
      dropout_param->dropout_p_ = atof(val.c_str());
    }
    if(!var.compare("random_seed")) {
      dropout_param->random_seed_ = atoi(val.c_str());
    }
  } else {
    LOG(FATAL) << var << ": Keywords not exists" << std::endl;
  }
}

struct BypassParam {
	BypassParam() {}
};

inline std::ostream &operator<<(std::ostream &os,
                                const BypassParam &bypass_param) {
  os << std::endl;
  os << "[Bypass Param]" << std::endl;
  return os;
}

} // namespace dnnmark

#endif // CORE_INCLUDE_DNN_PARAM_H_
