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

#ifndef CORE_INCLUDE_CONFIG_KEYWORDS_H_
#define CORE_INCLUDE_CONFIG_KEYWORDS_H_

#include <vector>
#include <string>
#include <algorithm>

namespace dnnmark {

// Configuration section keywords
const std::vector<std::string> general_section_keywords = {
  "[DNNMark]"
};
const std::vector<std::string> layer_section_keywords = {
  "[Convolution]",
  "[Pooling]",
  "[LRN]",
  "[Activation]",
  "[FullyConnected]",
  "[Softmax]",
  "[BatchNorm]",
  "[Dropout]",
  "[Bypass]"
};

// DNNMark keywords
const std::vector<std::string> dnnmark_config_keywords = {
  "run_mode"
};

// Data config keywords
const std::vector<std::string> data_config_keywords = {
  "name",
  "n",
  "c",
  "h",
  "w",
  "previous_layer"
};

// Convolution layer keywords
const std::vector<std::string> conv_config_keywords = {
  "conv_mode",
  "algo",
  "algod",
  "propagation",
  "num_output",
  "kernel_size",
  "pad",
  "stride",
  "kernel_size_h",
  "kernel_size_w",
  "pad_h",
  "pad_w",
  "stride_h",
  "stride_w",
  "conv_fwd_pref",
  "conv_bwd_filter_pref",
  "conv_bwd_data_pref"
};

// Pooling layer keywords
const std::vector<std::string> pool_config_keywords = {
  "pool_mode",
  "kernel_size",
  "pad",
  "stride",
  "kernel_size_h",
  "kernel_size_w",
  "pad_h",
  "pad_w",
  "stride_h",
  "stride_w"
};

// LRN layer keywords
const std::vector<std::string> lrn_config_keywords = {
  "lrn_mode",
  "local_size",
  "alpha",
  "beta",
  "k"
};

// Activation layer keywords
const std::vector<std::string> activation_config_keywords = {
  "activation_mode"
};

// FC layer keywords
const std::vector<std::string> fc_config_keywords = {
  "num_output"
};

// Softmax layer keywords
const std::vector<std::string> softmax_config_keywords = {
  "softmax_algo",
  "softmax_mode"
};

// BN layer keywords
const std::vector<std::string> bn_config_keywords = {
  "batchnorm_mode",
  "save_intermediates",
  "exp_avg_factor",
  "epsilon"
};

// DROPOUT layer keywords
const std::vector<std::string> dropout_config_keywords = {
  "dropout_probability",
  "random_seed"
};

// BYPASS layer keywords
const std::vector<std::string> bypass_config_keywords = {
};

bool isSection(const std::string &s);
bool isGeneralSection(const std::string &s);
bool isLayerSection(const std::string &s);
bool isKeywordExist(const std::string &s,
                    const std::vector<std::string> &config_keywords);

} // namespace dnnmark

#endif // CORE_INCLUDE_CONFIG_KEYWORDS_H_

