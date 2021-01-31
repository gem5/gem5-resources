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

#include "dnn_config_keywords.h"

namespace dnnmark {

bool isSection(const std::string &s) {
  return (std::find(general_section_keywords.begin(),
                    general_section_keywords.end(), s)
         != general_section_keywords.end()) &&
         (std::find(layer_section_keywords.begin(),
                    layer_section_keywords.end(), s)
         != layer_section_keywords.end());
}

bool isGeneralSection(const std::string &s) {
  return std::find(general_section_keywords.begin(),
                   general_section_keywords.end(), s)
         != general_section_keywords.end();
}

bool isLayerSection(const std::string &s) {
  return std::find(layer_section_keywords.begin(),
                   layer_section_keywords.end(), s)
         != layer_section_keywords.end();
}

bool isKeywordExist(const std::string &s,
                    const std::vector<std::string> &config_keywords) {
  return std::find(config_keywords.begin(),
                   config_keywords.end(), s)
         != config_keywords.end();
}

}


