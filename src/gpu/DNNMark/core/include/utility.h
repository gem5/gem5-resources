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

#ifndef CORE_INCLUDE_UTILITY_H_
#define CORE_INCLUDE_UTILITY_H_

#include <string>
#include <algorithm>
#include <cctype>
#include <functional>

namespace dnnmark {

//
// Inplace trim string by eliminating the space like characters on both ends
//

void TrimStr(std::string *s);
void TrimStrLeft(std::string *s);
void TrimStrRight(std::string *s);

//
// Splict actual parameter into variable and value
//

void SplitStr(const std::string &s, std::string *var, std::string *val,
              std::string delimiter = "=");

//
// Detect useless str
//

bool isCommentStr(const std::string &s, char comment_marker = '#');

bool isEmptyStr(const std::string &s);

} // namespace dnnmark

#endif // CORE_INCLUDE_UTILITY_H_
