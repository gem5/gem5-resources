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

#ifndef BENCHMARKS_USAGE_H_
#define BENCHMARKS_USAGE_H_

#include <gflags/gflags.h>

DECLARE_string(config);
DECLARE_int32(debuginfo);
DECLARE_int32(warmup);
DECLARE_int32(iterations);
DECLARE_string(mmap);

#define INIT_FLAGS(X, Y) \
gflags::SetUsageMessage(\
      "\n[DNNMark benchmark usage]\n"\
      "./<benchmark> <args>\n"\
      );\
google::ParseCommandLineFlags(&X, &Y, true)

#define INIT_LOG(X) \
google::InitGoogleLogging(X[0]);\
FLAGS_logtostderr = FLAGS_debuginfo;\
CHECK_GT(FLAGS_config.size(), 0) << "Configuration file is needed."

#endif // BENCHMARKS_USAGE_H_

