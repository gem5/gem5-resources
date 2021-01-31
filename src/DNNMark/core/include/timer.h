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

#ifndef CORE_INCLUDE_TIMER_H_
#define CORE_INCLUDE_TIMER_H_

#include <sys/time.h>
#include <numeric>
#include <map>

namespace dnnmark {

// Class StopWatch
class StopWatch {
 private:
  struct timeval start_;
  struct timeval end_;
  bool started_;

 public:
  StopWatch() :
    started_(false) {}

  void Start() {
    if (!started_)
      gettimeofday(&start_, NULL);
    else
      LOG(FATAL) << "The Stop Watch has already started";
    started_ = true;
  }
  void Stop() {
    if (started_) {
      gettimeofday(&end_, NULL);
      started_ = false;
    } else {
      LOG(FATAL) << "No Stop Watch has been started yet";
    }
  }
  double DiffInMs() {
    return static_cast<double>(end_.tv_sec * 1000 +
        static_cast<double>(end_.tv_usec) / 1000) -
        static_cast<double>(start_.tv_sec * 1000 +
            static_cast<double>(start_.tv_usec) / 1000);
  }
};

class Timer {
 private:
  StopWatch watch_;
  std::vector<std::string> layer_table_;
  std::vector<double> timing_table_;
  int num_records_;
  double total_time_;

 public:
  Timer()
  : watch_(), num_records_(0), total_time_(0.0) {}

  void Start(const std::string &layer) {
    watch_.Start();
    layer_table_.push_back(layer);
  }

  // Stop and record the current elapsed time and record it
  void Stop(const std::string &layer) {
    watch_.Stop();
    if (!layer.compare(layer_table_.back()))
      timing_table_.push_back(watch_.DiffInMs());
    else
      LOG(FATAL) << "Layer to measure doesn't match";
  }

  // Reset all the recorded value to 0
  void Clear() {
	  layer_table_.clear();
	  timing_table_.clear();
  }

  // Sum up all the recorded times and store the sum to vec_
  void SumRecords() {
    int index = 0;
    for (auto const &i : layer_table_) {
      LOG(INFO) << i << ": " << timing_table_[index] << "ms";
      total_time_ += timing_table_[index];
      index++;
    }
  }

  double GetTotalTime() {
    return total_time_;
  }

  int GetNumRecords() {
    return num_records_;
  }

};

} // namespace dnnmark

#endif // CORE_INCLUDE_TIMER_H_
