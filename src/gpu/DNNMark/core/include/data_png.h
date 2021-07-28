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

#ifndef CORE_INCLUDE_DATA_PNG_H_
#define CORE_INCLUDE_DATA_PNG_H_

#include <fcntl.h>
#include <map>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace dnnmark {

// Seed of random number generator
static unsigned long long int seed = 1234;

class PseudoNumGenerator {
 private:
#ifdef NVIDIA_CUDNN
  curandGenerator_t gen_;
#endif

  // Constructor
  PseudoNumGenerator(const std::string &mmap_file) :
    mmap_file_(mmap_file.c_str()), use_mmap(!mmap_file.empty()) {
#ifdef NVIDIA_CUDNN
    CURAND_CALL(curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen_, seed));
#endif
  }

  PseudoNumGenerator() {
#ifdef NVIDIA_CUDNN
    CURAND_CALL(curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen_, seed));
#endif
  }

  // PNG instance
  static std::unique_ptr<PseudoNumGenerator> instance_;
  static uint64_t offset;
  const char *mmap_file_;
  bool use_mmap;
 public:

  ~PseudoNumGenerator() {
#ifdef NVIDIA_CUDNN
    CURAND_CALL(curandDestroyGenerator(gen_));
#endif
  }

  static void CreateInstance(const std::string &mmap_file) {
    if (instance_.get())
      return;
    instance_.reset(new PseudoNumGenerator(mmap_file));
  }

  static PseudoNumGenerator *GetInstance() {
    if (instance_.get())
      return instance_.get();
    instance_.reset(new PseudoNumGenerator());
    return instance_.get();
  }
  void GenerateUniformData(float *dev_ptr, int size) {
#ifdef NVIDIA_CUDNN
    CURAND_CALL(curandGenerateUniform(gen_, dev_ptr, size));
#endif
#ifdef AMD_MIOPEN
    float *host_ptr;
    if (use_mmap) {
        int fd = open(mmap_file_, O_RDONLY);
        LOG_IF(FATAL, size*sizeof(float) > lseek(fd, 0, SEEK_END)) <<
            "Requested data larger than binary file size";

        if (offset + (size * sizeof(float)) > lseek(fd, 0, SEEK_END)) {
            LOG(INFO) << "Mapped binary data insufficient, rolling over";
            offset = 0;
        }

        host_ptr = (float *)mmap(NULL, size*sizeof(float), PROT_READ, MAP_SHARED,
                                fd, offset);
        offset += size*sizeof(float);
        offset -= offset % sysconf(_SC_PAGESIZE);
        close(fd);
    } else {
        host_ptr = new float[size];
        for (int i = 0; i < size; i++)
          host_ptr[i] = static_cast <float> (rand()) /
                        (static_cast <float> (RAND_MAX/seed));
    }

    memcpy(dev_ptr, host_ptr, size * sizeof(float));
    if (use_mmap) {
        munmap(host_ptr, size*sizeof(float));
    } else {
        delete []host_ptr;
    }

#endif
  }
  void GenerateUniformData(double *dev_ptr, int size) {
#ifdef NVIDIA_CUDNN
    CURAND_CALL(curandGenerateUniformDouble(gen_, dev_ptr, size));
#endif
#ifdef AMD_MIOPEN
    double *host_ptr = new double[size];
    if (use_mmap) {
        int fd = open(mmap_file_, O_RDONLY);
        LOG_IF(FATAL, size*sizeof(double) > lseek(fd, 0, SEEK_END)) <<
            "Requested data larger than binary file size";

        if (offset + (size * sizeof(double)) > lseek(fd, 0, SEEK_END)) {
            LOG(INFO) << "Mapped binary data insufficient, rolling over";
            offset = 0;
        }

        host_ptr = (double *)mmap(NULL, size*sizeof(double), PROT_READ, MAP_SHARED,
                                  fd, offset);
        offset += size*sizeof(float);
        offset -= offset % sysconf(_SC_PAGESIZE);
        close(fd);
    } else {
        for (int i = 0; i < size; i++)
          host_ptr[i] = static_cast <double> (rand()) /
                        (static_cast <double> (RAND_MAX/seed));
    }

    memcpy(dev_ptr, host_ptr, size * sizeof(double));

    if (use_mmap) {
        munmap(host_ptr, size*sizeof(double));
    } else {
        delete []host_ptr;
    }

#endif
  }
};

uint64_t PseudoNumGenerator::offset = 0;

std::unique_ptr<PseudoNumGenerator> PseudoNumGenerator::instance_ = nullptr;

} // namespace dnnmark

#endif // CORE_INCLUDE_DATA_PNG_H_

