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

#include "gemm_wrapper.h"

namespace dnnmark {

template <>
void dnnmarkGEMM(const Handle &handle, RunMode mode, int idx,
                 bool is_a_transpose, bool is_b_transpose,
                 int m, int n, int k,
                 float *alpha,
                 float *a, int lda,
                 float *b, int ldb,
                 float *beta,
                 float *c, int ldc) {
#ifdef NVIDIA_CUDNN
  cublasOperation_t transa = is_a_transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transb = is_b_transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CALL(cublasSgemm(mode == COMPOSED ?
                          handle.GetBlas(idx) : handle.GetBlas(),
                          transa, transb,
                          m, n, k,
                          alpha,
                          a, lda,
                          b, ldb,
                          beta,
                          c, ldc));
#endif
#ifdef AMD_MIOPEN
  rocblas_operation transa = is_a_transpose ? rocblas_operation_transpose :
                             rocblas_operation_none;
  rocblas_operation transb = is_b_transpose ? rocblas_operation_transpose :
                             rocblas_operation_none;

  ROCBLAS_CALL(rocblas_sgemm(mode == COMPOSED ?
                             handle.GetBlas(idx) : handle.GetBlas(),
                             transa, transb,
                             m, n, k,
                             alpha,
                             a, lda,
                             b, ldb,
                             beta,
                             c, ldc));
#endif

}

template <>
void dnnmarkGEMM(const Handle &handle, RunMode mode, int idx,
                 bool is_a_transpose, bool is_b_transpose,
                 int m, int n, int k,
                 double *alpha,
                 double *a, int lda,
                 double *b, int ldb,
                 double *beta,
                 double *c, int ldc) {
#ifdef NVIDIA_CUDNN
  cublasOperation_t transa = is_a_transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transb = is_b_transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CALL(cublasDgemm(mode == COMPOSED ?
                          handle.GetBlas(idx) : handle.GetBlas(),
                          transa, transb,
                          m, n, k,
                          alpha,
                          a, lda,
                          b, ldb,
                          beta,
                          c, ldc));
#endif
#ifdef AMD_MIOPEN
  rocblas_operation transa = is_a_transpose ? rocblas_operation_transpose :
                             rocblas_operation_none;
  rocblas_operation transb = is_b_transpose ? rocblas_operation_transpose :
                             rocblas_operation_none;

  ROCBLAS_CALL(rocblas_dgemm(mode == COMPOSED ?
                             handle.GetBlas(idx) : handle.GetBlas(),
                             transa, transb,
                             m, n, k,
                             alpha,
                             a, lda,
                             b, ldb,
                             beta,
                             c, ldc));
#endif
}

} // namespace dnnmark

