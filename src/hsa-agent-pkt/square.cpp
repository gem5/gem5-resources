/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <stdio.h>
#include "hip/hip_runtime.h"
#include "HSA_Interface.h"

/*
 * Square each element in the array A and write to array C.
 */
template <typename T>
__global__ void
vector_square(hipLaunchParm lp, T *C_d, const T *A_d, size_t N)
{
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x ;

    for (size_t i=offset; i<N; i+=stride) {
        C_d[i] = A_d[i] * A_d[i];
    }
}


int main(int argc, char *argv[])
{
    float *A_h, *C_h;
    size_t N = 1000000;

    if (argc == 2)
        N = atoi(argv[1]);

    const unsigned threadsPerBlock = 256;
    unsigned blocks = (N + threadsPerBlock - 1) / threadsPerBlock; 

    size_t Nbytes = N * sizeof(float);

    HSA_Interface * hsa= new HSA_Interface();

    printf ("info: allocate host mem (%6.2f MB)\n", 2*Nbytes/1024.0/1024.0);
    A_h = (float*)malloc(Nbytes);
    CHECK(A_h == 0 ? hipErrorMemoryAllocation : hipSuccess );
    C_h = (float*)malloc(Nbytes);
    CHECK(C_h == 0 ? hipErrorMemoryAllocation : hipSuccess );

    // Fill with Phi + i
    for (size_t i=0; i<N; i++)
    {
        A_h[i] = 1.618f + i;
    }

    int kernel_id = 0;
    printf ("info: launch 'vector_square' kernel: "
            "N = %lu | Blocks = %u | kernel_id %d\n",
            N, blocks, kernel_id);
    hipLaunchKernel(vector_square, dim3(blocks),
                    dim3(threadsPerBlock), 0, hsa->getStream(),
                    C_h, A_h, N);


    //Kernel_id must match that of the launched kernel (ie launch order)
    printf("info: Stealing kernel completion signal (kid: %d)\n",
            kernel_id);
    hsa->steal_kernel_signal(kernel_id);

    //Theoretically Equivalent to hipDeviceSynchronize();
    printf("info: Waiting on kernel completion signal (kid: %d)\n",
            kernel_id);
    hsa->wait_kernel(kernel_id);

    //Increment the Kernel_id every time any kernel is launched.
    kernel_id++;

    printf ("info: check result\n");
    for (size_t i=0; i<N; i++)  {
        if (C_h[i] != A_h[i] * A_h[i]) {
            CHECK(hipErrorUnknown);
        }
    }
    printf ("PASSED!\n");
	return 0;
}
