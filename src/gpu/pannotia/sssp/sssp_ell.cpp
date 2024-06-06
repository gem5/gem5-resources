/************************************************************************************\ 
 *                                                                                  *
 * Copyright � 2014 Advanced Micro Devices, Inc.                                    *
 * Copyright (c) 2015 Mark D. Hill and David A. Wood                                *
 * Copyright (c) 2021 Gaurav Jain and Matthew D. Sinclair                           *
 * All rights reserved.                                                             *
 *                                                                                  *
 * Redistribution and use in source and binary forms, with or without               *
 * modification, are permitted provided that the following are met:                 *
 *                                                                                  *
 * You must reproduce the above copyright notice.                                   *
 *                                                                                  *
 * Neither the name of the copyright holder nor the names of its contributors       *
 * may be used to endorse or promote products derived from this software            *
 * without specific, prior, written permission from at least the copyright holder.  *
 *                                                                                  *
 * You must include the following terms in your license and/or other materials      *
 * provided with the software.                                                      *
 *                                                                                  *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"      *
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE        *
 * IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, AND FITNESS FOR A       *
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER        *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,         *
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT  *
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS      *
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN          *
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING  *
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY   *
 * OF SUCH DAMAGE.                                                                  *
 *                                                                                  *
 * Without limiting the foregoing, the software may implement third party           *
 * technologies for which you must obtain licenses from parties other than AMD.     *
 * You agree that AMD has not obtained or conveyed to you, and that you shall       *
 * be responsible for obtaining the rights to use and/or distribute the applicable  *
 * underlying intellectual property rights related to the third party technologies. *
 * These third party technologies are not licensed hereunder.                       *
 *                                                                                  *
 * If you use the software (in whole or in part), you shall adhere to all           *
 * applicable U.S., European, and other export laws, including but not limited to   *
 * the U.S. Export Administration Regulations ("EAR") (15 C.F.R Sections 730-774),  *
 * and E.U. Council Regulation (EC) No 428/2009 of 5 May 2009.  Further, pursuant   *
 * to Section 740.6 of the EAR, you hereby certify that, except pursuant to a       *
 * license granted by the United States Department of Commerce Bureau of Industry   *
 * and Security or as otherwise permitted pursuant to a License Exception under     *
 * the U.S. Export Administration Regulations ("EAR"), you will not (1) export,     *
 * re-export or release to a national of a country in Country Groups D:1, E:1 or    *
 * E:2 any restricted technology, software, or source code you receive hereunder,   *
 * or (2) export to Country Groups D:1, E:1 or E:2 the direct product of such       *
 * technology or software, if such foreign produced direct product is subject to    *
 * national security controls as identified on the Commerce Control List (currently *
 * found in Supplement 1 to Part 774 of EAR).  For the most current Country Group   *
 * listings, or for additional information about the EAR or your obligations under  *
 * those regulations, please refer to the U.S. Bureau of Industry and Security's    *
 * website at http://www.bis.doc.gov/.                                              *
 *                                                                                  *
\************************************************************************************/

#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <algorithm>
#include "../graph_parser/parse.h"
#include "../graph_parser/util.h"
#include "kernel.h"

#if defined(GEM5_FUSION) || defined(GEM5_FS)
#include <stdint.h>
#include <gem5/m5ops.h>
#endif

#ifdef GEM5_FS
#include <util/m5/src/m5_mmap.h>
#endif

#define BIGNUM 99999999

void print_vector(int *vector, int num);

int main(int argc, char **argv)
{
    char *tmpchar;
    bool directed = 1;

    int num_nodes;
    int num_edges;
    int file_format = 1;

    hipError_t err = hipSuccess;

    if (argc == 3) {
        tmpchar = argv[1];  // Graph inputfile
        file_format = atoi(argv[2]);
    } else {
        fprintf(stderr, "You did something wrong!\n");
        exit(1);
    }

    // Allocate the csr structure
    csr_array *csr;

    // Parse the graph and store it into the CSR structure
    if (file_format == 1) {
        csr = parseMetis_transpose(tmpchar, &num_nodes, &num_edges, directed);
    } else if (file_format == 0) {
        csr = parseCOO_transpose(tmpchar, &num_nodes, &num_edges, directed);
    } else {
        printf("reserve for future");
        exit(1);
    }

    // Allocate ell and transform from csr
    ell_array *ell = csr2ell(csr, num_nodes, num_edges, BIGNUM);
    int height = ell->max_height;

    // Allocate the cost array
    int *cost_array = (int *)malloc(num_nodes * sizeof(int));
    if (!cost_array) fprintf(stderr, "malloc failed cost_array\n");

    // Set the cost array to zero
    for (int i = 0; i < num_nodes; i++) {
        cost_array[i] = 0;
    }

    // Create device-side buffers
    int *ell_col_d;
    int *ell_data_d;
    int *vector_d1;
    int *vector_d2;
    int *stop_d;

    // Create the device-side graph structure
    err = hipMalloc(&ell_col_d, height * num_nodes * sizeof(int));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc ell_col_d (size:%d) => %s\n", height * num_nodes, hipGetErrorString(err));
        return -1;
    }
    err = hipMalloc(&ell_data_d, height * num_nodes * sizeof(int));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc ell_data_d (size:%d) => %s\n", height * num_nodes, hipGetErrorString(err));
        return -1;
    }

    // Termination variable
    err = hipMalloc(&stop_d, sizeof(int));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc stop_d (size:%d) => %s\n", 1, hipGetErrorString(err));
        return -1;
    }

    // Create the device-side buffers for sssp
    err = hipMalloc(&vector_d1, num_nodes * sizeof(int));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc vector_d1 (size:%d) => %s\n", num_nodes, hipGetErrorString(err));
        return -1;
    }
    err = hipMalloc(&vector_d2, num_nodes * sizeof(int));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc vector_d2 (size:%d) => %s\n", num_nodes, hipGetErrorString(err));
        return -1;
    }

    double timer1 = gettime();

#ifdef GEM5_FUSION
    m5_work_begin(0, 0);
#endif

#ifdef GEM5_FS
    m5op_addr = 0xFFFF0000;
    map_m5_mem();
    m5_work_begin_addr(0, 0);
#endif

    // Copy data to device side buffers
    err = hipMemcpy(ell_col_d, ell->col_array, height * num_nodes * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMemcpy ell_col_d (size:%d) => %s\n", height * num_nodes, hipGetErrorString(err));
        return -1;
    }

    err = hipMemcpy(ell_data_d, ell->data_array, height * num_nodes * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMemcpy ell_data_d (size:%d) => %s\n", height * num_nodes, hipGetErrorString(err));
        return -1;
    }

    double timer3 = gettime();

    // Work dimensions
    int block_size = 64;
    int num_blocks = (num_nodes + block_size - 1) / block_size;

    dim3 threads(block_size, 1, 1);
    dim3 grid(num_blocks, 1, 1);

    // Source vertex 0
    int sourceVertex = 0;

    // Launch the initialization kernel
    hipLaunchKernelGGL(vector_init, dim3(grid), dim3(threads), 0, 0, vector_d1, vector_d2, sourceVertex, num_nodes);
    hipDeviceSynchronize();
    err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: vector_init failed (%s)\n", hipGetErrorString(err));
        return -1;
    }

    int stop = 1;
    int cnt = 0;
    // Main computation loop
    for (int i = 1; i < num_nodes; i++) {
        // Reset the termination variable
        stop = 0;

        // Copy the termination variable to the device
        err = hipMemcpy(stop_d, &stop, sizeof(int), hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            fprintf(stderr, "ERROR: write stop_d (%s)\n", hipGetErrorString(err));
            return -1;
        }

        // Launch the assignment kernel
        hipLaunchKernelGGL(vector_assign, dim3(grid), dim3(threads), 0, 0, vector_d1, vector_d2, num_nodes);

        // Launch the min.+ kernel
        hipLaunchKernelGGL(ell_min_dot_plus_kernel, dim3(grid), dim3(threads), 0, 0, num_nodes, height,
                                                    ell_col_d, ell_data_d,
                                                    vector_d1, vector_d2);

        // Launch the check kernel
        hipLaunchKernelGGL(vector_diff, dim3(grid), dim3(threads), 0, 0, vector_d1, vector_d2,
                                        stop_d, num_nodes);

        // Read the termination variable back
        err = hipMemcpy(&stop, stop_d, sizeof(int), hipMemcpyDeviceToHost);
        if (err != hipSuccess) {
            fprintf(stderr, "ERROR: read stop_d (%s)\n", hipGetErrorString(err));
            return -1;
        }

        // Exit the loop
        if (stop == 0) {
            break;
        }
        cnt++;
    }
    hipDeviceSynchronize();
    double timer4 = gettime();

    // Read the cost_array back
    err = hipMemcpy(cost_array, vector_d1, num_nodes * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: read vector_d1 (%s)\n", hipGetErrorString(err));
        return -1;
    }

#ifdef GEM5_FUSION
    m5_work_end(0, 0);
#endif

#ifdef GEM5_FS
    m5_work_end_addr(0, 0);
    unmap_m5_mem();
#endif

    double timer2 = gettime();

    // Print the timing statistics
    printf("kernel + memcpy time = %lf ms\n", (timer2 - timer1) * 1000);
    printf("kernel time = %lf ms\n", (timer4 - timer3) * 1000);
    printf("number iterations = %d\n", cnt);

#if 1
    // Print cost_array
    print_vector(cost_array, num_nodes);
#endif

    // Clean up the host arrays
    free(cost_array);
    csr->freeArrays();
    free(csr);

    free(ell->col_array);
    free(ell->data_array);
    free(ell);

    // Clean up the device-side buffers
    hipFree(ell_col_d);
    hipFree(ell_data_d);
    hipFree(stop_d);
    hipFree(vector_d1);
    hipFree(vector_d2);

    return 0;
}

void print_vector(int *vector, int num)
{

    FILE * fp = fopen("result.out", "w");
    if (!fp) {
        printf("ERROR: unable to open result.txt\n");
    }

    for (int i = 0; i < num; i++)
        fprintf(fp, "%d: %d\n", i + 1, vector[i]);

    fclose(fp);
}
