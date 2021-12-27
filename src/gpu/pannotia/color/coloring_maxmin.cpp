/************************************************************************************\ 
 *                                                                                  *
 * Copyright © 2014 Advanced Micro Devices, Inc.                                    *
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
#include "../graph_parser/parse.h"
#include "../graph_parser/util.h"
#include "kernel_maxmin.h"

#ifdef GEM5_FUSION
#include <stdint.h>
#include <gem5/m5ops.h>
#endif

#define RANGE 2048

void print_vector(int *vector, int num);

int main(int argc, char **argv)
{
    char *tmpchar;

    int num_nodes;
    int num_edges;
    int file_format = 1;
    bool directed = 0;

    hipError_t err = hipSuccess;

    if (argc == 3) {
        tmpchar = argv[1];  //graph inputfile
        file_format = atoi(argv[2]); //graph format
    } else {
        fprintf(stderr, "You did something wrong!\n");
        exit(1);
    }

    srand(7);

    // Allocate the CSR structure
    csr_array *csr;

    // Parse graph file and store into a CSR format
    if (file_format == 1)
        csr = parseMetis(tmpchar, &num_nodes, &num_edges, directed);
    else if (file_format == 0)
        csr = parseCOO(tmpchar, &num_nodes, &num_edges, directed);
    else {
        printf("reserve for future");
        exit(1);
    }

    // Allocate the vertex value array
    int *node_value = (int *)malloc(num_nodes * sizeof(int));
    if (!node_value) fprintf(stderr, "node_value malloc failed\n");
    // Allocate the color array
    int *color = (int *)malloc(num_nodes * sizeof(int));
    if (!color) fprintf(stderr, "color malloc failed\n");

    // Initialize all the colors to -1
    // Randomize the value for each vertex
    for (int i = 0; i < num_nodes; i++) {
        color[i] = -1;
        node_value[i] = rand() % RANGE;
    }

    int *row_d;
    int *col_d;
    int *max_d;
    int *min_d;

    int *color_d;
    int *node_value_d;
    int *stop_d;

    // Create device-side buffers for the graph
    err = hipMalloc(&row_d, num_nodes * sizeof(int));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc row_d (size:%d) => %s\n",  num_nodes , hipGetErrorString(err));
        return -1;
    }
    err = hipMalloc(&col_d, num_edges * sizeof(int));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc col_d (size:%d): %s\n",  num_edges , hipGetErrorString(err));
        return -1;
    }

    // Termination variable
    err = hipMalloc(&stop_d, sizeof(int));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc stop_d (size:%d) => %s\n",  1 , hipGetErrorString(err));
        return -1;
    }

    // Create device-side buffers for color
    err = hipMalloc(&color_d, num_nodes * sizeof(int));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc color_d (size:%d) => %s\n", num_nodes , hipGetErrorString(err));
        return -1;
    }
    err = hipMalloc(&node_value_d, num_nodes * sizeof(int));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc node_value_d (size:%d) => %s\n", num_nodes , hipGetErrorString(err));
        return -1;
    }
    err = hipMalloc(&max_d, num_nodes * sizeof(int));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc max_d (size:%d) => %s\n",  num_nodes , hipGetErrorString(err));
        return -1;
    }
    err = hipMalloc(&min_d, num_nodes * sizeof(int));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc min_d (size:%d) => %s\n",  num_nodes , hipGetErrorString(err));
        return -1;
    }

    // Copy data to device-side buffers
    double timer1 = gettime();

#ifdef GEM5_FUSION
    m5_work_begin(0, 0);
#endif

    err = hipMemcpy(color_d, color, num_nodes * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMemcpy color_d (size:%d) => %s\n", num_nodes, hipGetErrorString(err));
        return -1;
    }

    err = hipMemcpy(row_d, csr->row_array, num_nodes * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMemcpy row_d (size:%d) => %s\n", num_nodes, hipGetErrorString(err));
        return -1;
    }

    err = hipMemcpy(col_d, csr->col_array, num_edges * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMemcpy col_d (size:%d) => %s\n", num_nodes, hipGetErrorString(err));
        return -1;
    }

    err = hipMemcpy(node_value_d, node_value, num_nodes * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMemcpy node_value_d (size:%d) => %s\n", num_nodes, hipGetErrorString(err));
        return -1;
    }

    int block_size = 256;
    int num_blocks = (num_nodes + block_size - 1) / block_size;

    // Set up kernel dimensions
    dim3 threads(block_size,  1, 1);
    dim3 grid(num_blocks, 1,  1);

    int stop = 1;
    int graph_color = 1;

    // Initialize arrays
    hipLaunchKernelGGL(ini, dim3(grid), dim3(threads ), 0, 0, max_d, min_d, num_nodes);

    // Main computation loop
    double timer3 = gettime();

    while (stop) {

        stop = 0;

        // Copy the termination variable to the device
        err = hipMemcpy(stop_d, &stop, sizeof(int), hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            fprintf(stderr, "ERROR: write stop_d: %s\n", hipGetErrorString(err));
        }

        // Launch the color kernel 1
        hipLaunchKernelGGL(color1, dim3(grid), dim3(threads ), 0, 0, row_d, col_d, node_value_d, color_d,
                                     stop_d, max_d, min_d, graph_color,
                                     num_nodes, num_edges);

        // Launch the color kernel 2
        hipLaunchKernelGGL(color2, dim3(grid), dim3(threads ), 0, 0, node_value_d, color_d, max_d, min_d,
                                     graph_color, num_nodes, num_edges);

        err = hipMemcpy(&stop, stop_d, sizeof(int), hipMemcpyDeviceToHost);
        if (err != hipSuccess) {
            fprintf(stderr, "ERROR: read stop_d: %s\n", hipGetErrorString(err));
        }

        // Update the color label for the next iter
        graph_color = graph_color + 2;

    }
    hipDeviceSynchronize();

    double timer4 = gettime();

    // Copy back the color array
    err = hipMemcpy(color, color_d, num_nodes * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        printf("ERROR: hipMemcpy(): %s\n", hipGetErrorString(err));
        return -1;
    }

#ifdef GEM5_FUSION
    m5_work_end(0, 0);
#endif

    double timer2 = gettime();

    // Print out color and timing statistics
    printf("total number of colors used: %d\n", graph_color);
    printf("kernel time = %lf ms\n", (timer4 - timer3) * 1000);
    printf("kernel + memcpy time = %lf ms\n", (timer2 - timer1) * 1000);

#if 1
    // Dump the color array into an output file
    print_vector(color, num_nodes);
#endif

    // Free host-side buffers
    free(node_value);
    free(color);
    csr->freeArrays();
    free(csr);

    // Free CUDA buffers
    hipFree(row_d);
    hipFree(col_d);
    hipFree(max_d);
    hipFree(color_d);
    hipFree(node_value_d);
    hipFree(stop_d);

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
