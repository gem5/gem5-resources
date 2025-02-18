// The program parallelizes matrix multiplication using OpenMP.
// From: http://blog.speedgocomputing.com/2010/08/parallelizing-matrix-multiplication.html
// License: Creative Commons Attribution-Noncommercial-ShareAlike 3.0 Unported

#include "matrix-omp.h"

#include <iostream>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include <omp.h>
#include <stdio.h>
int iterations = 20;
int ncores = 4;

int main(int argc, char*argv[])
{
    init();

    // From http://stackoverflow.com/questions/11095309/openmp-set-num-threads-is-not-working
    omp_set_dynamic(0);     // Explicitly disable dynamic teams

    // Initialize buffers.
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            a[i][j] = (float)i + j;
            b[i][j] = (float)i - j;
            c[i][j] = 0.0f;
        }
    }

    // Initialize buffers.
    for (int i = 0; i < size2; ++i) {
        for (int j = 0; j < size2; ++j) {
            d[i][j] = (float)i + j;
            e[i][j] = (float)i - j;
            f[i][j] = 0.0f;
        }
    }

    if (argc >= 2) {
        errno = 0;
        iterations = strtol(argv[1], NULL, 10);
        if ((errno == ERANGE && (iterations == LONG_MAX || iterations == LONG_MIN))
               || (errno != 0 && iterations == 0)) {
  //      if (errno != 0) {
           std::cerr << "Unable to convert parameter to an iteration count\n";
        }
    }

    if (argc >= 3) {
        errno = 0;
        ncores = strtol(argv[2], NULL, 10);
        if ((errno == ERANGE && (ncores == LONG_MAX || ncores == LONG_MIN))
               || (errno != 0 && ncores == 0)) {
  //      if (errno != 0) {
           std::cerr << "Unable to convert parameter to a core count\n";
        }
    }
    omp_set_num_threads(ncores); // Use ncores threads for all consecutive parallel regions

    std::cout << "Using " << iterations << " iterations\n";
    std::cout << "Using " << ncores << " threads\n";

    for (int itr = 0 ; itr < iterations ; itr++) {
    // Compute matrix multiplication.
    // C <- C + A x B
    #pragma omp parallel default(none) shared(a,b,c)
    {
        bool if_print = true;
    #pragma omp for
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
/*
        if(if_print)
        {
            printf("Hello\n");
            if_print = false;
        }*/
    }
    }
//    }

//    for (int itr = 0 ; itr < iterations2 ; itr++) {
    // Compute matrix multiplication.
    // C <- C + A x B
    #pragma omp parallel for default(none) shared(d,e,f)
    for (int i = 0; i < size2; ++i) {
        for (int j = 0; j < size2; ++j) {
            for (int k = 0; k < size2; ++k) {
                f[i][j] += d[i][k] * e[k][j];
            }
        }
    }
    }

    return 0;
}
