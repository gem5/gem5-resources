#include <stdlib.h>
#include "matrix-omp.h"

float a[size][size];
float b[size][size];
float c[size][size];

float d[size2][size2];
float e[size2][size2];
float f[size2][size2];

void init() {
    // Initialize buffers.
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            a[i][j] = (float)i + j + rand() % 29;
            b[i][j] = (float)i - j;
            c[i][j] = 0.0f;
        }
    }

    // Initialize buffers.
    for (int i = 0; i < size2; ++i) {
        for (int j = 0; j < size2; ++j) {
            d[i][j] = (float)i + j + rand() % 29;
            e[i][j] = (float)i - j;
            f[i][j] = 0.0f;
        }
    }

}
