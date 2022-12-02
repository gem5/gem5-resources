/*
 * Copyright (c) 2022 The Regents of the University of California
 * All rights reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>

int main()
{
    const int size = 100;
    int first[size][size], second[size][size], multiply[size][size];

    printf("Populating the first and second matrix...\n");
    for(int x=0; x<size; x++)
    {
        for(int y=0; y<size; y++)
        {
            first[x][y] = x + y;
            second[x][y] = (4 * x) + (7 * y);
        }
    }
    printf("Done!\n");

    printf("Multiplying the matrixes...\n");
    for(int c=0; c<size; c++)
    {
        for(int d=0; d<size; d++)
        {
            int sum = 0;
            for(int k=0; k<size; k++)
            {
                sum += first[c][k] * second[k][d];
            }
           multiply[c][d] = sum;
        }
    }
    printf("Done!\n");

    printf("Calculating the sum of all elements in the matrix...\n");
    long int sum = 0;
    for(int x=0; x<size; x++)
        for(int y=0; y<size; y++)
            sum += multiply[x][y];
    printf("Done\n");

    printf("The sum is %ld\n", sum);
}
