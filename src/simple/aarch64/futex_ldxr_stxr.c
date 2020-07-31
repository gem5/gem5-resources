/* Copyright (c) 2020 ARM Limited
 * All rights reserved
 *
 * The license below extends only to copyright in the software and shall
 * not be construed as granting a license to any other intellectual
 * property including but not limited to intellectual property relating
 * to a hardware implementation of the functionality of the software
 * licensed hereunder.  You may use the software subject to the license
 * terms below provided that you ensure that this notice is replicated
 * unmodified and in its entirety in all distributions of the software,
 * modified or unmodified, in source code or in binary form.
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
 *
 * Correct outcome:
 * Exiting @ tick 18446744073709551615 because simulate() limit reached
 * Incorrect behaviour due to: https://gem5.atlassian.net/browse/GEM5-537
 * Exits successfully. */

#define _GNU_SOURCE
#include <assert.h>
#include <inttypes.h>
#include <linux/futex.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>

#define LDXR_OPS_SIZE 1024
static int futex1 = 1;
static int futex2 = 1;
/* We do this to ensure that those two varibles are well separated.
 * If they are too close (same cache line?), then the str to ldxr_done
 * can make CPU1 lose the lock. */
static uint64_t ldxr_ops[LDXR_OPS_SIZE];
static uint64_t *ldxr_done = ldxr_ops;
static uint64_t *ldxr_var = ldxr_ops + LDXR_OPS_SIZE - 1;

void __attribute__ ((noinline)) busy_loop(
    unsigned long long max,
    unsigned long long max2
) {
    for (unsigned long long i = 0; i < max2; i++) {
        for (unsigned long long j = 0; j < max; j++) {
            __asm__ __volatile__ ("" : "+g" (i), "+g" (j) : :);
        }
    }
}

static int
futex(int *uaddr, int futex_op, int val,
        const struct timespec *timeout, int *uaddr2, int val3)
{
    register uint64_t x0 __asm__ ("x0") = (uint64_t)uaddr;
    register uint64_t x1 __asm__ ("x1") = futex_op;
    register uint64_t x2 __asm__ ("x2") = val;
    register const struct timespec *x3 __asm__ ("x3") = timeout;
    register int *x4 __asm__ ("x4") = uaddr2;
    register uint64_t x5 __asm__ ("x5") = val3;
    register uint64_t x8 __asm__ ("x8") = SYS_futex; /* syscall number */
    __asm__ __volatile__ (
        "svc 0;"
        : "+r" (x0)
        : "r" (x1), "r" (x2), "r" (x3), "r" (x4), "r" (x5), "r" (x8)
        : "memory"
    );
    return x0;
}

void* thread_main(void *arg) {
    (void)arg;
    __asm__ __volatile__ (
        "ldxr x0, [%[ldxr_var]];mov %[ldxr_done], 1"
        : [ldxr_done] "=r" (*ldxr_done)
        : [ldxr_var] "r" (ldxr_var)
        : "x0", "memory"
    );
    futex(&futex1, FUTEX_WAIT, futex1, NULL, NULL, 0);
    futex(&futex2, FUTEX_WAIT, futex2, NULL, NULL, 0);
    return NULL;
}

int main(void) {
    pthread_t thread;
    pthread_create(&thread, NULL, thread_main, NULL);
    while (!*ldxr_done) {}
    /* Wait for thread1 to sleep on futex1. Works because */
    busy_loop(1000, 1);
    /* Try to wake up the thread with an LLSC event.
     * It should not wake up in a correct implementation,
     * but it used to happen in gem5 before it was fixed. */
    __asm__ __volatile__ (
        "mov x0, 1;ldxr x0, [%0]; stxr w1, x0, [%0]"
        :
        : "r" (ldxr_var) : "x0", "x1", "memory"
    );
    /* Wait for thread1 to sleep on futex2. */
    busy_loop(1000, 1);
    /* Before it was fixed in gem5, this would wrongly wake a futex2
     * because the previous futex1 was woken up via LLSC. */
    futex(&futex1, FUTEX_WAKE, 1, NULL, NULL, 0);
    assert(!pthread_join(thread, NULL));
}
