/*
 * Copyright (c) 2011-2015 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * For use for simulation and test purposes only
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * # example test compile and run parameters
 * # Note: the absolute path to the chdir-print binary should be specified
 * # in the run command even if running from the same folder. This is needed
 * # because chdir is executed before triggering a clone for the file read,
 * # and the cloned process won't be able to find the executable if a relative
 * # path is provided.
 *
 * # compile examples
 * scons --default=X86 ./build/X86/gem5.opt PROTOCOL=MOESI_hammer
 * scons --default=X86 ./build/X86/gem5.opt PROTOCOL=MESI_Three_Level
 *
 * # run parameters
 * <GEM5_ROOT>/build/X86/gem5.opt <GEM5_ROOT>/configs/example/se.py -c \
 *   <GEM5_ROOT>/tests/test-progs/chdir-print/chdir-print -n2 --ruby
 *
 *
 * # example successful output for MESI_Three_Level:
 *
 * <...>
 *
 * **** REAL SIMULATION ****
 * info: Entering event queue @ 0.  Starting simulation...
 * warn: Replacement policy updates recently became the responsibility of
 *   SLICC state machines. Make sure to setMRU() near callbacks in .sm files!
 * cwd: /proj/research_simu/users/jalsop/gem5-mem_dif_debug/tests/
 *   test-progs/chdir-print/
 * cwd: /proc
 *
 * <...>
 *
 * processor       : 0
 * vendor_id       : Generic
 * cpu family      : 0
 * model           : 0
 * model name      : Generic
 * stepping        : 0
 * cpu MHz         : 2000
 * cache size:     : 2048K
 * physical id     : 0
 * siblings        : 2
 * core id         : 0
 * cpu cores       : 2
 * fpu             : yes
 * fpu exception   : yes
 * cpuid level     : 1
 * wp              : yes
 * flags           : fpu
 * cache alignment : 64
 *
 * processor       : 1
 * vendor_id       : Generic
 * cpu family      : 0
 * model           : 0
 * model name      : Generic
 * stepping        : 0
 * cpu MHz         : 2000
 * cache size:     : 2048K
 * physical id     : 0
 * siblings        : 2
 * core id         : 1
 * cpu cores       : 2
 * fpu             : yes
 * fpu exception   : yes
 * cpuid level     : 1
 * wp              : yes
 * flags           : fpu
 * cache alignment : 64
 *
 * SUCCESS
 * Exiting @ tick 2694923000 because exiting with last active thread context
 */

#include <assert.h>
#include <linux/limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

const int BUFFER_SIZE = 64;

// Tests the functionality of RegisterFilesystem
int main(void)
{
    char *cwd = getcwd(NULL, PATH_MAX);
    printf("cwd: %s\n", cwd);
    free(cwd);

    assert(!chdir("/proc"));

    cwd = getcwd(NULL, PATH_MAX);
    printf("cwd: %s\n", cwd);
    free(cwd);

    FILE *fp;
    char buffer[BUFFER_SIZE];

    bool found_procline = false;
    fp = popen("cat cpuinfo", "r");
    if (fp != NULL) {
        while (fgets(buffer, BUFFER_SIZE, fp) != NULL) {
            printf("%s", buffer);
            if (strstr(buffer, "processor")) {
                found_procline = true;
            }
        }
        pclose(fp);
    }

    if (found_procline) {
        printf("SUCCESS\n");
        return EXIT_SUCCESS;
    }

    printf("FAILURE\n");
    return EXIT_FAILURE;
}
