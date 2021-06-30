# -*- coding: utf-8 -*-
# Copyright (c) 2019 The Regents of the University of California.
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

""" Script to run PARSEC benchmarks with gem5.
    The script expects kernel, diskimage, cpu (kvm or timing),
    benchmark, benchmark size, and number of cpu cores as arguments.
    This script is best used if your disk-image has workloads tha have
    ROI annotations compliant with m5 utility. You can use the script in
    ../disk-images/parsec/ with the parsec-benchmark repo at
    https://github.com/darchr/parsec-benchmark.git to create a working
    disk-image for this script.
"""
import errno
import os
import sys
import time
import m5
import m5.ticks
from m5.objects import *

sys.path.append('gem5/configs/common/') # For the next line...
import SimpleOpts

from system import *

def writeBenchScript(dir, bench, size, num_cpus):
    """
    This method creates a script in dir which will be eventually
    passed to the simulated system (to run a specific benchmark
    at bootup).
    """
    file_name = '{}/run_{}'.format(dir, bench)
    bench_file = open(file_name, 'w+')
    bench_file.write('cd /home/gem5/parsec-benchmark\n')
    bench_file.write('source env.sh\n')
    bench_file.write('parsecmgmt -a run -p \
            {} -c gcc-hooks -i {} -n {}\n'.format(bench, size, num_cpus))

    # sleeping for sometime makes sure
    # that the benchmark's output has been
    # printed to the console
    bench_file.write('sleep 5 \n')
    bench_file.write('m5 exit \n')
    bench_file.close()
    return file_name

if __name__ == "__m5_main__":
    (opts, args) = SimpleOpts.parse_args()
    kernel, disk, cpu, benchmark, size, num_cpus = args

    if not cpu in ['kvm', 'timing']:
        m5.fatal("cpu not supported")

    # create the system
    system = MySystem(kernel, disk, cpu, int(num_cpus))

    # Exit from guest on workbegin/workend
    system.exit_on_work_items = True

    # Create and pass a script to the simulated system to run the reuired
    # benchmark
    system.readfile = writeBenchScript(m5.options.outdir, benchmark, size,
                                       num_cpus)

    # set up the root SimObject and start the simulation
    root = Root(full_system = True, system = system)

    if system.getHostParallel():
        # Required for running kvm on multiple host cores.
        # Uses gem5's parallel event queue feature
        # Note: The simulator is quite picky about this number!
        root.sim_quantum = int(1e9) # 1 ms

    #needed for long running jobs
    m5.disableAllListeners()

    # instantiate all of the objects we've created above
    m5.instantiate()

    globalStart = time.time()

    print("Running the simulation")
    print("Using cpu: {}".format(cpu))

    start_tick = m5.curTick()
    end_tick = m5.curTick()
    start_insts = system.totalInsts()
    end_insts = system.totalInsts()
    m5.stats.reset()

    exit_event = m5.simulate()

    if exit_event.getCause() == "workbegin":
        print("Done booting Linux")
        # Reached the start of ROI
        # start of ROI is marked by an
        # m5_work_begin() call
        print("Resetting stats at the start of ROI!")
        m5.stats.reset()
        start_tick = m5.curTick()
        start_insts = system.totalInsts()
        # switching to timing cpu if argument cpu == timing
        if cpu == 'timing':
            system.switchCpus(system.cpu, system.detailedCpu)
    else:
        print("Unexpected termination of simulation!")
        print()
        m5.stats.dump()
        end_tick = m5.curTick()
        end_insts = system.totalInsts()
        m5.stats.reset()
        print("Performance statistics:")

        print("Simulated time: %.2fs" % ((end_tick-start_tick)/1e12))
        print("Instructions executed: %d" % ((end_insts-start_insts)))
        print("Ran a total of", m5.curTick()/1e12, "simulated seconds")
        print("Total wallclock time: %.2fs, %.2f min" % \
                    (time.time()-globalStart, (time.time()-globalStart)/60))
        exit()

    # Simulate the ROI
    exit_event = m5.simulate()

    # Reached the end of ROI
    # Finish executing the benchmark with kvm cpu
    if exit_event.getCause() == "workend":
        # Reached the end of ROI
        # end of ROI is marked by an
        # m5_work_end() call
        print("Dump stats at the end of the ROI!")
        m5.stats.dump()
        end_tick = m5.curTick()
        end_insts = system.totalInsts()
        m5.stats.reset()
        # switching to timing cpu if argument cpu == timing
        if cpu == 'timing':
            system.switchCpus(system.timingCpu, system.cpu)
    else:
        print("Unexpected termination of simulation!")
        print()
        m5.stats.dump()
        end_tick = m5.curTick()
        end_insts = system.totalInsts()
        m5.stats.reset()
        print("Performance statistics:")

        print("Simulated time: %.2fs" % ((end_tick-start_tick)/1e12))
        print("Instructions executed: %d" % ((end_insts-start_insts)))
        print("Ran a total of", m5.curTick()/1e12, "simulated seconds")
        print("Total wallclock time: %.2fs, %.2f min" % \
                    (time.time()-globalStart, (time.time()-globalStart)/60))
        exit()

    # Simulate the remaning part of the benchmark
    exit_event = m5.simulate()

    print("Done with the simulation")
    print()
    print("Performance statistics:")

    print("Simulated time in ROI: %.2fs" % ((end_tick-start_tick)/1e12))
    print("Instructions executed in ROI: %d" % ((end_insts-start_insts)))
    print("Ran a total of", m5.curTick()/1e12, "simulated seconds")
    print("Total wallclock time: %.2fs, %.2f min" % \
                (time.time()-globalStart, (time.time()-globalStart)/60))
