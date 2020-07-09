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
# Authors: Jason Lowe-Power, Ayaz Akram, Hoa Nguyen

""" Script to run a SPEC benchmark in full system mode with gem5.

    Inputs:
    * This script expects the following as arguments:
        ** kernel:
                  This is a positional argument specifying the path to
                  vmlinux.
        ** disk:
                  This is a positional argument specifying the path to the
                  disk image containing the installed SPEC benchmarks.
        ** cpu:
                  This is a positional argument specifying the name of the
                  detailed CPU model. The names of the available CPU models
                  are available in the getDetailedCPUModel(cpu_name) function.
                  The function should be modified to add new CPU models.
                  Currently, the available CPU models are:
                    - kvm: this is not a detailed CPU model, ideal for testing.
                    - o3: DerivO3CPU.
                    - atomic: AtomicSimpleCPU.
                    - timing: TimingSimpleCPU.
        ** benchmark:
                  This is a positional argument specifying the name of the
                  SPEC benchmark to run. Most SPEC benchmarks are available.
                  Please follow this link to check the availability of the
                  benchmarks. The working benchmark matrix is near the end
                  of the page:
         (SPEC 2006) https://gem5art.readthedocs.io/en/latest/tutorials/spec2006-tutorial.html#appendix-i-working-spec-2006-benchmarks-x-cpu-model-table
         (SPEC 2017) https://gem5art.readthedocs.io/en/latest/tutorials/spec2017-tutorial.html#appendix-i-working-spec-2017-benchmarks-x-cpu-model-table
        ** size:
                  This is a positional argument specifying the size of the
                  benchmark. The available sizes are: ref, test, train.
        ** --no-copy-logs:
                  This is an optional argument specifying the reports of
                  the benchmark run is not copied to the output folder.
                  The reports are copied by default.
        ** --allow-listeners:
                  This is an optional argument specifying gem5 to open GDB
                  listening ports. Usually, the ports are opened for debugging
                  purposes.
                  By default, the ports are off.
"""
import os
import sys

import m5
import m5.ticks
from m5.objects import *

import argparse

from system import MySystem


def writeBenchScript(dir, benchmark_name, size, output_path):
    """
    This method creates a script in dir which will be eventually
    passed to the simulated system (to run a specific benchmark
    at bootup).
    """
    input_file_name = '{}/run_{}_{}'.format(dir, benchmark_name, size)
    with open(input_file_name, "w") as f:
        f.write('{} {} {}'.format(benchmark_name, size, output_path))
    return input_file_name

def parse_arguments():
    parser = argparse.ArgumentParser(description=
                                "gem5 config file to run SPEC benchmarks")
    parser.add_argument("kernel", type = str, help = "Path to vmlinux")
    parser.add_argument("disk", type = str,
                  help = "Path to the disk image containing SPEC benchmarks")
    parser.add_argument("cpu", type = str, help = "Name of the detailed CPU")
    parser.add_argument("benchmark", type = str,
                        help = "Name of the SPEC benchmark")
    parser.add_argument("size", type = str,
                        help = "Available sizes: test, train, ref")
    parser.add_argument("-l", "--no-copy-logs", default = False,
                        action = "store_true",
                        help = "Not to copy SPEC run logs to the host system;"
                               "Logs are copied by default")
    parser.add_argument("-z", "--allow-listeners", default = False,
                        action = "store_true",
                        help = "Turn on ports;"
                               "The ports are off by default")
    return parser.parse_args()

def getDetailedCPUModel(cpu_name):
    '''
    Return the CPU model corresponding to the cpu_name.
    '''
    available_models = {"kvm": X86KvmCPU,
                        "o3": DerivO3CPU,
                        "atomic": AtomicSimpleCPU,
                        "timing": TimingSimpleCPU
                       }
    try:
        available_models["FlexCPU"] = FlexCPU
    except NameError:
        # FlexCPU is not defined
        pass
    # https://docs.python.org/3/library/stdtypes.html#dict.get
    # dict.get() returns None if the key does not exist
    return available_models.get(cpu_name)

def getBenchmarkName(benchmark_name):
    if benchmark_name.endswith("(base)"):
        benchmark_name = benchmark_name[:-6]
    return benchmark_name

def create_system(linux_kernel_path, disk_image_path, detailed_cpu_model):
    # create the system we are going to simulate
    system = MySystem(kernel = linux_kernel_path,
                      disk = disk_image_path,
                      num_cpus = 1, # run the benchmark in a single thread
                      no_kvm = False,
                      TimingCPUModel = detailed_cpu_model)

    # For workitems to work correctly
    # This will cause the simulator to exit simulation when the first work
    # item is reached and when the first work item is finished.
    system.work_begin_exit_count = 1
    system.work_end_exit_count = 1

    # set up the root SimObject and start the simulation
    root = Root(full_system = True, system = system)

    if system.getHostParallel():
        # Required for running kvm on multiple host cores.
        # Uses gem5's parallel event queue feature
        # Note: The simulator is quite picky about this number!
        root.sim_quantum = int(1e9) # 1 ms

    return root, system


def boot_linux():
    '''
    Output 1: False if errors occur, True otherwise
    Output 2: exit cause
    '''
    print("Booting Linux")
    exit_event = m5.simulate()
    exit_cause = exit_event.getCause()
    success = exit_cause == "m5_exit instruction encountered"
    if not success:
        print("Error while booting linux: {}".format(exit_cause))
        exit(1)
    print("Booting done")
    return success, exit_cause

def run_spec_benchmark():
    '''
    Output 1: False if errors occur, True otherwise
    Output 2: exit cause
    '''
    print("Start running benchmark")
    exit_event = m5.simulate()
    exit_cause = exit_event.getCause()
    success = exit_cause == "m5_exit instruction encountered"
    if not success:
        print("Error while running benchmark: {}".format(exit_cause))
        exit(1)
    print("Benchmark done")
    return success, exit_cause

def copy_spec_logs():
    '''
    Output 1: False if errors occur, True otherwise
    Output 2: exit cause
    '''
    print("Copying SPEC logs")
    exit_event = m5.simulate()
    exit_cause = exit_event.getCause()
    success = exit_cause == "m5_exit instruction encountered"
    if not success:
        print("Error while copying SPEC log files: {}".format(exit_cause))
        exit(1)
    print("Copying done")
    return success, exit_cause

if __name__ == "__m5_main__":
    args = parse_arguments()

    cpu_name = args.cpu
    benchmark_name = getBenchmarkName(args.benchmark)
    benchmark_size = args.size
    linux_kernel_path = args.kernel
    disk_image_path = args.disk
    no_copy_logs = args.no_copy_logs
    allow_listeners = args.allow_listeners

    if not no_copy_logs and not os.path.isabs(m5.options.outdir):
        print("Please specify the --outdir (output directory) of gem5"
              " in the form of an absolute path")
        print("An example: build/X86/gem5.opt --outdir /home/user/m5out/"
              " configs-spec-tests/run_spec ...")
        exit(1)

    output_dir = os.path.join(m5.options.outdir, "speclogs")

    # Get the DetailedCPU class from its name
    detailed_cpu = getDetailedCPUModel(cpu_name)
    if detailed_cpu == None:
        print("'{}' is not define in the config script.".format(cpu_name))
        print("Change getDeatiledCPUModel() in run_spec.py "
              "to add more CPU Models.")
        exit(1)

    if not benchmark_size in ["ref", "train", "test"]:
        print("Benchmark size must be one of the following: ref, train, test")
        exit(1)

    root, system = create_system(linux_kernel_path, disk_image_path,
                                 detailed_cpu)

    # Create and pass a script to the simulated system to run the reuired
    # benchmark
    system.readfile = writeBenchScript(m5.options.outdir, benchmark_name,
                                       benchmark_size, output_dir)

    # needed for long running jobs
    if not allow_listeners:
        m5.disableAllListeners()

    # instantiate all of the objects we've created above
    m5.instantiate()

    # booting linux
    success, exit_cause = boot_linux()

    # reset stats
    print("Reset stats")
    m5.stats.reset()

    # switch from KVM to detailed CPU
    if not cpu_name == "kvm":
        print("Switching to detailed CPU")
        system.switchCpus(system.cpu, system.detailed_cpu)
        print("Switching done")

    # running benchmark
    print("Benchmark: {}; Size: {}".format(benchmark_name, benchmark_size))
    success, exit_cause = run_spec_benchmark()

    # output the stats after the benchmark is complete
    print("Output stats")
    m5.stats.dump()

    if not no_copy_logs:
        # create the output folder
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # switch from detailed CPU to KVM
        if not cpu_name == "kvm":
            print("Switching to KVM")
            system.switchCpus(system.detailed_cpu, system.cpu)
            print("Switching done")

        # copying logs
        success, exit_cause = copy_spec_logs()
