# Copyright (c) 2021 The Regents of the University of California
# All rights reserved.
#
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

"""
"""

import argparse
import os
import time

import m5
import m5.ticks
from m5.objects import *

from system import *

supported_protocols = ["classic", "MI_example", "MESI_Two_Level",
                       "MOESI_CMP_directory"]
supported_cpu_types = ['kvm', 'atomic', 'simple', 'o3']

def parse_options():
    parser = argparse.ArgumentParser(description='For use with gem5. Runs a '
                'simple system through Linux boot. Expects the disk image to '
                'call the simulator exit event after boot. Only works with '
                'x86 ISA.')
    parser.add_argument("--allow_listeners", default=False,
                        action="store_true",
                        help="Listeners disabled by default")
    parser.add_argument("kernel", help="Path to the kernel binary to boot")
    parser.add_argument("disk", help="Path to the disk image to boot")
    parser.add_argument("cpu_type", choices=supported_cpu_types,
                        help="The type of CPU to use in the system")
    parser.add_argument("mem_sys", choices=supported_protocols,
                        help="Type of memory system or coherence protocol")
    parser.add_argument("num_cpus", type=int, help="Number of CPU cores")
    parser.add_argument("boot_type", choices=["init", "systemd"],
                        help="How to boot the kernel. Either to a simple init "
                        "script or all of the way through systemd")

    return parser.parse_args()

if __name__ == "__m5_main__":

    args = parse_options()

    # create the system we are going to simulate
    if args.mem_sys == "classic":
        system = MySystem(args.kernel, args.disk, args.cpu_type, args.num_cpus)
    else:
        system = MyRubySystem(args.kernel, args.disk, args.cpu_type,
                              args.mem_sys, args.num_cpus)

    if args.boot_type == "init":
        # Simply run "exit.sh"
        system.workload.command_line += ' init=/root/exit.sh'

    # set up the root SimObject and start the simulation
    root = Root(full_system = True, system = system)

    if system.getHostParallel():
        # Required for running kvm on multiple host cores.
        # Uses gem5's parallel event queue feature
        # Note: The simulator is quite picky about this number!
        root.sim_quantum = int(1e9) # 1 ms

    # Required for long-running jobs
    if not args.allow_listeners:
        m5.disableAllListeners()

    # instantiate all of the objects we've created above
    m5.instantiate()

    globalStart = time.time()

    print("Running the simulation")
    exit_event = m5.simulate()

    if exit_event.getCause() != "m5_exit instruction encountered":
        print("Failed to exit correctly")
        exit(1)
    else:
        print("Success!")
        exit(0)
