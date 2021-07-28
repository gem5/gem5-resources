# Copyright (c) 2021 The Regents of the University of California.
# All Rights Reserved
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
This script is supposed to run full system simulation for RISCV targets.
It has been tested with classic memory system and Atomic,
TimingSimpleCPU, and MinorCPU so far.
"""

import time
import argparse

import m5
import m5.ticks
from m5.objects import *

from system import *

def parse_options():
    parser = argparse.ArgumentParser(description='Runs Linux fs test with'
                'RISCV.')
    parser.add_argument("bbl", help='Path to the bbl (berkeley bootloader)'
                                        'binary with kernel payload')
    parser.add_argument("disk", help="Path to the disk image to boot")
    parser.add_argument("cpu_type", help="The type of CPU in the system")
    parser.add_argument("num_cpus", type=int, help="Number of CPU cores")

    return parser.parse_args()

if __name__ == "__m5_main__":

    args = parse_options()

    # create the system we are going to simulate

    system = RiscvSystem(args.bbl, args.disk, args.cpu_type, args.num_cpus)

    # set up the root SimObject and start the simulation
    root = Root(full_system = True, system = system)

    # Uncomment for long-running jobs
    # and when the user does not
    # need to interact with the
    # simulated sytem

    # m5.disableAllListeners()

    # instantiate all of the objects we've created above
    m5.instantiate()

    globalStart = time.time()

    print("Running the simulation")
    exit_event = m5.simulate()

    if exit_event.getCause() == "m5_exit instruction encountered":
        print("The user has terminated the simulation using m5")
        exit(0)
    else:
        print("Simulation terminated without using m5")
        exit(1)
