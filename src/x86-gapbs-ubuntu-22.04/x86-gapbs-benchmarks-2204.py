# Copyright (c) 2023 The Regents of the University of California.
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
Script to run GAPBS benchmarks with gem5. The script expects the
benchmark program and the simulation size to run. The input is in the format
  
The system is fixed with 2 CPU cores, MESI Two Level system cache and 3 GB
DDR4 memory. It uses the x86 board.

This script will count the total number of instructions executed
in the ROI. It also tracks how much wallclock and simulated time.

Usage:
------

```
scons build/X86/gem5.opt
./build/X86/gem5.opt \
    configs/example/gem5_library/x86-gabps-benchmarks.py \
    --benchmark  \
    --synthetic  \
    --size 
```
"""
# adapted from the original x86-gapbs-benchmarks.py example

import argparse
import time
import sys

import m5
from m5.objects import Root

from gem5.utils.requires import requires
from gem5.components.boards.x86_board import X86Board
from gem5.components.memory import DualChannelDDR4_2400
from gem5.components.processors.simple_switchable_processor import (
    SimpleSwitchableProcessor,
)
from gem5.components.processors.cpu_types import CPUTypes
from gem5.isas import ISA
from gem5.coherence_protocol import CoherenceProtocol
from gem5.resources.resource import Resource, CustomResource, DiskImageResource, AbstractResource, obtain_resource
from gem5.simulate.simulator import Simulator
from gem5.simulate.exit_event import ExitEvent
from gem5.utils.override import overrides



requires(
    isa_required=ISA.X86,
    coherence_protocol_required=CoherenceProtocol.MESI_TWO_LEVEL,
    kvm_required=True,
)

# Following are the list of benchmark programs for gapbs

benchmark_choices = ["cc", "bc", "tc", "pr", "bfs"]

synthetic_choices = ["0", "1"]

size_choices = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "USA-road-d.NY.gr",
]

parser = argparse.ArgumentParser(
    description="An example configuration script to run the gapbs benchmarks."
)

# The only positional argument accepted is the benchmark name in this script.

parser.add_argument(
    "--benchmark",
    type=str,
    required=True,
    help="Input the benchmark program to execute.",
    choices=benchmark_choices,
)

parser.add_argument(
    "--synthetic",
    type=str,
    required=True,
    help="Synthetic Graph:: 1: synthetic graph is True; 0: real graph",
    choices=synthetic_choices,
)

parser.add_argument(
    "--size",
    type=str,
    required=True,
    help="Graph Size:: If synthetic is True, then specify a size [1 .. 15]. \
    Otherwise, specify a graph name [USA-road-d.NY.gr]",
    choices=size_choices,
)

args = parser.parse_args()

# Setting up all the fixed system parameters here
# Caches: MESI Two Level Cache Hierarchy

from gem5.components.cachehierarchies.ruby.mesi_two_level_cache_hierarchy import (
    MESITwoLevelCacheHierarchy,
)

cache_hierarchy = MESITwoLevelCacheHierarchy(
    l1d_size="32kB",
    l1d_assoc=8,
    l1i_size="32kB",
    l1i_assoc=8,
    l2_size="256kB",
    l2_assoc=16,
    num_l2_banks=2,
)
# Memory: Dual Channel DDR4 2400 DRAM device.
# The X86 board only supports 3 GB of main memory.

memory = DualChannelDDR4_2400(size="3GB")

# Here we setup the processor. This is a special switchable processor in which
# a starting core type and a switch core type must be specified. Once a
# configuration is instantiated a user may call `processor.switch()` to switch
# from the starting core types to the switch core types. In this simulation
# we start with KVM cores to simulate the OS boot, then switch to the Timing
# cores for the command we wish to run after boot.

processor = SimpleSwitchableProcessor(
    starting_core_type=CPUTypes.KVM,
    #starting_core_type=CPUTypes.MINOR,
    switch_core_type=CPUTypes.TIMING,
    #switch_core_type=CPUTypes.O3,
    isa=ISA.X86,
    num_cores=2,
)

# Here we setup the board. The X86Board allows for Full-System X86 simulations

"""
Uncomment the X86Board_sda class and board setup when using Linux kernel version 5.15.36.
"""

# class X86Board_sda(X86Board):
#     @overrides(X86Board)
#     def get_disk_device(self):
#         return "/dev/sda"

# board = X86Board_sda(
#     clk_freq="3GHz",
#     processor=processor,
#     memory=memory,
#     cache_hierarchy=cache_hierarchy,
# )

board = X86Board(
    clk_freq="3GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
)

# Here we set the FS workload, i.e., gapbs benchmark program
# After simulation has ended you may inspect
# `m5out/system.pc.com_1.device` to the stdout, if any.

# After the system boots, we execute the benchmark program and wait till the
# ROI `workbegin` annotation is reached. We start collecting the number of
# committed instructions till ROI ends (marked by `workend`). We then finish
# executing the rest of the benchmark.

# GAPBS benchmarks can be run using a synthetic graph

if args.synthetic == "1":
    if args.size == "USA-road-d.NY.gr":
        print(
            "fatal: cannot use a real graph with --synthetic 1",
            file=sys.stderr,
        )
        exit(-1)

    command = f"./{args.benchmark} -g {args.size}\n"
else:
    command = f"./{args.benchmark} -sf ../{args.size}"

board.set_kernel_disk_workload(
    kernel=obtain_resource("x86-linux-kernel-5.4.49"),
    disk_image=DiskImageResource(
        local_path="./qemu_files/x86_64-hpc-2204.img",
        root_partition="1"
        ),
    readfile_contents=command,
)


def handle_workbegin():
    print("Done booting Linux")
    print("Resetting stats at the start of ROI!")
    m5.stats.reset()
    global start_tick
    start_tick = m5.curTick()
    processor.switch()
    yield False  # E.g., continue the simulation.


def handle_workend():
    print("Dump stats at the end of the ROI!")
    m5.stats.dump()
    yield True  # Stop the simulation. We're done.


simulator = Simulator(
    board=board,
    on_exit_event={
        ExitEvent.WORKBEGIN: handle_workbegin(),
        ExitEvent.WORKEND: handle_workend(),
    },
)

# We maintain the wall clock time.

globalStart = time.time()

print("Running the simulation")
print("Using KVM cpu")

# There are a few thihngs to note regarding the gapbs benchamrks. The first is
# that there are several ROI annotations in the code present in the disk image.
# These ROI begin and end calls are inside a loop. Therefore, we only simulate
# the first ROI annotation in details. The X86Board currently does not support
#  `work items started count reached`.

simulator.run()
end_tick = m5.curTick()
# Since we simulated the ROI in details, therefore, simulation is over at this
# point.

# Simulation is over at this point. We acknowledge that all the simulation
# events were successful.
print("All simulation events were successful.")

# We print the final simulation statistics.
print("Done with the simulation")
print()
print("Performance statistics:")

print(
    f"Simulated time in ROI: {(end_tick - start_tick) / 1000000000000.0:.4f}s"
)
print(
    "Ran a total of", simulator.get_current_tick() / 1e12, "simulated seconds"
)
print(
    "Total wallclock time: %.2fs, %.2f min"
    % (time.time() - globalStart, (time.time() - globalStart) / 60)
)