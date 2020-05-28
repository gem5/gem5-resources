# -*- coding: utf-8 -*-
# Copyright (c) 2016 Jason Lowe-Power
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
#
# Authors: Jason Lowe-Power

"""
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

SimpleOpts.set_usage(
    "usage: %prog [options] kernel disk cpu_type mem_sys num_cpus boot_type")

SimpleOpts.add_option("--allow_listeners", default=False, action="store_true",
                      help="Listeners disabled by default")

if __name__ == "__m5_main__":
    (opts, args) = SimpleOpts.parse_args()

    if len(args) != 6:
        SimpleOpts.print_help()
        m5.fatal("Bad arguments")

    kernel, disk, cpu_type, mem_sys, num_cpus, boot_type = args
    num_cpus = int(num_cpus)

    # create the system we are going to simulate
    ruby_protocols = [ "MI_example", "MESI_Two_Level", "MOESI_CMP_directory"]
    if mem_sys == "classic":
        system = MySystem(kernel, disk, cpu_type, num_cpus, opts)
    elif mem_sys in ruby_protocols:
        system = MyRubySystem(kernel, disk, cpu_type, mem_sys, num_cpus, opts)
    else:
        m5.fatal("Bad option for mem_sys, should be "
        "{}, or 'classic'".format(', '.join(ruby_protocols)))

    if boot_type == "init":
        # Simply run "exit.sh"
        system.workload.command_line += ' init=/root/exit.sh'
    else:
        if boot_type != "systemd":
            SimpleOpts.print_help()
            m5.fatal("Bad option for boot_type. init or systemd.")

    # set up the root SimObject and start the simulation
    root = Root(full_system = True, system = system)

    if system.getHostParallel():
        # Required for running kvm on multiple host cores.
        # Uses gem5's parallel event queue feature
        # Note: The simulator is quite picky about this number!
        root.sim_quantum = int(1e9) # 1 ms

    # Required for long-running jobs
    if not opts.allow_listeners:
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
