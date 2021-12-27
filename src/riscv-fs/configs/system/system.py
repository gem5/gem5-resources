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
#

import m5
from m5.objects import *
from m5.util import convert
from os import path

'''
This class creates a bare bones RISCV full system.

The targeted system is  based on SiFive FU540-C000.
Reference:
[1] https://sifive.cdn.prismic.io/sifive/b5e7a29c-
d3c2-44ea-85fb-acc1df282e21_FU540-C000-v1p3.pdf
'''

# Dtb generation code from configs/example/riscv/fs_linux.py
def generateMemNode(state, mem_range):
    node = FdtNode("memory@%x" % int(mem_range.start))
    node.append(FdtPropertyStrings("device_type", ["memory"]))
    node.append(FdtPropertyWords("reg",
        state.addrCells(mem_range.start) +
        state.sizeCells(mem_range.size()) ))
    return node

def generateDtb(system):
    """
    Autogenerate DTB. Arguments are the folder where the DTB
    will be stored, and the name of the DTB file.
    """
    state = FdtState(addr_cells=2, size_cells=2, cpu_cells=1)
    root = FdtNode('/')
    root.append(state.addrCellsProperty())
    root.append(state.sizeCellsProperty())
    root.appendCompatible(["riscv-virtio"])

    for mem_range in system.mem_ranges:
        root.append(generateMemNode(state, mem_range))

    sections = [*system.cpu, system.platform]

    for section in sections:
        for node in section.generateDeviceTree(state):
            if node.get_name() == root.get_name():
                root.merge(node)
            else:
                root.append(node)

    fdt = Fdt()
    fdt.add_rootnode(root)
    fdt.writeDtsFile(path.join(m5.options.outdir, 'device.dts'))
    fdt.writeDtbFile(path.join(m5.options.outdir, 'device.dtb'))

class RiscvSystem(System):

    def __init__(self, bbl, disk, cpu_type, num_cpus):
        super(RiscvSystem, self).__init__()

        # Set up the clock domain and the voltage domain
        self.clk_domain = SrcClockDomain()
        self.clk_domain.clock = '3GHz'
        self.clk_domain.voltage_domain = VoltageDomain()

        # DDR memory range starts from base address 0x80000000
        # based on [1]
        self.mem_ranges = [AddrRange(start=0x80000000, size='1GB')]

        # Create the main memory bus
        # This connects to main memory
        self.membus = SystemXBar(width = 64) # 64-byte width

        # Add a bad addr responder
        self.membus.badaddr_responder = BadAddr()
        self.membus.default = self.membus.badaddr_responder.pio

        # Set up the system port for functional access from the simulator
        self.system_port = self.membus.cpu_side_ports

        # Create the CPUs for our system.
        self.createCPU(cpu_type, num_cpus)

        # HiFive platform
        # This is based on a HiFive RISCV board and has
        # only a limited number of devices so far i.e.
        # PLIC, CLINT, UART, VirtIOMMIO
        self.platform = HiFive()

        # create and intialize devices currently supported for RISCV
        self.initDevices(self.membus, disk)

        # Create the cache heirarchy for the system.
        self.createCacheHierarchy()

        # Create the memory controller
        self.createMemoryControllerDDR3()

        # Set number of CPU cores
        self.platform.setNumCores(num_cpus)

        self.setupInterrupts()

        # using RiscvLinux as the base full system workload
        self.workload = RiscvLinux()

        # this is user passed berkeley boot loader binary
        # currently the Linux kernel payload is compiled into this
        # as well
        self.workload.object_file = bbl

        # Generate DTB (from configs/example/riscv/fs_linux.py)
        generateDtb(self)
        self.workload.dtb_filename = path.join(m5.options.outdir, 'device.dtb')
        # Default DTB address if bbl is bulit with --with-dts option
        self.workload.dtb_addr = 0x87e00000

        # Linux boot command flags
        kernel_cmd = [
            "console=ttyS0",
            "root=/dev/vda",
            "ro"
        ]
        self.workload.command_line = " ".join(kernel_cmd)

    def createCPU(self, cpu_type, num_cpus):
        if cpu_type == "atomic":
            self.cpu = [AtomicSimpleCPU(cpu_id = i)
                              for i in range(num_cpus)]
            self.mem_mode = 'atomic'
        elif cpu_type == "simple":
            self.cpu = [TimingSimpleCPU(cpu_id = i)
                        for i in range(num_cpus)]
            self.mem_mode = 'timing'
        elif cpu_type == "minor":
            self.cpu = [MinorCPU(cpu_id = i)
                        for i in range(num_cpus)]
            self.mem_mode = 'timing'
        else:
            m5.fatal("No CPU type {}".format(cpu_type))

        for cpu in self.cpu:
            cpu.createThreads()


    def createCacheHierarchy(self):
        class L1Cache(Cache):
            """Simple L1 Cache with default values"""

            assoc = 8
            size = '32kB'
            tag_latency = 1
            data_latency = 1
            response_latency = 1
            mshrs = 16
            tgts_per_mshr = 20
            writeback_clean = True

            def __init__(self):
                super(L1Cache, self).__init__()

        for cpu in self.cpu:
            # Create a very simple cache hierarchy

            # Create an L1 instruction, data and mmu cache
            cpu.icache = L1Cache()
            cpu.dcache = L1Cache()
            cpu.mmucache = L1Cache()

            # Connecting icache and dcache to memory bus and cpu
            cpu.icache.mem_side = self.membus.cpu_side_ports
            cpu.dcache.mem_side = self.membus.cpu_side_ports

            cpu.icache.cpu_side = cpu.icache_port
            cpu.dcache.cpu_side = cpu.dcache_port

            # Need a new crossbar for mmucache

            cpu.mmucache.mmubus = L2XBar()

            cpu.mmucache.cpu_side = cpu.mmucache.mmubus.mem_side_ports
            cpu.mmucache.mem_side = self.membus.cpu_side_ports

            # Connect the itb and dtb to mmucache
            cpu.mmu.connectWalkerPorts(
                cpu.mmucache.mmubus.cpu_side_ports, cpu.mmucache.mmubus.cpu_side_ports)


    def setupInterrupts(self):
        for cpu in self.cpu:
            # create the interrupt controller CPU and connect to the membus
            cpu.createInterruptController()


    def createMemoryControllerDDR3(self):
        self.mem_cntrls = [
            MemCtrl(dram = DDR3_1600_8x8(range = self.mem_ranges[0]),
                    port = self.membus.mem_side_ports)
        ]

    def initDevices(self, membus, disk):

        self.iobus = IOXBar()

        # Set the frequency of RTC (real time clock) used by
        # CLINT (core level interrupt controller).
        # This frequency is 1MHz in SiFive's U54MC.
        # Setting it to 100MHz for faster simulation (from riscv/fs_linux.py)
        self.platform.rtc = RiscvRTC(frequency=Frequency("100MHz"))

        # RTC sends the clock signal to CLINT via an interrupt pin.
        self.platform.clint.int_pin = self.platform.rtc.int_pin

        # VirtIOMMIO
        image = CowDiskImage(child=RawDiskImage(read_only=True), read_only=False)
        image.child.image_file = disk
        # using reserved memory space
        self.platform.disk = MmioVirtIO(
            vio=VirtIOBlock(image=image),
            interrupt_id=0x8,
            pio_size = 4096,
            pio_addr=0x10008000
        )

        # From riscv/fs_linux.py
        uncacheable_range = [
            *self.platform._on_chip_ranges(),
            *self.platform._off_chip_ranges()
        ]
        # PMA (physical memory attribute) checker is a hardware structure
        # that ensures that physical addresses follow the memory permissions

        # PMA checker can be defined at system-level (system.pma_checker)
        # or MMU-level (system.cpu[0].mmu.pma_checker). It will be resolved
        # by RiscvTLB's Parent.any proxy

        for cpu in self.cpu:
            cpu.mmu.pma_checker =  PMAChecker(uncacheable=uncacheable_range)

        self.bridge = Bridge(delay='50ns')
        self.bridge.mem_side_port = self.iobus.cpu_side_ports
        self.bridge.cpu_side_port = self.membus.mem_side_ports
        self.bridge.ranges = self.platform._off_chip_ranges()

        # Connecting on chip and off chip IO to the mem
        # and IO bus
        self.platform.attachOnChipIO(self.membus)
        self.platform.attachOffChipIO(self.iobus)

        # Attach the PLIC (platform level interrupt controller)
        # to the platform. This initializes the PLIC with
        # interrupt sources coming from off chip devices
        self.platform.attachPlic()
