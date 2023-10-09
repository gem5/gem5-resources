---
title: GCN3 HSA Agent Packet Test
tags:
    - x86
    - amdgpu
layout: default
permalink: resources/hsa-agent-pkt
shortdoc: >
    Resources to build a disk image with the GCN3 HSA Agent Packet workload.
---

# Resource: HSA Agent Packet Example

Based off of the Square resource in this repository, this resource serves as
an example for using an HSA Agent Packet to send commands to the GPU command
processor included in the GCN_X86 build of gem5.

The example command extracts the kernel's completion signal from the domain
of the command processor and the GPU's dispatcher. Initially this was a
workaround for the hipDeviceSynchronize bug, now fixed. The method of
waiting on a signal can be applied to other agent packet commands though.

Custom commands can be added to the command processor in gem5 to control
the GPU in novel ways.

## Compilation

To compile:

```
cd src/gpu/hsa-agent-pkt
docker run --rm -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID ghcr.io/gem5/gcn-gpu:v22-1 make gfx8-apu
```

The compiled binary can be found in `src/gpu/hsa-agent-pkt/bin`
