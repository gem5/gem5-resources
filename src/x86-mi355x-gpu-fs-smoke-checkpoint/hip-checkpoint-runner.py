#!/usr/bin/env python3

# Copyright (c) 2026 The Regents of The University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer; redistributions in
# binary form must reproduce the above copyright notice, this list of
# conditions and the following disclaimer in the documentation and/or other
# materials provided with the distribution; neither the name of the copyright
# holders nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written
# permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Prepare and resume the initialized HIP loader for the MI355X smoke kernel."""

import base64
import ctypes
import subprocess


def check(result, operation):
    if result != 0:
        raise RuntimeError(f"{operation} failed with HIP error {result}")


hip = ctypes.CDLL("libamdhip64.so.7")
hip.hipInit.argtypes = (ctypes.c_uint,)
hip.hipHostMalloc.argtypes = (
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_size_t,
    ctypes.c_uint,
)
hip.hipHostFree.argtypes = (ctypes.c_void_p,)
hip.hipHostFree.restype = ctypes.c_int
hip.hipModuleLoad.argtypes = (
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_char_p,
)
hip.hipModuleGetFunction.argtypes = (
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_void_p,
    ctypes.c_char_p,
)
hip.hipModuleLaunchKernel.argtypes = (
    ctypes.c_void_p,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_void_p),
)
hip.hipDeviceSynchronize.argtypes = ()

check(hip.hipInit(0), "hipInit")
value_pointer = ctypes.c_void_p()
check(
    hip.hipHostMalloc(ctypes.byref(value_pointer), ctypes.sizeof(ctypes.c_int), 0),
    "hipHostMalloc",
)
value = ctypes.c_int.from_address(value_pointer.value)
value.value = 41

print("HIP loader initialized; taking reusable GPU checkpoint", flush=True)
subprocess.run(("/sbin/m5", "checkpoint"), check=True)

encoded_kernel = subprocess.check_output(("/sbin/m5", "readfile"))
kernel_path = "/tmp/gpu-checkpoint-kernel.hsaco"
with open(kernel_path, "wb") as kernel_file:
    kernel_file.write(base64.b64decode(encoded_kernel, validate=True))

module = ctypes.c_void_p()
function = ctypes.c_void_p()
check(hip.hipModuleLoad(ctypes.byref(module), kernel_path.encode()), "hipModuleLoad")
check(
    hip.hipModuleGetFunction(ctypes.byref(function), module, b"_Z9incrementPi"),
    "hipModuleGetFunction",
)

kernel_argument = ctypes.c_void_p(value_pointer.value)
kernel_parameters = (ctypes.c_void_p * 1)(
    ctypes.cast(ctypes.byref(kernel_argument), ctypes.c_void_p)
)


def launch_and_synchronize():
    check(
        hip.hipModuleLaunchKernel(
            function,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            None,
            kernel_parameters,
            None,
        ),
        "hipModuleLaunchKernel",
    )
    check(hip.hipDeviceSynchronize(), "hipDeviceSynchronize")


launch_and_synchronize()
if value.value != 42:
    raise RuntimeError(f"GPU warmup returned {value.value}, expected 42")
value.value = 41
print("GPU dispatch warmed; taking CI checkpoint", flush=True)
subprocess.run(("/sbin/m5", "checkpoint"), check=True)

launch_and_synchronize()
if value.value != 42:
    raise RuntimeError(f"GPU kernel returned {value.value}, expected 42")

check(hip.hipHostFree(value_pointer), "hipHostFree")
print("GPU checkpoint restore test passed", flush=True)
