#!/bin/bash

# Copyright (c) 2025 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

# Build and install the gem5-bridge (m5) binary, library, and headers
echo "Building and installing gem5-bridge (m5) and libm5."

# Ensure the ISA environment variable is set
if [ -z "$ISA" ]; then
  echo "Error: ISA environment variable is not set."
  exit 1
fi

# Just get the files we need
git clone https://github.com/gem5/gem5.git --depth=1 --filter=blob:none --no-checkout --sparse --single-branch --branch=stable
pushd gem5
# Checkout just the files we need
git sparse-checkout add util/m5
git sparse-checkout add util/gem5_bridge
git sparse-checkout add include
git checkout
# Install the headers globally so that other benchmarks can use them
cp -r include/gem5 /usr/local/include/

# Build the library and binary
pushd util/m5
scons build/${ISA}/out/m5
cp build/${ISA}/out/m5 /usr/local/bin/
cp build/${ISA}/out/libm5.a /usr/local/lib/
popd   # util/m5

if [ "${ISA}" = "x86" ]; then
    # We need to build the kernel module for x86
    # but not for ARM as we are copying the pre-built
    # kernel module for ARM from host to the disk images
    pushd util/gem5_bridge
    make build install
    depmod --quick
    popd
fi

popd   # gem5

# rename the m5 binary to gem5-bridge
mv /usr/local/bin/m5 /usr/local/bin/gem5-bridge
# Set the setuid bit on the m5 binary
chmod 4755 /usr/local/bin/gem5-bridge
chmod u+s /usr/local/bin/gem5-bridge

#create a symbolic link to the gem5 binary for backward compatibility
ln -s /usr/local/bin/gem5-bridge /usr/local/bin/m5

# delete the git repo for gem5
rm -rf gem5
echo "Done building and installing gem5-bridge (m5) and libm5."