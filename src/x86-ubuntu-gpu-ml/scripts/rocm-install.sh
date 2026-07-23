#!/bin/bash

# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD 3-Clause

# The following instructions were obtained from the ROCm installation guide:
# https://rocm.docs.amd.com/en/docs-7.14.0/install/rocm.html

# Install additional packages needed by ROCm
sudo apt install -y libatomic1 libquadmath0

# Register ROCm repositories
# Download and install GPG key
sudo mkdir --parents --mode=0755 /etc/apt/keyrings

# ROCm release signing key
wget https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg -O - | \
        gpg --dearmor | sudo tee /etc/apt/keyrings/amdrocm.gpg > /dev/null

sudo tee /etc/apt/sources.list.d/rocm.list << EOF
deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] https://repo.amd.com/rocm/packages-multi-arch/ubuntu2604 stable main
EOF

sudo apt update

# Install ROCm packages
sudo apt -y install amdrocm7.14-gfx950
sudo apt -y install cmake
