#!/bin/bash
set -e

# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD 3-Clause

# The following instructions were obtained from the ROCm installation guide:
# https://rocm.docs.amd.com/en/latest/install/rocm.html

export DEBIAN_FRONTEND=noninteractive

# Install additional packages needed by ROCm
sudo apt-get install -y libatomic1 libquadmath0

# Register ROCm repositories
# Download and install GPG key
sudo mkdir --parents --mode=0755 /etc/apt/keyrings

# ROCm release signing key
wget https://stable.repo.amd.com/rocm/gpg/packages.gpg -O - | \
        gpg --dearmor | sudo tee /etc/apt/keyrings/amdrocm.gpg > /dev/null

sudo tee /etc/apt/sources.list.d/amdrocm-stable.sources << EOF
X-Repo-Id: amdrocm-stable
Types: deb
URIs: https://stable.repo.amd.com/rocm/core/packages/ubuntu2604/
Suites: stable
Components: main
Architectures: amd64
Signed-By: /etc/apt/keyrings/amdrocm.gpg
Enabled: yes
EOF

sudo apt-get update

# Install ROCm packages
sudo apt-get install -y amdrocm10.0-gfx950
sudo apt-get install -y cmake
