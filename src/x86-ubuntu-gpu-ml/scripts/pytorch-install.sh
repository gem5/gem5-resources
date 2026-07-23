#!/bin/bash

# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD 3-Clause

# Note about pip: This disk is created for the express purpose of being run in
# gem5 and is therefore effectively sandboxed enough that we can use the pip
# option --break-system-packages. If you plan to modify this disk image with
# pip packages that might conflict, it is up to you to resolve the conflicts.

# See https://pytorch.org/.
# Build: 2.12.0
# OS: Linux
# Package: Pip
# Language: Python
# Compute Platform: ROCm 7.2
sudo apt update && sudo apt -y install pip

# /tmp is tmpfs (limited to ~4 GB) which is too small for the torch ROCm wheel.
# Redirect pip's temp download dir to the root filesystem.
export TMPDIR=/root/pip-tmp
mkdir -p $TMPDIR

pip install --break-system-packages --no-cache-dir --resume-retries 5 torch torchvision \
    --index-url https://download.pytorch.org/whl/rocm7.2

rm -rf $TMPDIR

# Try and import pytorch, exit failure if it does not exist
python3 -c "import torch"
if [ "$?" != "0" ]; then
    echo "Pytorch installation failed. See log."
    exit 1
fi
