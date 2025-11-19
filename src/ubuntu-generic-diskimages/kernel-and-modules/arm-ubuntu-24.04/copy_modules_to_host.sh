#!/bin/bash

# Copyright (c) 2025 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

# This script assumes you are running it on an ARM host.

DOCKERFILE="./Dockerfile"
OUTPUT="my-arm-6.8.12-kernel"

# Build the Docker image
DOCKER_BUILDKIT=1 docker build --no-cache \
    --file "$DOCKERFILE" \
    --output "$OUTPUT" .

if [ $? -eq 0 ];
then
    echo "Build completed for Arm 24.04 kernel: Output directory is $OUTPUT"
else
    echo "Build failed for Arm 24.04 kernel"
fi