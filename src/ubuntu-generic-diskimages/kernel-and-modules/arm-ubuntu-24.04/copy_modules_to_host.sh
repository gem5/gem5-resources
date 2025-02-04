#!/bin/bash

# Copyright (c) 2025 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

DOCKERFILE="./Dockerfile"
OUTPUT="my-arm-6.8.12-kernel"

# Build the Docker image
DOCKER_BUILDKIT=1 docker build --no-cache \
    --file "$DOCKERFILE" \
    --output "$OUTPUT" .

echo "Build completed for $1: Output directory is $OUTPUT"