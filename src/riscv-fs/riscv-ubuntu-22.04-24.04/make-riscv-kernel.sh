#!/bin/bash

# Ensure an argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 24.04"
    echo "         $0 22.04"
    exit 1
fi

# Set variables based on the argument
if [ "$1" == "24.04" ]; then
    DOCKERFILE="./24.04-dockerfile/Dockerfile"
    OUTPUT="my-riscv-6.8.12-kernel"
elif [ "$1" == "22.04" ]; then
    DOCKERFILE="./22.04-dockerfile/Dockerfile"
    OUTPUT="my-riscv-5.15.167-kernel"
else
    echo "Invalid version: $1"
    echo "Supported versions: 24.04, 22.04"
    exit 1
fi

# Build the Docker image
DOCKER_BUILDKIT=1 docker build --no-cache \
    --file "$DOCKERFILE" \
    --output "$OUTPUT" .

echo "Build completed for $1: Output directory is $OUTPUT"
