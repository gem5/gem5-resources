# Start from Ubuntu 24.04 base image
FROM ubuntu:24.04

# Install necessary packages for kernel and module build
RUN apt update && apt install -y \
    build-essential \
    libncurses-dev \
    bison \
    flex \
    libssl-dev \
    libelf-dev \
    bc \
    wget \
    git \
    kmod

# Set the default working directory
WORKDIR /workspace

# Use bash as the default shell
CMD ["/bin/bash"]
