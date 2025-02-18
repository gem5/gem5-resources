FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt -y update
RUN apt -y upgrade
RUN apt -y install \
  binutils build-essential libtool texinfo gzip zip unzip patchutils curl git \
  make cmake ninja-build automake bison flex gperf grep sed gawk bc \
  zlib1g-dev libexpat1-dev libmpc-dev libglib2.0-dev libfdt-dev libpixman-1-dev \
  vim tmux python-is-python3 \
  libncurses-dev gawk openssl libssl-dev dkms libelf-dev libudev-dev libpci-dev libiberty-dev \
  gcc-12-riscv64-linux-gnu g++-12-riscv64-linux-gnu

RUN ln -s /usr/bin/riscv64-linux-gnu-cpp-12 /usr/bin/riscv64-linux-gnu-cpp & \
    ln -s /usr/bin/riscv64-linux-gnu-cpp-12 /usr/bin/riscv64-linux-gnu-cpp & \
    ln -s /usr/bin/riscv64-linux-gnu-g++-12 /usr/bin/riscv64-linux-gnu-g++ & \
    ln -s /usr/bin/riscv64-linux-gnu-gcc-12 /usr/bin/riscv64-linux-gnu-gcc & \
    ln -s /usr/bin/riscv64-linux-gnu-gcc-ar-12 /usr/bin/riscv64-linux-gnu-gcc-ar & \
    ln -s /usr/bin/riscv64-linux-gnu-gcc-nm-12 /usr/bin/riscv64-linux-gnu-gcc-nm & \
    ln -s /usr/bin/riscv64-linux-gnu-gcc-ranlib-12 /usr/bin/riscv64-linux-gnu-gcc-ranlib & \
    ln -s /usr/bin/riscv64-linux-gnu-gcov-12 /usr/bin/riscv64-linux-gnu-gcov & \
    ln -s /usr/bin/riscv64-linux-gnu-gcov-dump-12 /usr/bin/riscv64-linux-gnu-gcov-dump & \
    ln -s /usr/bin/riscv64-linux-gnu-gcov-tool-12 /usr/bin/riscv64-linux-gnu-gcov-tool & \
    ln -s /usr/bin/riscv64-linux-gnu-lto-dump-12 /usr/bin/riscv64-linux-gnu-lto-dump
