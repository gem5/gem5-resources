# Start from Ubuntu 24.04 base image
FROM ubuntu:24.04 AS stage1

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


RUN sed -i 's/^Types: deb$/Types: deb deb-src/' /etc/apt/sources.list.d/ubuntu.sources
RUN apt update
RUN mkdir /workspace 
RUN cd /workspace && apt source linux-image-unsigned-6.8.0-47-generic

RUN cd /workspace/linux-6.8.0 && \
    make defconfig && \
    make -j $(nproc) && \
    make INSTALL_MOD_PATH=/workspace/output modules_install

RUN git clone https://github.com/gem5/gem5.git --depth=1 --filter=blob:none --no-checkout --sparse --single-branch --branch=stable && \
    cd gem5 && \
    git sparse-checkout add util/m5 && \
    git sparse-checkout add util/gem5_bridge && \
    git sparse-checkout add include && \
    git checkout

RUN cd gem5/util/gem5_bridge && \
    make KMAKEDIR=/workspace/linux-6.8.0 INSTALL_MOD_PATH=/workspace/output build install

RUN cd /workspace/output/lib/modules/6.8.12 && \
    rm -rf build

FROM scratch AS export-stage
COPY --from=stage1 /workspace/output/lib/modules .
COPY --from=stage1 /workspace/linux-6.8.0/vmlinux .