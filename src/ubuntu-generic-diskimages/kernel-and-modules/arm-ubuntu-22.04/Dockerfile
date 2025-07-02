FROM ubuntu:22.04 AS stage1

# Install necessary packages for kernel build
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
    kmod \
    apt-src \
    vim \
    curl \
    file

RUN sed -Ei 's/^# deb-src /deb-src /' /etc/apt/sources.list
RUN apt update
RUN mkdir /workspace 
RUN cd /workspace && apt source linux-image-unsigned-5.15.0-25-generic

RUN cd /workspace/linux-5.15.0 && \
    cd scripts && \
    chmod +x pahole-version.sh && \
    cd .. && \
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
    make KMAKEDIR=/workspace/linux-5.15.0 INSTALL_MOD_PATH=/workspace/output build install

RUN cd /workspace/output/lib/modules/5.15.168 && \
    rm -rf build source

FROM scratch AS export-stage
COPY --from=stage1 /workspace/output/lib/modules .
COPY --from=stage1 workspace/linux-5.15.0/vmlinux .