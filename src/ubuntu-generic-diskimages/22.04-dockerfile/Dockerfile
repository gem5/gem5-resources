FROM ubuntu:22.04

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
RUN chmod -R u+rwx  /workspace/linux-5.15.0
RUN cd /workspace/linux-5.15.0 && \
    chmod a+x debian/rules && \
    chmod a+x debian/scripts/* && \
    chmod a+x debian/scripts/misc/* && \
    chmod -R +x scripts/

RUN cd /workspace/linux-5.15.0 && \
    make defconfig && \
    make -j 32 && \
    make INSTALL_MOD_PATH=/workspace/output modules_install

RUN git clone https://github.com/nkrim/gem5.git --depth=1 --filter=blob:none --no-checkout --sparse --single-branch --branch=gem5-bridge && \
    cd gem5 && \
    git sparse-checkout add util/m5 && \
    git sparse-checkout add util/gem5_bridge && \
    git sparse-checkout add include && \
    git checkout

RUN cd gem5/util/gem5_bridge && \
    make KMAKEDIR=/workspace/linux-5.15.0 INSTALL_MOD_PATH=/workspace/output build install

RUN cd /workspace/output/lib/modules/5.15.167 && \
    rm -rf build source

CMD ["/bin/bash"]
