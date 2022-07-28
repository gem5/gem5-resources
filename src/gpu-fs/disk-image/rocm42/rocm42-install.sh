# Copyright (c) 2022 Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Allowing services to restart while updating some
# libraries.
sudo apt install -y debconf-utils
sudo debconf-get-selections | grep restart-without-asking > libs.txt
sed -i 's/false/true/g' libs.txt
while read line; do echo $line | sudo debconf-set-selections; done < libs.txt
sudo rm libs.txt
##

# Installing packages needed to build ROCm applications
sudo apt -y update
sudo apt -y upgrade
sudo apt -y install build-essential git m4 scons zlib1g zlib1g-dev \
    libprotobuf-dev protobuf-compiler libprotoc-dev libgoogle-perftools-dev \
    python3-dev python-is-python3 doxygen libboost-all-dev \
    libhdf5-serial-dev python3-pydot libpng-dev libelf-dev pkg-config gdb

# Requirements for ROCm itself
sudo apt -y install cmake mesa-common-dev libgflags-dev libgoogle-glog-dev

# Needed to get ROCm repo, build packages
sudo apt -y install wget gnupg2 rpm

wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -

# Modify apt sources to pull from ROCm 4.2 repository only
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/4.2/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt -y update
sudo apt -y install libnuma-dev

# Install the ROCm-dkms source
sudo apt -y install initramfs-tools
sudo apt -y install rocm-dkms

# Install kernel 5.4.0 required by ROCm and headers to build DKMS package
# Use unsigned kernel to avoid extra step of signing amdgpu DKMS package
# Also install extra modules to get amd_iommu_v2 module amdgpu depends on
sudo apt -y install linux-image-unsigned-5.4.0-105-generic
sudo apt -y install linux-modules-extra-5.4.0-105-generic
sudo apt -y install linux-headers-5.4.0-105-generic

# Extract a kernel that gem5 can boot from
sudo wget -O /root/extract-vmlinux https://raw.githubusercontent.com/torvalds/linux/master/scripts/extract-vmlinux
sudo chmod +x /root/extract-vmlinux
sudo /root/extract-vmlinux /boot/vmlinuz-5.4.0-105-generic > /boot/vmlinux-5.4.0-105-generic

sudo cp -v '/home/gem5/serial-getty@.service' /lib/systemd/system/

# Download inputs for gem5 benchmarks
mkdir -p /home/gem5/data/pannotia
wget http://dist.gem5.org/dist/develop/datasets/pannotia/bc/1k_128k.gr -O /home/gem5/data/pannotia/1k_128k.gr
wget http://dist.gem5.org/dist/develop/datasets/pannotia/pagerank/coAuthorsDBLP.graph -O /home/gem5/data/pannotia/coAuthorsDBLP.graph

# Downloard rodinia 3.0 hip
sudo apt -y install git
cd /home/gem5
git clone https://github.com/ROCm-Developer-Tools/HIP-Examples.git
cd HIP-Examples/
git checkout c7e197d62a6ff327826f9e7279148cd66bfa2218
