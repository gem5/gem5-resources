wget https://cloud-images.ubuntu.com/releases/22.04/release-20230616/ubuntu-22.04-server-cloudimg-arm64.img
qemu-img convert ubuntu-22.04-server-cloudimg-arm64.img -O raw ./arm64-hpc-2204.img
qemu-img resize -f raw arm64-hpc-2204.img +16G