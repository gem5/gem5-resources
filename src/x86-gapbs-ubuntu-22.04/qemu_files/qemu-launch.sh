qemu-system-x86_64 \
     -nographic -m 16384 -smp 8 \
     -device virtio-net-pci,netdev=eth0 -netdev user,id=eth0,hostfwd=tcp::5558-:22 \
     -drive file=x86_64-hpc-2204.img,format=raw \
     -drive if=none,id=cloud,file=cloud.img -device virtio-blk-pci,drive=cloud
