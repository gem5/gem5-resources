# gem5 amd64 Ubuntu 22.04 ROCm disk image

This README describes how to use the contents of this directory to create a gem5 disk image capable of running GPU applications. The disk installs ROCm 5.4.2 and third party libraries to cover most gem5 simulation requirements. The main libraries consist of essential HIP libraries for GPU simulations in gem5 such as rocBLAS, hipCUB, hipSPARSE, and rocTHRUST. The third party library installs include PyTorch 2.0.1 and TensorFlow 2.11.0 to simulate ML/AI applications on the GPU.

## Building the Disk Image

This README is following steps used to build the x86-64 Ubuntu gapbs 22.04 disk image from
https://github.com/gem5/gem5-resources/pull/6
and https://github.com/takekoputa/hpc-disk-image/tree/main ,

Building a disk image with GPU compute libraries requires many dependencies. It is expected to take approximately 2 hours to create the disk image depending on download speed and host machine performance. Check the advanced notes for tips on improving this time if needed or desired. You will need 36GB of disk space to create the image including Ubuntu 22.04 and all of the ROCm/GPU libraries provided by this script.

How to build GPU image:
1. Obtain Packer by running `packer-obtain.sh` in the packer_files directory.
   - This will download a specific packer version compatible with the json files in this directory
2. Obtain the X86 Ubuntu disk image by running `get-disk-img.sh` in the qemu_files directory.
   - This script will download a cloud disk image which is about 2GB in size and resize by +33GB it to fit all GPU libraries, math libraries, python libraries, and their friends.
3. The image requires running the cloud disk image and remote login to install the GPU packages
   - The assumed current working directory is qemu_files.
   - It is recommended to create a new ssh key connect to the disk image
   - Create a new key as follows:
       - `ssh-keygen -t rsa -N "" -C "ubuntu@localhost" -f rocm5_key`
       - Note that the output name of rocm5_key corresponds to the name in `packer_files/x86_64-rocm.json`
   - Tell the cloud image about your key
       - Copy and paste the output of `cat rocm5_key.pub` into cloud.txt, replacing '(edit me)'
4. Create "cloud.img" with `make-cloud-img.sh`.
   - Run `./make-cloud-img.sh` which will create a small mini disk with your key on the main image.
5. Launch a QEMU instance with `qemu-launch.sh`.
   - This will take over the terminal. Open a new terminal window / tmux pane / etc. to continue to step 6.
6. Open a new terminal and add the ssh key.
   - Be sure to `cd` into the qemu_files directory in the same gem5-resources
   - `ssh-add rocm5_key`
   - If you see an error here, see ssh agent related issues
7. Run Packer with `./packer build x86_64-rocm.json` in the packer_files directory.
   - This will try to connect to the qemu instance in step 5.
   - This should not take more than 20 seconds, see issues below for common problems.
8. Power off the QEMU instance from qemu_files directory.
   - Make sure not to use the disk in gem5 before powering off.
   - Run ./qemu-logout from the qemu_files directory


## Using the Disk Image in legacy gem5 configs

The disk image will be created in the qemu_files directory with the name `ubuntu2204-rocm542.img`. Assuming this gem5-resources directory is located within a gem5 clone, similar to a standard gem5 build:

1. Build the VEGA_X86 GPU ISA
   - scons -j`nproc` build/VEGA_X86/gem5.opt
2. Specify this disk image, kernel, and the simulated GPU mmio trace to gem5. Examples:
   - build/VEGA_X86/gem5.opt configs/example/gpufs/vega10_kvm.py --disk-image gem5-resources/src/x86-ubuntu-22.04-rocm-5.4.2/qemu_files/ubuntu2204-rocm542.img  --kernel gem5-resources/src/x86-ubuntu-22.04-rocm-5.4.2/qemu_files/vmlinux-5.15.0-75-generic --gpu-mmio-trace gem5-resources/src/gpu-fs/vega_mmio.log --app some/where/my_great_hip_application --opts="1234"
   - build/VEGA_X86/gem5.opt configs/example/gpufs/vega10_kvm.py --disk-image gem5-resources/src/x86-ubuntu-22.04-rocm-5.4.2/qemu_files/ubuntu2204-rocm542.img  --kernel gem5-resources/src/x86-ubuntu-22.04-rocm-5.4.2/qemu_files/vmlinux-5.15.0-75-generic --gpu-mmio-trace gem5-resources/src/gpu-fs/vega_mmio.log --app some/where/pytorch_app.sh
   - build/VEGA_X86/gem5.opt configs/example/gpufs/vega10_kvm.py --disk-image gem5-resources/src/x86-ubuntu-22.04-rocm-5.4.2/qemu_files/ubuntu2204-rocm542.img  --kernel gem5-resources/src/x86-ubuntu-22.04-rocm-5.4.2/qemu_files/vmlinux-5.15.0-75-generic --gpu-mmio-trace gem5-resources/src/gpu-fs/vega_mmio.log --app some/where/tensorflow_app.sh


## Issues


### Packer hangs

To show debug messages when running Packer the PACKER_LOG environment variable:
`PACKER_LOG=1 ./packer build x86_64-rocm.json`

### SSH key denied

Make sure ssh-agent is running on the terminal you are trying to connect from:
`eval $(ssh-agent)`

You may try to connect manually which may add the rocm5_key to ~/.ssh/known_hosts
`ssh -p 5558 ubuntu@localhost -i ../qemu_files/rocm5_key`

# Advanced notes and developer notes

## Disk image iteration performance

The DKMS and linux-kernel-extra-`uname -r` take a long time. Editing the "-smp 8" value in qemu_files/qemu-launch.sh to something close to your `nproc` will help with iterating on building this disk image.

## Getting the vmlinux kernel resource

The amdgpu driver is a DKMS package. This means it must be built from source against the Linux kernel running inside the cloud image. This means the kernel running in the cloud image at the time of install must be the same kernel run in gem5.  In other words, you cannot run just any kernel with the --kernel parameter (legacy config) and expect it to work. There is a safeguard for this in gem5 legacy configs and in *this* directory's packer.

The kernel is provided as a resource, however if you need to get the kernel for gem5 manually, follow these instructions:

The disk image must have the Linux kernel source headers installed for the Linux kernel that is current running. This is probably handled by the amdgpu-dkms apt package, but to be explicit, this package is needed:

sudo apt install linux-headers-`uname -r`

This will create a directory in /usr/src/linux-headers-`uname -r`. Next run:

/usr/src/linux-headers-`uname -r`/scripts/extract-vmlinux /boot/vmlinuz-`uname -r` > /home/ubuntu/vmlinux-`uname -r`

If you see an issue with this, the most likely cause is missing the apt package to decompress vmlinuz. The most common missing package is lz4 which can be installed with:

`sudo apt install lz4`
