
# Make Kernel and the gem5 Bridge Driver

This document outlines the steps to build a Linux kernel and its modules with the gem5-bridge driver for the ARM Ubuntu disk images. Below are separate instructions for Ubuntu 24.04 and Ubuntu 22.04.

**Note**: These dockerfiles assume that you are running on arm host to build them. If you are not on ARM host then you would need to use a cross compiler to make the kernel and the modules.

## Table of Contents

- [Ubuntu 24.04 Disk Image](#ubuntu-2404-disk-image)
- [Ubuntu 22.04 Disk Image](#ubuntu-2204-disk-image)

## **Ubuntu 24.04 Disk Image**

### Build the Docker Image

- Navigate to the `24.04-dockerfile` directory and build the Docker image:

  ```bash
  cd src/ubuntu-generic-diskimages/24.04-dockerfile
  docker build -t ubuntu-kernel-build .
  cd ..
  ```

### Build the Kernel and Modules

- Create a container from the built image:

  ```bash
  docker create --name kernel-builder ubuntu-kernel-build
  ```

- Start the container to build the kernel:

  ```bash
  docker start -a kernel-builder
  ```

- Copy the kernel and modules to the host:

  ```bash
  mkdir my-arm-6.8.12-kernel
  docker cp kernel-builder:/workspace/linux-6.8.0/vmlinux my-arm-6.8.12-kernel/
  docker cp kernel-builder:/workspace/output/lib/modules/6.8.12 my-arm-6.8.12-kernel/
  ```

- Clean up the container:

  ```bash
  docker rm kernel-builder
  ```

### Verify Output

- Check the contents of `my-arm-6.8.12-kernel`:

  ```bash
  ls my-arm-6.8.12-kernel/
  ```

  You should see:
  - `vmlinux` — The built kernel image.
  - `6.8.12/` — Directory containing the kernel modules.

### Add Kernel Modules to the Disk Image

- Add a Packer file provisioner to copy the modules to the disk image. The Packer script is located at `ubuntu-generic-diskimages/packer-scripts/arm-ubuntu.pkr.hcl`.
Make sure that this provisioner is added before the shell provisioner, as these files are used when the shell provisioner runs`post-installation.sh`:

  ```hcl
  provisioner "file" {
    destination = "/home/gem5"
    source      = "my-arm-6.8.12-kernel/6.8.12"
  }
  ```

- Add the following code snippet to the post-install script to move the modules into the correct location and regenerate the initramfs. The post-install script is located at `ubuntu-generic-diskimages/scripts/post-installation.sh`.
Make sure the modules are moved before using `gem5-bridge` or compiling benchmarks with `gem5-bridge`, i.e. add the snippet before the line `echo "Building and installing gem5-bridge (m5) and libm5"`:

  ```bash
  mv /home/gem5/6.8.12 /lib/modules/6.8.12
  depmod --quick -a 6.8.12
  update-initramfs -u -k 6.8.12
  ```

### Build the Disk Image

- Build the disk image using your build script:

  ```bash
  ./build-arm.sh 24.04
  ```

### Test with gem5

- Use the disk image and the kernel to run a gem5 filesystem simulation, ensuring the new kernel and modules are correctly set up.

## **Ubuntu 22.04 Disk Image**

### Build the Docker Image

- Navigate to the `22.04-dockerfile` directory and build the Docker image:

  ```bash
  cd src/ubuntu-generic-diskimages/22.04-dockerfile
  docker build -t ubuntu-22.04-kernel-build .
  cd ..
  ```

### Build the Kernel and Modules

- Create a container from the built image:

  ```bash
  docker create --name kernel-builder ubuntu-22.04-kernel-build
  ```

- Start the container to build the kernel:

  ```bash
  docker start -a kernel-builder
  ```

- Copy the kernel and modules to the host:

  ```bash
  mkdir my-arm-5.15.167-kernel
  docker cp kernel-builder:/workspace/linux-5.15.0/vmlinux my-arm-5.15.167-kernel/
  docker cp kernel-builder:/workspace/output/lib/modules/5.15.167 my-arm-5.15.167-kernel/
  ```

- Clean up the container:

  ```bash
  docker rm kernel-builder
  ```

### Verify Output

- Check the contents of `my-arm-5.15.167-kernel`:

  ```bash
  ls my-arm-5.15.167-kernel/
  ```

  You should see:
  - `vmlinux` — The built kernel image.
  - `5.15.167/` — Directory containing the kernel modules.

### Add Kernel Modules to the Disk Image

- Add a Packer file provisioner to copy the modules to the disk image. The Packer script is located at `ubuntu-generic-diskimages/packer-scripts/arm-ubuntu.pkr.hcl`.
Make sure that this provisioner is added before the shell provisioner as we will used these files in the `post-installation.sh`:

  ```hcl
  provisioner "file" {
    destination = "/home/gem5"
    source      = "my-arm-5.15.167-kernel/5.15.167"
  }
  ```

- Add the following code snippet to the post-install script to move the modules into the correct location and regenerate the initramfs. The post-install script is located at `ubuntu-generic-diskimages/scripts/post-installation.sh`.
Make sure the modules are moved before using `gem5-bridge` or compiling benchmarks with `gem5-bridge`, i.e. add the snippet before the line `echo "Building and installing gem5-bridge (m5) and libm5"`:

  ```bash
  mv /home/gem5/5.15.167 /lib/modules/5.15.167
  depmod --quick -a 5.15.167
  update-initramfs -u -k 5.15.167
  ```

### Build the Disk Image

- Build the disk image using your build script:

  ```bash
  ./build-arm.sh 22.04
  ```

### Test with gem5

- Use the disk image and the kernel to run a gem5 filesystem simulation, ensuring the new kernel and modules are correctly set up. See the bottom of this file for an example.

- You can use the following code snippet to use the disk image and kernel you made.

```python
image = DiskImageResource("/path/to/gem5-resources/src/ubuntu-generic-diskimages/arm-disk-image-22-04/arm-ubuntu")
image._root_partition = "2"

board.set_kernel_disk_workload(
    kernel=KernelResource("/path/to/gem5-resources/src/ubuntu-generic-diskimages/my-arm-5.15.167-kernel/vmlinux"),
    disk_image=image,
    bootloader=obtain_resource("arm64-bootloader-foundation"),
)
```
