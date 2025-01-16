
# Make Kernel and the gem5 Bridge Driver

This document outlines the steps to build a Linux kernel and its modules with the gem5-bridge driver for the ARM Ubuntu disk images. Below are separate instructions for Ubuntu 24.04 and Ubuntu 22.04.

**Note**: These dockerfiles assume that you are running on arm host to build them. If you are not on ARM host then you would need to use a cross compiler to make the kernel and the modules.

## Table of Contents

- [Ubuntu 24.04 Disk Image](#ubuntu-2404-disk-image)
- [Ubuntu 22.04 Disk Image](#ubuntu-2204-disk-image)

## **Ubuntu 24.04 Disk Image**

### Build the Docker Image

The Docker image build process will copy the built kernel and modules to your host system, making them readily available for further use.

 Run the `make-arm-kernel.sh` script located in the directory `src/ubuntu-generic-diskimages`.
 Since we are building the kernel that is included by default in Ubuntu 24.04, you need to use the `24.04` argument with the script:

```bash
./make-arm-kernel.sh 24.04
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

The Docker image build process will copy the built kernel and modules to your host system, making them readily available for further use.

 Run the `make-arm-kernel.sh` script located in the directory `src/ubuntu-generic-diskimages`.
 Since we are building the kernel that is included by default in Ubuntu 22.04, you need to use the `22.04` argument with the script:

```bash
./make-arm-kernel.sh 22.04
```

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
