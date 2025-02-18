---
title: Building the base x86-ubuntu and arm-ubuntu image for Ubuntu 22.04 and 24.04
authors:
    - Harshil Patel
---

This document provides instructions for creating the **x86-ubuntu** and **arm-ubuntu** disk images. The images can be built for **Ubuntu 22.04** or **Ubuntu 24.04**.

## Directory Structure

- **`files/`**: Contains files that are copied to the disk image.
- **`kernel-and-modules/`**: Contains scripts and Dockerfiles necessary for building the default kernel in Ubuntu 22.04 and 24.04 for ARM disk images. Each subdirectory corresponds to a specific Ubuntu version:
  - **`arm-ubuntu-22.04/`**: Contains a Dockerfile and `copy_modules_to_host.sh` for building the default Ubuntu 22.04 kernel and modules with the `gem5-bridge` module installed.
  - **`arm-ubuntu-24.04/`**: Contains a Dockerfile and `copy_modules_to_host.sh` for building the default Ubuntu 24.04 kernel and modules with the `gem5-bridge` module installed.

  After building the Dockerfile, you can retrieve the kernel and modules on your host using the `copy_modules_to_host.sh` script.
- **`scripts/`**: Contains scripts that run on the disk image after installation.
  - **`disable-network.sh`**: Disables networking by renaming the Netplan configuration file (`.yaml` → `.yaml.bak`) and disabling network services in systemd. Disabling network decreases boot time by removing the 2 minute wait for network service to get online in systemd.
  - **`disable-systemd-services-x86.sh`**: Disables non-essential systemd services for x86 disk images to reduce boot time in gem5 simulations.
  - **`extract-x86-kernel.sh`**: Extracts the kernel from the x86 disk image and moves it to `/home/gem5`. Packer then copies the extracted kernel from the disk image to the host.
  - **`increase-system-entropy-for-arm-disk.sh`**: Uses `haveged` to increase system entropy for ARM disk images, reducing boot delays caused by low entropy.
  - **`install-common-packages.sh`**: Installs necessary packages common to all disk images.
  - **`install-gem5-bridge.sh`**: Clones and builds `gem5-bridge`, allowing the disk image to use `m5ops` commands. For more information about using `m5ops` you can take a look at <https://bootcamp.gem5.org/#02-Using-gem5/03-running-in-gem5>.
  - **`install-user-benchmarks.sh`**: User-editable script for installing custom benchmarks.
  - **`install-user-packages.sh`**: User-editable script for installing additional packages beyond those in `install-common-packages.sh`.
  - **`update-gem5-init.sh`**: Updates the `init` file with `gem5_init.sh` from the `files` directory. The `gem5_init.sh` script updates the `init` script that is run when ubuntu boots to include the `no_systemd` kernel arg, initialize the `gem5-bridge` driver and call an exit event indicating that the kernel has booted.
  - **`update-modules-arm-22.04.sh`**: Installs kernel modules built via the Dockerfile in `kernel-and-modules/arm-ubuntu-22.04` for the ARM 22.04 disk image.
  - **`update-modules-arm-24.04.sh`**: Installs kernel modules built via the Dockerfile in `kernel-and-modules/arm-ubuntu-24.04` for the ARM 24.04 disk image.

- **`http/`**: Contains Ubuntu cloud-init autoinstall files for different architectures and versions.
  - `arm-22-04/`: Autoinstall files for ARM Ubuntu 22.04.
  - `arm-24-04/`: Autoinstall files for ARM Ubuntu 24.04.
  - `x86/`: Autoinstall files for x86 Ubuntu 22.04 and 24.04.

- **Disk Image Output Directories**:
  - `x86-disk-image-22.04/`: x86 Ubuntu 22.04 disk image output.
  - `x86-disk-image-24.04/`: x86 Ubuntu 24.04 disk image output.
  - `arm-disk-image-22.04/`: ARM Ubuntu 22.04 disk image output.
  - `arm-disk-image-24.04/`: ARM Ubuntu 24.04 disk image output.

## ARM Image Specific Requirements

To successfully build and boot an ARM disk image, you need to prepare the **kernel modules** and an **EFI boot file**.

### **Building and Installing Kernel Modules (ARM)**

Since the ARM disk image requires the `gem5-bridge` module to enable running `gem5-bridge` commands without using `sudo` inside gem5 simulations, we must build the kernel modules before running the Packer script.

#### **Steps to Build the Kernel and Modules**

1. **Navigate to the Appropriate Directory**
   Change to the directory corresponding to the Ubuntu version of the disk image you are building:

   ```sh
   cd kernel-and-modules/arm-ubuntu-22.04  # For Ubuntu 22.04
   cd kernel-and-modules/arm-ubuntu-24.04  # For Ubuntu 24.04
   ```

2. **Run the `copy_modules_to_host.sh` Script**
   This script builds the kernel and modules inside a Docker container and then copies them to the host machine:

   ```sh
   ./copy_modules_to_host.sh
   ```

3. **Verify the Output Directory**
   After running the script, a directory named **`my-arm-<kernel_version>-kernel`** will be created in the `kernel-and-modules` directory. This directory contains:

   - `vmlinux`: The built kernel (**used in gem5 simulations but not copied onto the built disk image**).
   - A subdirectory containing all kernel modules, including `gem5-bridge`.

### **Generating the EFI Boot File**

The ARM disk image requires an **EFI file** to boot in qemu. Running `build-arm.sh` automatically generates this file.This is provided as `flash0.img` in the Packer configuration.

To generate `flash0.img` manually, run the following commands in the `files/` directory:

```bash
dd if=/dev/zero of=flash0.img bs=1M count=64
dd if=/usr/share/qemu-efi-aarch64/QEMU_EFI.fd of=flash0.img conv=notrunc
```

## Building the Disk Image

- **For x86**:
  Run `build-x86.sh` with either `22.04` or `24.04` as an argument to build the respective x86 disk image in the `ubuntu-generic-diskimages` directory.

- **For ARM**:
  Run `build-arm.sh` with `22.04` or `24.04` to build the respective ARM disk image in `ubuntu-generic-diskimages`.

  **ARM Build Assumption**:
  The build assumes execution on an **ARM machine**, as it uses KVM for virtualization. If running on a non-ARM host, update `build-arm.sh` by setting `"use_kvm=false"` in the `./packer build` command:

  ```bash
  ./packer build -var "use_kvm=false" -var "ubuntu_version=${ubuntu_version}" ./packer-scripts/arm-ubuntu.pkr.hcl
  ```

  You would also need to update the isa of the packer binary being downloaded in the `build-arm.sh` file. To download the `amd64` packer binary you can update the section that downloads the packer binary in `build-arm.sh` file to the following:

  ```bash
  if [ ! -f ./packer ]; then
      wget https://releases.hashicorp.com/packer/${PACKER_VERSION}/packer_${PACKER_VERSION}_linux_amd64.zip;
      unzip packer_${PACKER_VERSION}_linux_amd64.zip;
      rm packer_${PACKER_VERSION}_linux_amd64.zip;
  fi
  ```

  The `build-arm.sh` script downloads the Packer binary, initializes Packer, and builds the disk image.

## Kernel Extraction (x86 Only)

For **x86 disk images**, the kernel is extracted as part of the post-installation process.

- The extracted kernel is stored in `/home/gem5/vmlinux-x86-ubuntu` within the disk image.
- Packer's **file provisioner** (`direction=download`) copies this file to the host machine in the `x86-disk-image-24.04/` or `x86-disk-image-22.04/` output directory depending upon the disk image being built.
- The kernel version is printed before extraction in `post-installation.sh`.

This kernel can be used as a resource for **gem5 simulations** and is not restricted to this disk image.

## Changes from the Base Ubuntu Image

- **Default User**:
  - Username: `gem5`
  - Password: `12345`

- **Networking**:
  - **Disabled by default** by renaming `/etc/netplan/00-installer-config.yaml` or `/etc/netplan/50-cloud-init.yaml` to `.bak`.
  - **To re-enable networking**:

    ```sh
    sudo mv /etc/netplan/00-installer-config.yaml.bak /etc/netplan/00-installer-config.yaml
    sudo systemctl unmask systemd-networkd-wait-online.service
    sudo systemctl enable systemd-networkd-wait-online.service
    sudo systemctl start systemd-networkd-wait-online.service  # If enabling immediately
    ```

- **gem5-Bridge (`m5`)**:
  - Installed at `/usr/local/bin/gem5-bridge` with a symlink to `m5` for compatibility.
  - `libm5` is installed in `/usr/local/lib/`, and headers are placed in `/usr/local/include/m5`.

## Extending the Disk Image

- Add more packages by modifying `scripts/install-user-packages.sh`. Install benchmarks onto the base disk image by modifying `scripts/install-user-benchmarks.sh`.
- Transfer additional files using Packer’s file provisioner:

  ```hcl
  provisioner "file" {
      destination = "/home/gem5/"
      source      = "path/to/files"
  }
  ```
  
If you need to increase the size of the image when adding more libraries and files to the image update the size of the partition in the respective `http/*/user-data` file. Also, update the `disk_size` parameter in the packer file to be at least one mega byte more than the size you defined in the `user-data` file.

**NOTE:** You can extend this disk image by modifying the `install-user-benchmarks` and `install-user-packages.sh` script, but it requires building the image from scratch.

To take a pre-built image and add new files or packages, take a look at the following [documentation](https://www.gem5.org/documentation/gem5-stdlib/extending-disk-images).

## Troubleshooting

- **Enable Packer Logs**: This causes Packer to print additional debug messages.

  ```sh
  PACKER_LOG=INFO ./build.sh
  ```

- **Common Packer Bug**:
  - VM build may abort after **2-5 minutes**, even with `ssh_timeout` set.
  - Workaround: **Increase `ssh_handshake_attempts`** (e.g., `1000`).

- **Monitor Installation**:
  - Use a **VNC viewer** to watch installation. The port is displayed in the terminal.
  The output may appear as follows:

    ```bash
    ==> qemu.initialize: Waiting 10s for boot...
    ==> qemu.initialize: Connecting to VM via VNC (127.0.0.1:5995)
    ```

For further details, refer to:
[Ubuntu Autoinstall Documentation](https://ubuntu.com/server/docs/install/autoinstall).
