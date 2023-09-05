# ARM64 Ubuntu 22.04 Disk Image

This file first documents how to build an ARM64 disk image, then describes how to use it as a gem5 resource.

## Part 1: Making the disk image

The following tutorial was followed, with some changes:

https://github.com/takekoputa/hpc-disk-image/tree/main


Note: All shell scripts should be run in the same directory that they are located in. That is, you should `cd` into the directory and run them with `./script_name.sh`. They were either provided by the tutorial as a file, or were commands provided in the tutorial that were made into shell scripts for ease of use.

### Summary
In short, you will need to:

1. Obtain Packer by running `packer-obtain.sh` in the packer_files directory.

2. Obtain the ARM Ubuntu disk image by running `get-disk-img.sh` in the qemu_files directory.

3. Generate a new pair of ssh keys and modify `arm64-hpc.json` and `cloud.txt` to match.

4. Create `cloud.img` with `make-cloud-img.sh`.

5. Launch a QEMU instance with `qemu-launch.sh`.

6. Run Packer with `./packer build arm64-hpc.json`.

7. Power off the QEMU instance with `qemu-logout.sh`.

### Steps
(1) Instead of using step 2 of the tutorial to get Packer, change the Packer version at the top of `packer-obtain.sh` to the latest version, then run `packer-obtain.sh`.

After obtaining Packer, step 5 of the tutorial, "5. Building the arm64 Disk Image", was followed. 

(2) Most of the relevant files from the tutorial are already included here, with the exception of the ARM64 Ubuntu 22.04 disk image. This can be obtained by running `get-disk-img.sh` in the `qemu_files` directory, which corresponds to step 5.1 of the tutorial. The disk image will be placed in the `qemu_files` directory, and will be named `arm64-hpc-2204.img`.

(3) You will also need to generate a new pair of ssh keys, copy the public key into `cloud.txt`, and change the filepath of the private key in `arm64-hpc.json`. 
- The private key filepath in `arm64-hpc.json` is under "ssh_certificate_file", in "builders"
- The public key in `cloud.txt` is under ssh-authorized-keys, at the bottom of the file.

(4, 5) After this, make cloud.img by running `make-cloud-img.sh`and launch the disk image with `qemu-launch.sh`.

(6) After launching the QEMU instance, run Packer (in a different terminal) using `./packer build arm64-hpc.json` in the packer_files directory. 

If you want to see debug messages when running Packer, use
`PACKER_LOG=1 ./packer build arm64-hpc.json`

Please note that the `arm64-hpc.json` included in `packer_files/arm64-hpc.json` is a modified version of the one in the linked tutorial, in that all scripts except for the first two were removed.

If Packer hangs on "Waiting for ssh", follow the troubleshooting steps at the bottom of the linked tutorial. If a "Permission denied (publickey)" error occurs when attempting to ssh, try the following:
```
eval `ssh-agent -s`
ssh-add ~/.ssh/id_rsa    # or path to file with private key
```

(7) After Packer finishes, exit the QEMU instance by running `qemu-logout.sh`, which connects to the disk image with ssh and shuts it down remotely. 

## Part 2: Using the disk image as a resource
The disk image can be used as a DiskImageResource when setting the board's workload in the Python script:

```
board.set_kernel_disk_workload(
    kernel=obtain_resource("arm64-linux-kernel-5.15.36"),
    disk_image=DiskImageResource
    (
        local_path="local/path/to/arm64-hpc-2204.img",
        root_partition="1" #make sure to have this, otherwise it won't boot
    ),
    bootloader=obtain_resource("arm64-bootloader-foundation"),
    readfile_contents=command

    )
```

