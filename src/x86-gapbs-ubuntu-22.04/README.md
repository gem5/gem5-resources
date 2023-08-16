# GAPBS Benchmarks on x86 Ubuntu 22.04

This file documents how to build a X86 Ubuntu 22.04 disk image that contains an installation of the GAPBS benchmarks. It also describes how to use it as a gem5 resource.

## Part 1: Building the Disk Image

The following tutorial was used to build the x86-64 Ubuntu 22.04 disk image, with some changes: 
https://github.com/takekoputa/hpc-disk-image/tree/main


In short, you will need to:
1. Obtain Packer by running `packer-obtain.sh` in the packer_files 
    directory.
2. Obtain the X86 Ubuntu disk image by running `get-disk-img.sh` in the 
    qemu_files directory.
    
3. Generate a new pair of ssh keys and modify `X86_64-hpc.json` and/or  `cloud.txt` to match. 
4. Create `cloud.img` with `make-cloud-img.sh`.
5. Launch a QEMU instance with `qemu-launch.sh`.
6. Run Packer with `./packer build x86_64-hpc.json`.
7. Power off the QEMU instance.

All shell scripts should be run in the same directory that they are located in. That is, you should `cd` into the directory and run them with `./script_name.sh`. They were either provided by the tutorial as a file, or were commands provided in the tutorial that were made into shell scripts for ease of use.

The steps are explained in greater detail below:

(1) Instead of using step 2 of the tutorial to get Packer, change the Packer version at the top of `packer-obtain.sh` to the latest version, then run `packer-obtain.sh`. The script should be run in the packer_files directory.

After obtaining Packer, step 7 of the tutorial, "7. Building the x86_64 Disk Image", was followed. 

(2) Most of the relevant files from the tutorial are already included here, with the exception of the X86 Ubuntu 22.04 disk image. This can be obtained by running `get-disk-img.sh` in the qemu_files folder, which corresponds to step 7.1 of the tutorial. The disk image will be placed in the qemu_files folder, and will be named `x86_64-hpc-2204.img`.

See Part 3 of this README for a diagram of the file structure.

(3) You will also need to either (a)generate a new pair of ssh keys, replace the public key in `cloud.txt`, and change the filepath of the private key in `x86_64-hpc.json`, or (b)generate a new private key from the provided public key and update the filepath in `x86-64-hpc.json`. 
    - The private key filepath in `x86-64-hpc.json` is under "ssh_certificate_file", in "builders"
    - The public key in `cloud.txt` is under ssh-authorized-keys, at the bottom of the file.

(4, 5) After this, make cloud.img by running `make-cloud-img.sh`and launch the disk image with `qemu-launch.sh`.

(6) After launching the qemu instance, run Packer (in a different terminal) using `./packer build x86_64-hpc.json` in the packer_files directory. 

Please note that the `x86_64-hpc.json` included in packer_files/x86_64-hpc.json is a modified version of the one in the linked tutorial. All scripts except for the first two were removed, as they weren't necessary for the GAPBS benchmarks, and `gapbs-install.sh` and `post-installation.sh` were added.

If Packer hangs on "Waiting for ssh", follow the troubleshooting steps at the bottom of the linked tutorial. If a "Permission denied (publickey)" error occurs when attempting to ssh, try the following:
```
eval `ssh-agent -s`
ssh-add ~/.ssh/id_rsa    # or path to file with private key
```

If you want to see debug messages when running Packer, use
`PACKER_LOG=1 ./packer build x86_64-hpc.json`

 The scripts gapbs-install.sh and post-installation.sh are run automatically when Packer builds the disk image. However, if Packer encounters an issue, they can be run manually by copy-and-pasting the file contents into the QEMU instance's terminal.

(7) After Packer finishes, exit the QEMU instance by running qemu-logout.sh, which connects to the disk image with ssh and shuts it down remotely. 

## Part 2: Using the Disk Image as a gem5 Resource

To use this resource, set up a Python script to run a simulation. In this script, in `board.set_kernel_disk_workload(...)`, use 

```
disk_image=DiskImageResource(
    local_path="path_to_disk_image"
    root_partition="1"  # This line is necessary to mount the disk image
)
```

If using Linux kernel 5.15.36, add the following to the script.
This change is necessary as of gem5 v.23:

```
class X86Board_sda(X86Board):
    @overrides(X86Board)
    def get_disk_device(self):
        return "/dev/sda"

board = X86Board_sda(
    ...
    #board setup here
)
```
Otherwise, the kernel will be unable to mount the disk image.

An example script, x86-gapbs-example-2204.py, is included in this file structure. It provides options to choose the size of the benchmark (1-16 and USA-roads) and the type of benchmark to run on the graph. The script also requires the user to specify if the graph is real or synthetic.

The arguments and their options are presented below:
```
--benchmark {cc,bc,tc,pr,bfs}

--synthetic {0,1} 

--size {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,USA-road-d.NY.gr}
```

## Part 3: File Structure Diagram


```
x86-gapbs-ubuntu-22.04/
  |___ packer_files/
  |      |___ 1.packages-install.sh
  |      |___ 2.m5-install.sh
  |      |___ gapbs-install.sh
  |      |___ gem5-init.sh
  |      |___ packer-obtain.sh
  |      |___ post-installation.sh
  |      |___ runscript.sh
  |      |___ serial-getty@.service
  |      |___ x86_64-hpc.json
  |             
  |            
  |
  |___ qemu_files/
  |      |___ cloud.img
  |      |___ cloud.txt
  |      |___ flash0.img
  |      |___ flash1.img
  |      |___ get-disk-img.sh
  |      |___ make-cloud-img.sh
  |      |___ qemu-launch.sh
  |      |___ qemu-logout.sh
  |   
  |
  |___ x86-gapbs-benchmarks-2204.py            # Sample gem5 script
  |
  |___ README.md

```