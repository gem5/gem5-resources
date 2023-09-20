# SPEC 2006 Benchmark Suite on ARM64 Ubuntu 22.04 Disk Image

This file first documents how to build an ARM64 disk image with an installation of the SPEC 2006 benchmark suite, then describes how to use it as a gem5 resource.

## Part 1: Making the disk image

The following tutorial was followed, with some changes:

https://github.com/takekoputa/hpc-disk-image/tree/main


Note: All shell scripts should be run in the same directory that they are located in. That is, you should `cd` into the directory and run them with `./script_name.sh`. They were either provided by the tutorial as a file, or were commands provided in the tutorial that were made into shell scripts for ease of use.

### Summary
In short, you will need to:

0. Place your copy of the SPEC 2006 disk image in the `packer-files` directory, or alter the filepath in `arm64-hpc.json` so it points to your copy of the disk image.

1. Obtain Packer by running `packer-obtain.sh` in the `packer-files` directory.

2. Obtain the ARM Ubuntu disk image by running `get-disk-img.sh` in the `qemu-files` directory.

3. Generate a new pair of ssh keys and modify `arm64-hpc.json` and `cloud.txt` to match.

4. Create `cloud.img` with `make-cloud-img.sh`.

5. Launch a QEMU instance with `qemu-launch.sh`.

6. Run Packer with `./packer build arm64-hpc.json`.

7. Power off the QEMU instance with `qemu-logout.sh`, then launch the QEMU instance again. You will now be able to interact with the disk image via the terminal as the root user.

8a. Run `install-spec2006.sh` line-by-line, up to `PERLFLAGS="-A libs=-lm -A libs=-ldl" ./buildtools`, to mount the SPEC 2006 disk image.

9. Before running the line
`PERLFLAGS="-A libs=-lm -A libs=-ldl" ./buildtools`, 
make edits to several of the files in the mounted SPEC2006, according to this 
document: https://github.com/studyztp/some-notes/blob/main/SPEC2006_RISCV_ubuntu20.04.4_issue_note.md . For your convenience, you can run the script `replace-configs.sh`, located in `/home/ubuntu` of the QEMU instance, to make the first change.

8b. Continue running `install-spec2006.sh` line-by-line, picking up from where you left off. 

10. Poweroff the QEMU instance.



### Steps
(0) You will need to obtain your own copy of the SPEC 2006 disk image. To use it with this tutorial, you can copy it into the `packer-files` directory. Alternatively, you can change the filepath to the disk image in `arm64-hpc.json`. The filepath can be found under "provisioners", then "source" of the third entry in the "provisioners" section.

(1) Change the Packer version at the top of `packer-obtain.sh` to the latest version, then run `packer-obtain.sh`.

After obtaining Packer, step 5 of the linked tutorial, "5. Building the arm64 Disk Image", was followed. 

(2) Most of the relevant files from the linked tutorial are already included here, with the exception of the ARM64 Ubuntu 22.04 disk image. This can be obtained by running `get-disk-img.sh` in the `qemu_files` directory, which corresponds to step 5.1 of the tutorial. The disk image will be placed in the `qemu_files` directory, and will be named `arm64-hpc-2204.img`.

(3) You will also need to generate a new pair of ssh keys, copy the public key into `cloud.txt`, and change the filepath of the private key in `arm64-hpc.json`. 
- The private key filepath in `arm64-hpc.json` is under "ssh_certificate_file", in "builders"
- The public key in `cloud.txt` is under `ssh-authorized-keys`, at the bottom of the file.

(4, 5) After this, make cloud.img by running `make-cloud-img.sh` and launch the disk image with `qemu-launch.sh`.

(6) After launching the QEMU instance, run Packer (in a different terminal) using `./packer build arm64-hpc.json` in the packer_files directory. 

If you want to see debug messages when running Packer, use
`PACKER_LOG=1 ./packer build arm64-hpc.json`

If Packer hangs on "Waiting for ssh", follow the troubleshooting steps at the bottom of the linked tutorial. If a "Permission denied (publickey)" error occurs when attempting to ssh, try the following:
```
eval `ssh-agent -s`
ssh-add ~/.ssh/id_rsa    # or path to file with private key
```

(7) Packer is unable to run `install-spec2006.sh` when building. There are several potential points of failure, but the one you will most likely encounter is that Packer will fail to mount the SPEC 2006 disk image. After Packer finishes building, you will be able to log into the QEMU instance as the root user.
Run `qemu-logout.sh`, then re-launch the QEMU instance wih `qemu-launch.sh`.


(8a) By this point, you should have a disk image that boots as the root user on an Ubuntu 22.04 operating system and has the SPEC 2006 disk image in `/home/ubuntu`. Paste the commands in `install-spec2006.sh`, up to but NOT including `PERLFLAGS="-A libs=-lm -A libs=-ldl" ./buildtools` into the QEMU instance's terminal. These commands mount the disk image.

(9)The following webpage contains various fixes for problems that were encountered during the tools installation:

https://github.com/studyztp/some-notes/blob/main/SPEC2006_RISCV_ubuntu20.04.4_issue_note.md

After making the changes listed in the above document, you should be able to finish the tools installation. The script `replace-configs.sh` will do the first change in the document, which is about updating config.guess and config.sub.

If you unmount and remount the disk, the changes will be reset. As such, it is much more efficient to run each command in `install-spec2006.sh` individually, rather than running the entire script. 

(8b) Continue running the rest of `install-spec2006.sh`.

After the tools installation, the benchmark installation should proceed relatively smoothly. You may need to run `source shrc` or `filepath_to_shrc/shrc` in the spec2006 directory before attempting the benchmark installation. Otherwise, you may receive the error "runspec: command not found".

After `install-spec2006.sh`, run `poweroff` in the QEMU instance to shut down the disk image.

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

For `readfile_contents`, you can use the following for `command`:

```
command = (
    "cd /home/ubuntu/gem5/spec2006;" +
    "source shrc;"+
    f"runspec --config=myconfig.cfg --size={args.size} --noreportable --action run {args.benchmark};")
```

These are the commands run by gem5 after Ubuntu finishes booting. If desired, you can add or remove flags and arguments from the last line.



### Part 3: Successfully built benchmarks
Not all of the benchmarks can be built successfully, so only some of them are available. The benchmarks that I observed could run are the following:

444.namd, 465.tonto, 401.bzip2, 445.gobmk, 470.lbm, 403.gcc, 471.omnetpp, 410.bwaves, 473.astar, 453.povray, 429.mcf, 454.calculix, 433.milc, 456.hmmer, 434.zeusmp, 458.sjeng, 998.specrand, 435.gromacs, 459.GemsFDTD, 999.specrand, 436.cactusADM, 462.libquantum, 437.leslie3d, 464.h264ref