# SPEC 2017 Benchmark Suite on ARM64 Ubuntu 22.04 Disk Image

This file first documents how to build an ARM64 disk image with an installation of the SPEC 2017 benchmark suite, then describes how to use it as a gem5 resource.

## Part 1: Making the disk image

The following tutorial was followed, with some changes:

https://github.com/takekoputa/hpc-disk-image/tree/main


Note: All shell scripts should be run in the same directory that they are located in. That is, you should `cd` into the directory and run them with `./script_name.sh`. They were either provided by the tutorial as a file, or were commands provided in the tutorial that were made into shell scripts for ease of use.

### Summary
In short, you will need to:

0. Place your copy of the SPEC 2017 disk image in the `packer-files` directory, or alter the filepath in `arm64-hpc.json` so it points to your copy of the disk image.

1. Obtain Packer by running `packer-obtain.sh` in the `packer-files` directory.

2. Obtain the ARM Ubuntu disk image by running `get-disk-img.sh` in the `qemu-files` directory.

3. Generate a new pair of ssh keys and modify `arm64-hpc.json` and `cloud.txt` to match.

4. Create `cloud.img` with `make-cloud-img.sh`.

5. Launch a QEMU instance with `qemu-launch.sh`.

6. Run Packer with `./packer build arm64-hpc.json`.

7. Power off the QEMU instance with `qemu-logout.sh`, then launch the QEMU instance again. You will now be able to interact with the disk image via the terminal as the root user.

8. Run `install-spec2017.sh` in the QEMU instance (as the root user).

9. Poweroff the QEMU instance.

### Steps
(0) You will need to obtain your own copy of the SPEC 2017 disk image. To use it with this tutorial, you can copy it into the `packer-files` directory. Alternatively, you can change the filepath to the disk image in `arm64-hpc.json`. The filepath can be found under "provisioners", then "source" of the third entry in the "provisioners" section.

(1) Change the Packer version at the top of `packer-obtain.sh` to the latest version, then run `packer-obtain.sh`.

After obtaining Packer, step 5 of the linked tutorial, "5. Building the arm64 Disk Image", was followed. 

(2) Most of the relevant files from the linked tutorial are already included here, with the exception of the ARM64 Ubuntu 22.04 disk image. This can be obtained by running `get-disk-img.sh` in the `qemu_files` directory, which corresponds to step 5.1 of the tutorial. The disk image will be placed in the `qemu_files` directory, and will be named `arm64-hpc-2204.img`.

(3) You will also need to generate a new pair of ssh keys, copy the public key into `cloud.txt`, and change the filepath to the private key in `arm64-hpc.json`. 
- The private key filepath in `arm64-hpc.json` is under "ssh_certificate_file", in "builders"
- The public key in `cloud.txt` is under ssh-authorized-keys, at the bottom of the file.

(4, 5) After this, make cloud.img by running `make-cloud-img.sh` and launch the disk image with `qemu-launch.sh`.

(6) After launching the QEMU instance, run Packer (in a different terminal) using `./packer build arm64-hpc.json` in the packer_files directory. 

If you want to see debug messages when running Packer, use
`PACKER_LOG=1 ./packer build arm64-hpc.json`.

If Packer hangs on "Waiting for ssh", follow the troubleshooting steps at the bottom of the linked tutorial. If a "Permission denied (publickey)" error occurs when attempting to ssh, try the following:
```
eval `ssh-agent -s`
ssh-add ~/.ssh/id_rsa    # or path to file with private key
```

(7) After Packer finishes, exit the QEMU instance by running `qemu-logout.sh` from another terminal window, which connects to the disk image with ssh and shuts it down remotely. Re-launch the QEMU instance with `qemu-launch.sh`. Unlike before, where you were prompted for an Ubuntu login and password, you will now be automatically logged in as the root user.

(8, 9, 10) If you try to run `install-spec2017.sh` as part of the Packer build, Packer won't be able to mount the SPEC 2017 disk image. However, you will be able to run `install-spec2017.sh` when logged in as the root user. Packer will have placed `install-spec2017.sh` into the directory `/home/ubuntu` of the QEMU instance. Run this script, then log out of the QEMU instance by running `qemu-logout.sh` in another terminal window, or by running the `poweroff` command in the QEMU instance.


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
    "cd /home/ubuntu/gem5/spec2017;" +
    "source shrc;" +
    f"runcpu --size {args.size} --iterations 1 --config myconfig.arm.cfg --define gcc_dir='/usr' --noreportable --nobuild {args.benchmark};")
```

These are the commands run by gem5 after Ubuntu finishes booting. If desired, you can add or remove flags and arguments from the last line.


### Part 3: Successfully built benchmarks
Some benchmarks encounter build errors, so not all of them are available. The  benchmarks that I observed to be successfully built are as follows:

```
"500.perlbench_r", 
"631.deepsjeng_s",
"638.imagick_s",  
"505.mcf_r",    
"641.leela_s",            
"507.cactuBSSN_r",  
"644.nab_s",             
"508.namd_r",      
"648.exchange2_s",     
"510.parest_r",     
"649.fotonik3d_s",     
"511.povray_r",   
"654.roms_s",     
"519.lbm_r",   
"657.xz_s",  
"520.omnetpp_r",  
"996.specrand_fs",   
"997.specrand_fr",  
"523.xalancbmk_r",
"998.specrand_is", 
"999.specrand_ir", 
"526.blender_r", 
"531.deepsjeng_r", 
"538.imagick_r" 
"541.leela_r", 
"544.nab_r", 
"548.exchange2_r",
"549.fotonik3d_r",
"554.roms_r", 
"557.xz_r", 
"600.perlbench_s",  
"602.gcc_s",  
"605.mcf_s",   
"603.bwaves_s",
"607.cactuBSSN_s"  
"619.lbm_s", 
"620.omnetpp_s",  
"623.xalancbmk_s", 
```

Some benchmark suites are missing benchmarks. These will still run, skipping the benchmarks that could not be successfully built. 

The benchmark suites are:

```
"fpspeed_mixed_cpp.bset",
"fpspeed_mixed_c.bset", 
"fpspeed_mixed_fortran.bset",
"fpspeed_pure_c.bset",
"fpspeed_pure_fortran.bset", 
"intopenmp.bset", 
"intrate.bset",
"intrate_any_c.bset", 
"intrate_any_cpp.bset", 
"intrate_any_fortran.bset",
"intrate_pure_c.bset",  
"intrate_pure_cpp.bset",    
"intrate_pure_fortran.bset", 
"CPU.bset",      
"intspeed.bset",  
"any_c.bset",      
"intspeed_any_c.bset",   
"any_cpp.bset", 
"intspeed_any_cpp.bset",   
"any_fortran.bset",  
"intspeed_any_fortran.bset",
"fpopenmp.bset",  
"intspeed_pure_c.bset",   
"fprate.bset",  
"intspeed_pure_cpp.bset",   
"fprate_any_c.bset",
"intspeed_pure_fortran.bset", 
"fprate_any_cpp.bset", 
"mixed.bset", 
"fprate_any_fortran.bset", 
"mixed_c.bset",
"fprate_mixed.bset",     
"mixed_cpp.bset", 
"fprate_mixed_c.bset",       
"mixed_fortran.bset",
"fprate_mixed_cpp.bset",      
"openmp.bset",
"fprate_mixed_fortran.bset",  
"pure_c.bset",
"fprate_pure_c.bset",         
"pure_cpp.bset",
"fprate_pure_cpp.bset"      
"pure_fortran.bset"
"fprate_pure_fortran.bset",   
"serial.bset",
"fpspeed.bset",               
"serial_speed.bset",
"fpspeed_any_c.bset",         
"specrate.bset",
"fpspeed_any_cpp.bset",       
"specspeed.bset",
"fpspeed_any_fortran.bset",
"fpspeed_mixed.bset"
```
