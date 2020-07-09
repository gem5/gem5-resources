# Copyright (c) 2020 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

# install build-essential (gcc and g++ included) and gfortran
echo "12345" | sudo DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential gfortran

# mount the SPEC2017 ISO file and install SPEC to the disk image
mkdir /home/gem5/mnt
mount -o loop -t iso9660 /home/gem5/cpu2017-1.1.0.iso /home/gem5/mnt
mkdir /home/gem5/spec2017
echo "y" | /home/gem5/mnt/install.sh -d /home/gem5/spec2017 -u linux-x86_64
cd /home/gem5/spec2017
. /home/gem5/mnt/shrc
umount /home/gem5/mnt
rm -f /home/gem5/cpu2017-1.1.0.iso

# use the example config as the template
cp /home/gem5/spec2017/config/Example-gcc-linux-x86.cfg /home/gem5/spec2017/config/myconfig.x86.cfg

# use sed command to replace the default gcc_dir
sed -i "s/\/opt\/rh\/devtoolset-7\/root\/usr/\/usr/g" /home/gem5/spec2017/config/myconfig.x86.cfg

# use sed command to remove the march=native flag when compiling
# this is necessary as the packer script runs in kvm mode, so the details of the CPU will be that of the host CPU
# the -march=native flag is removed to avoid compiling instructions that gem5 does not support
# finetuning flags should be manually added
sed -i "s/-march=native//g" /home/gem5/spec2017/config/myconfig.x86.cfg

# prevent runcpu from calling sysinfo
# https://www.spec.org/cpu2017/Docs/config.html#sysinfo-program
# this is necessary as the sysinfo program queries the details of the system's CPU
# the query causes gem5 runtime error
sed -i "s/command_add_redirect = 1/sysinfo_program =\ncommand_add_redirect = 1/g" /home/gem5/spec2017/config/myconfig.x86.cfg

# build all SPEC workloads
runcpu --config=myconfig.x86.cfg --define build_ncpus=$(nproc) --action=build all

# the above building process will produce a large log file
# this command removes the log files to avoid copying out large files unnecessarily
rm -f /home/gem5/spec2017/result/*
