# Copyright (c) 2020 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

# install build-essential (gcc and g++ included) and gfortran
echo "12345" | sudo apt-get install build-essential gfortran

# mount the SPEC2006 ISO file and install SPEC to the disk image
mkdir /home/ubuntu/gem5/mnt
mount -o loop -t iso9660 /home/ubuntu/CPU2006v1.0.1.iso /home/ubuntu/gem5/mnt
mkdir /home/ubuntu/gem5/spec2006
cd /home/ubuntu/gem5/spec2006/tools/src 
PERLFLAGS="-A libs=-lm -A libs=-ldl" ./buildtools
cd /home/ubuntu/gem5/spec2006
. /home/ubuntu/gem5/mnt/shrc
umount /home/ubuntu/gem5/mnt
# rm -f /home/ubuntu/CPU2006v1.0.1.iso

# use the gcc42 config as the template
cp /home/ubuntu/gem5/spec2006/config/linux64-amd64-gcc42.cfg /home/ubuntu/gem5/spec2006/config/myconfig.cfg

# Those 'sed' commands replace paths to gcc, g++ and gfortran binary from /usr/local/sles9/gcc42-0325/bin/* to /usr/bin/*
sed -i "s/\/usr\/local\/sles9\/gcc42-0325\/bin\/gcc/\/usr\/bin\/gcc/g" /home/ubuntu/gem5/spec2006/config/myconfig.cfg
sed -i "s/\/usr\/local\/sles9\/gcc42-0325\/bin\/g++/\/usr\/bin\/g++/g" /home/ubuntu/gem5/spec2006/config/myconfig.cfg
sed -i "s/\/usr\/local\/sles9\/gcc42-0325\/bin\/gfortran/\/usr\/bin\/gfortran/g" /home/ubuntu/gem5/spec2006/config/myconfig.cfg

# build all SPEC workloads
runspec --config=myconfig.cfg --action=build all

