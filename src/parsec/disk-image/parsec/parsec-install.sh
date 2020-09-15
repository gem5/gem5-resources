# Copyright (c) 2020 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

cd /home/gem5/
su gem5
echo "12345" | sudo -S apt update

# Allowing services to restart while updating some
# libraries.
sudo apt install -y debconf-utils
sudo debconf-get-selections | grep restart-without-asking > libs.txt
sed -i 's/false/true/g' libs.txt
while read line; do echo $line | sudo debconf-set-selections; done < libs.txt
sudo rm libs.txt
##

# Installing packages needed to build PARSEC
sudo apt install -y build-essential
sudo apt install -y autotools-dev
sudo apt install -y automake
sudo apt install -y m4
sudo apt install -y git
sudo apt install -y python
sudo apt install -y python-dev
sudo apt install -y gettext
sudo apt install -y libx11-dev
sudo apt install -y libxext-dev
sudo apt install -y xorg-dev
sudo apt install -y unzip
sudo apt install -y texinfo
sudo apt install -y freeglut3-dev
sudo apt install -y cmake
##

echo "12345" | sudo -S chown gem5 -R parsec-benchmark/
echo "12345" | sudo -S chgrp gem5 -R parsec-benchmark/
cd parsec-benchmark
./install.sh
cd ..
echo "12345" | sudo -S chown gem5 -R parsec-benchmark/
echo "12345" | sudo -S chgrp gem5 -R parsec-benchmark/
