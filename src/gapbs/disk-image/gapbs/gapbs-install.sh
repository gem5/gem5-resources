# Copyright (c) 2020 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

cd /home/gem5/
su gem5

sudo apt install -y debconf-utils
sudo debconf-get-selections | grep restart-without-asking > libs.txt
sed -i 's/false/true/g' libs.txt
while read line; do echo $line | sudo debconf-set-selections; done < libs.txt
sudo rm libs.txt

sudo apt install -y git
echo "12345" | sudo apt-get install -y build-essential libboost-all-dev

git clone https://github.com/darchr/gapbs.git
cd gapbs
make HOOKS=1
cd ..
