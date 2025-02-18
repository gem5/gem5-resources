# Copyright (c) 2020 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

#!/bin/bash
echo 'Post Installation Started'

mv /home/gem5/serial-getty@.service /lib/systemd/system/

mv /home/gem5/m5 /sbin
ln -s /sbin/m5 /sbin/gem5
# copy and run outside (host) script after booting
cat /home/gem5/runscript.sh >> /root/.bashrc

sudo chown -R gem5:gem5 /home/gem5/parsec-benchmark/

echo 'Post Installation Done'
