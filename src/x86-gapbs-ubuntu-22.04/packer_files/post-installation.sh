#!/bin/bash

# Copyright (c) 2020 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

echo 'Post Installation Started'

mv /home/ubuntu/serial-getty@.service /lib/systemd/system/

# copy and run outside (host) script after booting
cat /home/ubuntu/runscript.sh >> /root/.bashrc

echo 'Post Installation Done'
