#!/bin/bash

# Copyright (c) 2022 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

# This file validates the `cloud.txt` and `arm-ubuntu.json` files before
# starting packer.

USER=`whoami`
HOSTNAME=`hostname`

# Check if the  USER is `ubuntu`, otherwise tell the user to configure
# ssh-keygen and cloud.txt file.

if [ ! $USER = "ubuntu" ]; then
    echo "The username is $USER"
    echo "The script expects that the user HAVE ALREADY executed \"ssh-keygen\""

    # Checking for the cloud.txt file

    if [ ! -f "arm-ubuntu/cloud.txt" ]; then
        echo "cloud.txt not found! Please refer to the README.md file on how to create the cloud.txt file."
        exit
    else

        # cloud.txt file exists. Need to check whether it is modified or not.

        count=`grep "name: ubuntu" arm-ubuntu/cloud.txt |wc -l`
        if [ $count -ne 0 ]; then
            echo "cloud.txt 'name' is not modified correctly! Please refer to the README.md file on how to modify name and ssh-rsa in the cloud.txt file."
            exit
        fi

        # Checking whether the ssh-rsa line is modified.

        KEY=`grep "ssh-rsa" arm-ubuntu/cloud.txt`

        if [[ "$KEY" == *"$USER@$HOSTNAME"* ]]; then
            echo "cloud.txt verified!"
        else
            echo "cloud.txt 'ssh-rsa' key is not modified correctly! Please refer to the README.md file on how to modify ssh-rsa in the cloud.txt file."
            exit
        fi
    fi

    echo "All files are modified accordingly."
fi

# Modifying the json script.

sed "s/\/home\/ubuntu/\/home\/${USER}/g" arm-ubuntu/arm-ubuntu.json > arm-ubuntu/.arm-ubuntu.json
mv arm-ubuntu/.arm-ubuntu.json arm-ubuntu/arm-ubuntu.json
sed "s/\"ssh_username\": \"ubuntu\",/\"ssh_username\": \"${USER}\",/g" arm-ubuntu/arm-ubuntu.json > arm-ubuntu/.arm-ubuntu.json
mv arm-ubuntu/.arm-ubuntu.json arm-ubuntu/arm-ubuntu.json

# Modifying the post-installation script.

sed "s/\/home\/ubuntu/\/home\/${USER}/g" arm-ubuntu/post-installation.sh > arm-ubuntu/.post-installation.sh
mv arm-ubuntu/.post-installation.sh arm-ubuntu/post-installation.sh

# Downloading packer

PACKER_VERSION="1.8.0"

if [ ! -f ./packer ]; then
    wget https://releases.hashicorp.com/packer/${PACKER_VERSION}/packer_${PACKER_VERSION}_linux_amd64.zip;
    unzip packer_${PACKER_VERSION}_linux_amd64.zip;
    rm packer_${PACKER_VERSION}_linux_amd64.zip;
fi

# Validating and executing packer

./packer validate arm-ubuntu/arm-ubuntu.json
./packer build arm-ubuntu/arm-ubuntu.json

exit
