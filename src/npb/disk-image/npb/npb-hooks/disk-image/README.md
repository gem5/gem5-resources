# Steps to create the disk image  
Step 1. Download `packer` at [packer.io](https://www.packer.io/downloads.html).  
Step 2. Build the disk image  
```bash
./packer build ubuntu.json
```
The output will be in the folder `output-ubuntu1804`. The disk image is in RAW format.  
# Customize the disk image
`scripts/post-installation.sh`: the script that runs after Ubuntu Server is installed.
## How to execute a command as a root user?
For example, if the password is `12345`,
```bash
echo 12345 | sudo [command];
```
## How to access the disk image after the building process (e.g. to install packages/to inspect the image manually)?
Adding the following to the post installation scirpt would make it sleeps until a file exists, which means you can make the file to exit. For example, if the file is `/tmp/quit`, then add the following to the post installation script,
```bash
while [ ! -f /tmp/quit ]
do
  sleep 1m
done
```
and to exit,
```bash
touch /tmp/quit
```
