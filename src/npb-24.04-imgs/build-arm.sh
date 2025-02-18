PACKER_VERSION="1.10.0"

if [ ! -f ./packer ]; then
    wget https://releases.hashicorp.com/packer/${PACKER_VERSION}/packer_${PACKER_VERSION}_linux_arm64.zip;
    unzip packer_${PACKER_VERSION}_linux_arm64.zip;
    rm packer_${PACKER_VERSION}_linux_arm64.zip;
fi

# make the flash0.sh file
mkdir files
cd ./files
dd if=/dev/zero of=flash0.img bs=1M count=64
dd if=/usr/share/qemu-efi-aarch64/QEMU_EFI.fd of=flash0.img conv=notrunc
cd ..

# get the  base image from gem5 resource
if [! -f ./arm-ubuntu-24.04-20240823 ]; then
    wget https://storage.googleapis.com/dist.gem5.org/dist/develop/images/arm/ubuntu-24-04/arm-ubuntu-24.04-20240823.gz
    gunzip arm-ubuntu-24.04-20240823.gz 
fi
# Install the needed plugins
./packer init arm-npb.pkr.hcl

# Build the image
./packer build arm-npb.pkr.hcl
