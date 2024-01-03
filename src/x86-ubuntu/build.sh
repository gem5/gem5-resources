PACKER_VERSION="1.10.0"

if [ ! -f ./packer ]; then
    wget https://releases.hashicorp.com/packer/${PACKER_VERSION}/packer_${PACKER_VERSION}_linux_amd64.zip;
    unzip packer_${PACKER_VERSION}_linux_amd64.zip;
    rm packer_${PACKER_VERSION}_linux_amd64.zip;
fi

# Install the needed plugins
./packer init x86-ubuntu/x86-ubuntu.pkr.hcl 

# Build the image
./packer build x86-ubuntu/x86-ubuntu.pkr.hcl 