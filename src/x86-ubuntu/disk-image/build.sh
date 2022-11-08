PACKER_VERSION="1.7.8"

if [ ! -f ./packer ]; then
    wget https://releases.hashicorp.com/packer/${PACKER_VERSION}/packer_${PACKER_VERSION}_linux_amd64.zip;
    unzip packer_${PACKER_VERSION}_linux_amd64.zip;
    rm packer_${PACKER_VERSION}_linux_amd64.zip;
fi

./packer validate x86-ubuntu/x86-ubuntu.json
./packer build x86-ubuntu/x86-ubuntu.json
