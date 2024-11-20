# Make kernel and the gem5 bridge driver

This document will highlight the steps needed to make a kernel and its modules with the gem5-bridge driver.

## Ubuntu 24.04 disk image

Assuming you are in the `src/ubuntu-generic-diskiamges` directory. 
`cd` to the `24.04-dockerfile` directory and build the docker image.

```bash
cd 24.04-dockerfile
docker build -t ubuntu-kernel-build .
cd ..
```

Then lets make a new directory called `my-arm-6.8.12-kernel`.
This directory will have the kernel source and the modules of our built kernel.

```bash
mkdir my-arm-6.8.12-kernel
```

lets get the kernel source from `apt source`, we will be making the `6.8.12 96.8.0-47-generic)` kernel.

```bash
cd my-arm-6.8.12-kernel
apt source linux-image-unsigned-6.8.0-47-generic
```

You might need to update ubuntu source list to use the above command.
Checkout this post if you need to update: <https://askubuntu.com/questions/1512042/ubuntu-24-04-getting-error-you-must-put-some-deb-src-uris-in-your-sources-list>

Lets make an output directory that will have our built modules

```bash
mkdir output
```

Lets run the docker image from the `my-arm-6.8.12-kernel` directory.

```bash
docker run --rm -it -u $UID:$GID -v ./linux-6.8.0:/workspace/source -v ./output:/workspace/output --name kernel-builder ubuntu-kernel-build
```

Now in the docker terminal, lets build the kernel

```bash
cd source
make defconfig
make -j$nproc
```

The above commands will make the kernel and the modules.
lets install the modules to the output directory

```bash
make INSTALL_MOD_PATH=/workspace/output modules_install
```

After the modules are installed, lets install our gem5-bridge driver.

in the `/workspace/source`, lets get the driver files

```bash
git clone https://github.com/nkrim/gem5.git --depth=1 --filter=blob:none --no-checkout --sparse --single-branch --branch=gem5-bridge
cd gem5
git sparse-checkout add util/m5
git sparse-checkout add util/gem5_bridge
git sparse-checkout add include
git checkout
```
Now lets make our driver

```bash
cd util/gem5_bridge
make KMAKEDIR=/workspace/source INSTALL_MOD_PATH=/workspace/output build install
```

The above command specifies the kernel source path and the output path.
You can find the kernel at `src/ubuntu-generic-diskimages/my-arm-6.8.12-kernel/linux-6.8.0/vmlinux`
and the modules at `src/ubuntu-generic-diskimages/my-arm-6.8.12-kernel/output/lib/modules/6.8.12`.

Now lets move the modules to our disk image.

First you will need to delete the `build` file in `src/ubuntu-generic-diskimages/my-arm-6.8.12-kernel/output/lib/modules/6.8.12` as that is a symlink to the `/workspace` directory in source

```bash
rm output/lib/modules/6.8.12/build
```

now add the following file provisioner to move the files from host to the disk

```hcl
  provisioner "file" {
    destination= "/home/gem5"
    source = "my-arm-6.8.12-kernel/output/lib/modules/6.8.12"
  }
```

also add the following lines in the post install script to move the modules to `/lib/modules` and run `depmode` and `initramfs`

```bash
mv /home/gem5/6.8.12 /lib/modules/6.8.12
depmod --quick -a 6.8.12
update-initramfs -u -k 6.8.12
```

Now you can run a gem5 fs simulation with this disk and the kernel we just made to use the new gem 5-bridge driver.
