# Creating Linux Kernel Binary
This document provides instructions to create a Linux kernel binary. Such kernel binary can be used during a gem5 Full System simulation.
We assume the following initial directory structure before following the instructions in this README file:

```
Linux-kernel/
  |
  |___ linux-configs                           # Folder with Linux kernel configuration files
  |
  |___ README.md                               # This README file
```

## Linux Kernels
We assume the following five LTS (long term support) releases of the Linux kernel:
- 4.4.186
- 4.9.186
- 4.14.134
- 4.19.83
- 5.4.49

To compile the Linux binaries, follow these instructions (assuming that you are in `src/Linux-kernel/` directory):

```sh
# will create a `linux` directory and download the initial kernel files into it.
git clone https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git
cd linux
# replace version with any of the above listed version numbers
git checkout v[version]
# copy the appropriate Linux kernel configuration file from linux-configs/
cp ../linux-configs/config.[version] .config
make -j8
```

After this process succeeds, the compiled Linux binary, named as `vmlinux`, can be found in the `src/Linux-kernel/linux`. The final structure of the `src/Linux-kernel/` directory will look as following:

```
Linux-kernel/
  |
  |___ linux-configs                           # Folder with Linux kernel configuration files
  |
  |___ linux                                   # Linux source and the kernel binary are placed in this folder
  |
  |___ README.md                               # This README file
```

**Note:** The above instructions are tested with `gcc 7.5.0` and the compiled Linux binaries can be downloaded from the following links:

- [vmlinux-4.4.186](http://dist.gem5.org/dist/v20-1/kernels/x86/static/vmlinux-4.4.186)
- [vmlinux-4.9.186](http://dist.gem5.org/dist/v20-1/kernels/x86/static/vmlinux-4.9.186)
- [vmlinux-4.14.134](http://dist.gem5.org/dist/v20-1/kernels/x86/static/vmlinux-4.14.134)
- [vmlinux-4.19.83](http://dist.gem5.org/dist/v20-1/kernels/x86/static/vmlinux-4.19.83)
- [vmlinux-5.4.49](http://dist.gem5.org/dist/v20-1/kernels/x86/static/vmlinux-5.4.49)


**Licensing:**
Linux is released under the GNU General Public License version 2 (GPLv2), but it also contains several files under other compatible licenses. For more information about Linux Kernel Copy Right please refer to [here](https://www.kernel.org/legal.html) and [here](https://www.kernel.org/doc/html/latest/process/license-rules.html#kernel-licensing).
