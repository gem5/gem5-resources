packer {
  required_plugins {
    qemu = {
      source  = "github.com/hashicorp/qemu"
      version = "~> 1"
    }
  }
}

variable "image_name" {
  type    = string
  default = "riscv-ubuntu"
}

variable "ssh_password" {
  type    = string
  default = "12345"
}

variable "ssh_username" {
  type    = string
  default = "gem5"
}

source "qemu" "initialize" {
  cpus             = "4"
  disk_size        = "5000"
  format           = "raw"
  headless         = "true"
  disk_image       = "true"
  boot_command = ["<wait90>",
                  "gem5<enter><wait2>",
                  "12345<enter><wait2>",
                  "sudo mount -o remount,rw /<enter><wait2>", // remounting system as read-write as qemu does not like that we have m5 exits in the boot process so it mounts system as read ony.
                  "12345<enter><wait2>",
                  "sudo mv /etc/netplan/50-cloud-init.yaml.bak /etc/netplan/50-cloud-init.yaml<enter><wait2>",
                  "sudo netplan apply<enter><wait2>",
                  "<wait>"
                ]
  iso_checksum     = "sha256:ddf1ebb56454ef37e88d6de8aefdef7180fc9a2328a3bf8cff02f7a043e7127b"
  iso_urls         = ["/home/harshilp/gem5-resources-worktrees/make-riscv-kernel/src/riscv-fs/riscv-ubuntu-22.04-24.04/disk-image-24.04/riscv-ubuntu"]
  memory           = "8192"
  output_directory = "riscv-disk-image-ubuntu-24-04"
  qemu_binary      = "/usr/bin/qemu-system-riscv64"

  qemuargs       = [  ["-bios", "/usr/lib/riscv64-linux-gnu/opensbi/generic/fw_jump.elf"],
                      ["-machine", "virt"],
                      ["-kernel","/usr/lib/u-boot/qemu-riscv64_smode/uboot.elf"],
                      ["-device", "virtio-vga"],
                      ["-device", "qemu-xhci"],
                      ["-device", "usb-kbd"]
                  ]
  shutdown_command = "echo '${var.ssh_password}'|sudo -S shutdown -P now"
  ssh_password     = "${var.ssh_password}"
  ssh_username     = "${var.ssh_username}"
  ssh_wait_timeout = "60m"
  vm_name          = "${var.image_name}"
  ssh_handshake_attempts = "1000"
}

build {
  sources = ["source.qemu.initialize"]
  provisioner "file" {
    source      = "npb-with-roi/NPB/NPB3.4-OMP"
    destination = "/home/gem5/"
  }
  
  provisioner "file" {
    source      = "makefiles/riscv/make.def"
    destination = "/home/gem5/NPB3.4-OMP/config/"
  }

  provisioner "file" {
    source      = "npb-hook-files/non-addr-version/hooks.c"
    destination = "/home/gem5/NPB3.4-OMP/common/"
  }

  provisioner "shell" {
    execute_command = "echo '${var.ssh_password}' | {{ .Vars }} sudo -E -S bash '{{ .Path }}'"
    scripts         = ["scripts/post-installation.sh"]
  }
}