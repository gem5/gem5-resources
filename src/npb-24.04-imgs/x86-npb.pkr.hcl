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
  default = "x86-ubuntu-npb"
}

variable "ssh_password" {
  type    = string
  default = "12345"
}

variable "ssh_username" {
  type    = string
  default = "gem5"
}

variable "ubuntu_version" {
  type    = string
  default = "24.04"
  validation {
    condition     = contains(["24.04", "26.04"], var.ubuntu_version)
    error_message = "Ubuntu version must be either 24.04 or 26.04."
  }
}

locals {
  disk_image_data = {
    "24.04" = {
      disk_image_file     = "./x86-ubuntu-24.04-20250515"
      iso_checksum        = "sha256:58dfdabfd2510657776ad946c8c4e9cceacbd0806414690598a4f4808c2349b6"
      output_dir          = "disk-image-x86-npb-24-04"
      netplan_restore_cmd = "sudo mv /etc/netplan/50-cloud-init.yaml.bak /etc/netplan/50-cloud-init.yaml"
    }
    "26.04" = {
      # TODO: Fill in the actual base disk image file name, checksum, and
      # netplan restore command for Ubuntu 26.04 once it is available.
      disk_image_file     = "x86-ubuntu"
      iso_checksum        = "sha256:"
      output_dir          = "disk-image-x86-npb-26-04"
      netplan_restore_cmd = "sudo mv /etc/netplan/00-installer-config.yaml.bak /etc/netplan/00-installer-config.yaml"
    }
  }
}

source "qemu" "initialize" {
  accelerator      = "kvm"
  boot_command     = ["<wait120>",
                      "gem5<enter><wait>",
                      "12345<enter><wait>",
                      "${local.disk_image_data[var.ubuntu_version].netplan_restore_cmd}<enter><wait>",
                      "12345<enter><wait>",
                      "sudo netplan apply<enter><wait>",
                      "<wait>"]
  cpus             = "4"
  disk_size        = "5000"
  format           = "raw"
  headless         = "true"
  disk_image       = "true"
  iso_checksum     = local.disk_image_data[var.ubuntu_version].iso_checksum
  iso_urls         = [local.disk_image_data[var.ubuntu_version].disk_image_file]
  memory           = "8192"
  output_directory = local.disk_image_data[var.ubuntu_version].output_dir
  qemu_binary      = "/usr/bin/qemu-system-x86_64"
  qemuargs         = [["-cpu", "host"], ["-display", "none"]]
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
    source      = "makefiles/x86/make.def"
    destination = "/home/gem5/NPB3.4-OMP/config/"
  }

  provisioner "file" {
    source      = "npb-hook-files/addr-version/hooks.c"
    destination = "/home/gem5/NPB3.4-OMP/common/"
  }

  provisioner "shell" {
    execute_command = "echo '${var.ssh_password}' | {{ .Vars }} sudo --preserve-env=UBUNTU_VERSION -S bash '{{ .Path }}'"
    scripts         = ["scripts/post-installation.sh"]
    environment_vars = ["UBUNTU_VERSION=${var.ubuntu_version}"]
  }

}
