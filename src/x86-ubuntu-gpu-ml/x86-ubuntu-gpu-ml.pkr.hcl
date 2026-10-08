# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD 3-Clause

packer {
  required_plugins {
    qemu = {
      source  = "github.com/hashicorp/qemu"
      version = "~> 1"
    }
  }
}

variable "rocm_image_name" {
  type    = string
  default = "x86-ubuntu-rocm101"
}

variable "pytorch_image_name" {
  type    = string
  default = "x86-ubuntu-pytorch-r714"
}

variable "rocm_disk_size" {
  type    = string
  default = "12000"
}

variable "pytorch_disk_size" {
  type    = string
  default = "34000"
}

variable "ssh_password" {
  type    = string
  default = "12345"
}

variable "ssh_username" {
  type    = string
  default = "gem5"
}

variable "qemu_path" {
  type    = string
  default = "/usr/bin/qemu-system-x86_64"
}

locals {
  iso_checksum = "sha256:dec49008a71f6098d0bcfc822021f4d042d5f2db279e4d75bdd981304f1ca5d9"
  iso_url      = "https://releases.ubuntu.com/26.04/ubuntu-26.04-live-server-amd64.iso"
  boot_command = ["e<wait>",
                  "<down><down><down>",
                  "<end><bs><bs><bs><bs><wait>",
                  "autoinstall  ds=nocloud-net\\;s=http://{{ .HTTPIP }}:{{ .HTTPPort }}/ ---<wait>",
                  "<f10><wait>"
                ]
}

source "qemu" "rocm" {
  accelerator           = "kvm"
  boot_command          = local.boot_command
  cpus                  = "4"
  disk_size             = var.rocm_disk_size
  format                = "raw"
  headless              = "true"
  http_directory        = "http"
  iso_checksum          = local.iso_checksum
  iso_urls              = [local.iso_url]
  memory                = "8192"
  output_directory      = "disk-image-rocm-tmp"
  qemu_binary           = var.qemu_path
  qemuargs              = [["-cpu", "host"], ["-display", "none"]]
  shutdown_command      = "echo '${var.ssh_password}'|sudo -S shutdown -P now"
  ssh_password          = var.ssh_password
  ssh_username          = var.ssh_username
  ssh_wait_timeout      = "60m"
  vm_name               = var.rocm_image_name
  ssh_handshake_attempts = "1000"
}

source "qemu" "pytorch" {
  accelerator           = "kvm"
  boot_command          = local.boot_command
  cpus                  = "4"
  disk_size             = var.pytorch_disk_size
  format                = "raw"
  headless              = "true"
  http_directory        = "http"
  iso_checksum          = local.iso_checksum
  iso_urls              = [local.iso_url]
  memory                = "8192"
  output_directory      = "disk-image-pytorch-tmp"
  qemu_binary           = var.qemu_path
  qemuargs              = [["-cpu", "host"], ["-display", "none"]]
  shutdown_command      = "echo '${var.ssh_password}'|sudo -S shutdown -P now"
  ssh_password          = var.ssh_password
  ssh_username          = var.ssh_username
  ssh_wait_timeout      = "60m"
  vm_name               = var.pytorch_image_name
  ssh_handshake_attempts = "1000"
}

build {
  sources = ["source.qemu.rocm"]

  provisioner "file" {
    destination = "/home/gem5/"
    source      = "files/gem5_init.sh"
  }

  provisioner "file" {
    destination = "/home/gem5/"
    source      = "files/run_gem5_app.sh"
  }

  provisioner "file" {
    destination = "/home/gem5/"
    source      = "files/serial-getty@.service"
  }

  provisioner "shell" {
    execute_command = "echo '${var.ssh_password}' | {{ .Vars }} sudo -S bash '{{ .Path }}'"
    scripts         = ["scripts/base-install.sh", "scripts/rocm-install.sh"]
  }

  provisioner "file" {
    destination = "/root/roms/"
    source      = "files/mi200.rom"
  }

  provisioner "file" {
    destination = "/root/roms/"
    source      = "files/mi300.rom"
  }

  provisioner "file" {
    destination = "/usr/lib/firmware/amdgpu/mi300_discovery"
    source      = "files/mi300_discovery"
  }

  provisioner "file" {
    destination = "/usr/lib/firmware/amdgpu/mi350_discovery"
    source      = "files/mi350_discovery"
  }

  provisioner "file" {
    source      = "/home/gem5/vmlinux-gpu-ml"
    destination = "vmlinux-rocm101"
    direction   = "download"
  }

  post-processor "shell-local" {
    inline = [
      "mkdir -p disk-image",
      "mv disk-image-rocm-tmp/${var.rocm_image_name} disk-image/",
      "rmdir disk-image-rocm-tmp"
    ]
  }
}

build {
  sources = ["source.qemu.pytorch"]

  provisioner "file" {
    destination = "/home/gem5/"
    source      = "files/gem5_init.sh"
  }

  provisioner "file" {
    destination = "/home/gem5/"
    source      = "files/run_gem5_app.sh"
  }

  provisioner "file" {
    destination = "/home/gem5/"
    source      = "files/serial-getty@.service"
  }

  provisioner "shell" {
    execute_command = "echo '${var.ssh_password}' | {{ .Vars }} sudo -S bash '{{ .Path }}'"
    scripts         = ["scripts/base-install.sh", "scripts/pytorch-install.sh"]
  }

  provisioner "file" {
    destination = "/root/roms/"
    source      = "files/mi200.rom"
  }

  provisioner "file" {
    destination = "/root/roms/"
    source      = "files/mi300.rom"
  }

  provisioner "file" {
    destination = "/usr/lib/firmware/amdgpu/mi300_discovery"
    source      = "files/mi300_discovery"
  }

  provisioner "file" {
    destination = "/usr/lib/firmware/amdgpu/mi350_discovery"
    source      = "files/mi350_discovery"
  }

  provisioner "file" {
    source      = "/home/gem5/vmlinux-gpu-ml"
    destination = "vmlinux-pytorch-rocm714"
    direction   = "download"
  }

  post-processor "shell-local" {
    inline = [
      "mkdir -p disk-image",
      "mv disk-image-pytorch-tmp/${var.pytorch_image_name} disk-image/",
      "rmdir disk-image-pytorch-tmp"
    ]
  }
}
