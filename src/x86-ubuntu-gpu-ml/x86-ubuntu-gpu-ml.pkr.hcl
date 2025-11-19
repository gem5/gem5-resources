# Copyright (c) 2024 Advanced Micro Devices, Inc.
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

variable "image_name" {
  type    = string
  default = "x86-ubuntu-gpu-ml"
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
  accelerator      = "kvm"
  boot_command     = ["e<wait>",
                      "<down><down><down>",
                      "<end><bs><bs><bs><bs><wait>",
                      "autoinstall  ds=nocloud-net\\;s=http://{{ .HTTPIP }}:{{ .HTTPPort }}/ ---<wait>",
                      "<f10><wait>"
                    ]
  cpus             = "4"
  disk_size        = "56000"
  format           = "raw"
  headless         = "true"
  http_directory   = "http"
  iso_checksum     = "sha256:d6dab0c3a657988501b4bd76f1297c053df710e06e0c3aece60dead24f270b4d"
  iso_urls         = ["https://releases.ubuntu.com/24.04.2/ubuntu-24.04.2-live-server-amd64.iso"]
  memory           = "8192"
  output_directory = "disk-image"
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
    destination = "/home/gem5/"
    source      = "files/run_gem5_app.sh"
  }

  provisioner "file" {
    destination = "/home/gem5/"
    source      = "files/load_amdgpu.sh"
  }

  provisioner "file" {
    destination = "/home/gem5/"
    source      = "files/serial-getty@.service"
  }

  provisioner "file" {
    destination = "/home/gem5/"
    source      = "files/gem5_wmi/gem5_wmi.c"
  }

  provisioner "file" {
    destination = "/home/gem5/"
    source      = "files/gem5_wmi/Makefile"
  }

  provisioner "shell" {
    execute_command = "echo '${var.ssh_password}' | {{ .Vars }} sudo -E -S bash '{{ .Path }}'"
    scripts         = ["scripts/rocm-install.sh"]
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
    destination = "/usr/lib/firmware/amdgpu/ip_discovery.bin"
    source      = "files/mi300_discovery"
  }

  provisioner "file" {
    source      = "/home/gem5/vmlinux-gpu-ml"
    destination = "vmlinux-gpu-ml"
    direction   = "download"
  }
}
