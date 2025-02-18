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
  default = "arm-ubuntu"
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
  boot_command     = ["<wait130>",
                      "gem5<enter><wait>",
                      "12345<enter><wait>",
                      "sudo mv /etc/netplan/50-cloud-init.yaml.bak /etc/netplan/50-cloud-init.yaml<enter><wait>",
                      "12345<enter><wait>",
                      "sudo netplan apply<enter><wait>",
                      "<wait>"]
  cpus             = "4"
  disk_size        = "4600"
  format           = "raw"
  headless         = "true"
  disk_image       = "true"
  iso_checksum     = "sha256:eb94422a3908c6c5183c03666b278b6e8bcfbde04da3d7c3bb5374bc82e0ef48"
  iso_urls         = ["./arm-ubuntu-24.04-20240823"]
  memory           = "8192"
  output_directory = "disk-image-arm-npb"
  qemu_binary      = "/usr/bin/qemu-system-aarch64"
  qemuargs         = [  ["-boot", "order=dc"],
                        ["-bios", "./files/flash0.img"],
                        ["-cpu", "host"],
                        ["-enable-kvm"],
                        ["-machine", "virt"],
                        ["-machine", "gic-version=3"],
                        ["-device","virtio-gpu-pci"],
                        ["-device", "qemu-xhci"],
                        ["-device","usb-kbd"],

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
    source      = "makefiles/ARM/make.def"
    destination = "/home/gem5/NPB3.4-OMP/config/"
  }

  provisioner "shell" {
    execute_command = "echo '${var.ssh_password}' | {{ .Vars }} sudo -E -S bash '{{ .Path }}'"
    scripts         = ["scripts/post-installation.sh"]
  }

}
