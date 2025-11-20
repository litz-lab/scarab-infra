terraform {
  required_version = ">= 1.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Generate SSH key pair
resource "tls_private_key" "ssh_key" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

# Save private key locally
resource "local_file" "private_key" {
  content         = tls_private_key.ssh_key.private_key_openssh
  filename        = "${path.module}/.ssh/gcp-key"
  file_permission = "0600"
}

# Save public key locally
resource "local_file" "public_key" {
  content         = tls_private_key.ssh_key.public_key_openssh
  filename        = "${path.module}/.ssh/gcp-key.pub"
  file_permission = "0644"
}

# Create firewall rule for SSH
resource "google_compute_firewall" "allow_ssh" {
  name    = "allow-ssh-${var.instance_name}"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["ssh-enabled"]
}

# Create compute instance
resource "google_compute_instance" "vm_instance" {
  name         = var.instance_name
  machine_type = var.machine_type
  zone         = var.zone

  tags = ["ssh-enabled"]

  boot_disk {
    initialize_params {
      # Ubuntu 25.10 (Oracular Oriole)
      image = var.image
      size  = var.disk_size_gb
      type  = "pd-standard"
    }
  }

  network_interface {
    network = "default"

    access_config {
      # Ephemeral public IP
    }
  }

  metadata = {
    ssh-keys           = "${var.ssh_user}:${tls_private_key.ssh_key.public_key_openssh}"
    serial-port-enable = "TRUE"
  }

  # Copy cookies.txt to instance if it exists
  provisioner "file" {
    source      = "${path.module}/cookies.txt"
    destination = "/home/${var.ssh_user}/cookies.txt"

    connection {
      type        = "ssh"
      user        = var.ssh_user
      private_key = tls_private_key.ssh_key.private_key_openssh
      host        = self.network_interface[0].access_config[0].nat_ip
    }
  }

  # Execute startup commands directly with sudo
  provisioner "remote-exec" {
    inline = [
      "# Setup gdown cookies if available",
      "mkdir -p /home/${var.ssh_user}/.cache/gdown",
      "if [ -f /home/${var.ssh_user}/cookies.txt ]; then sudo mv /home/${var.ssh_user}/cookies.txt /home/${var.ssh_user}/.cache/gdown/; fi",
      "sudo chown -R ${var.ssh_user}:${var.ssh_user} /home/${var.ssh_user}/.cache",

      "# Update and install packages",
      "sudo apt-get update -y",
      "sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y",
      "sudo apt-get install -y git",

      "# Install Docker",
      "sudo apt-get install -y ca-certificates curl",
      "sudo install -m 0755 -d /etc/apt/keyrings",
      "sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc",
      "sudo chmod a+r /etc/apt/keyrings/docker.asc",

      "# Add Docker repository",
      "echo \"deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable\" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null",

      "sudo apt-get update -y",
      "sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin",

      "# Configure Docker for non-root user",
      "sudo groupadd -f docker",
      "sudo usermod -aG docker ${var.ssh_user}",
      "sudo systemctl enable docker",
      "sudo systemctl start docker",

      "# Clone repositories",
      "sudo -u ${var.ssh_user} bash -c 'cd /home/${var.ssh_user} && git clone https://github.com/abhijitramesh/scarab-infra.git'",
      "sudo -u ${var.ssh_user} bash -c 'cd /home/${var.ssh_user}/scarab-infra && git fetch origin cse220_fall_2025:cse220_fall_2025 && git checkout cse220_fall_2025'",
      "sudo -u ${var.ssh_user} bash -c 'cd /home/${var.ssh_user} && git clone https://github.com/litz-lab/scarab.git scarab'",
      "sudo -u ${var.ssh_user} bash -c 'cd /home/${var.ssh_user}/scarab && git checkout 4a03b768fcbe57b9f59e06fc6a29d83d8b7d25c0'",

      "# Set ownership",
      "sudo chown -R ${var.ssh_user}:${var.ssh_user} /home/${var.ssh_user}/scarab-infra /home/${var.ssh_user}/scarab",

      "# Mark completion",
      "echo \"Instance initialized successfully\" | sudo tee /tmp/startup-complete > /dev/null",
      "echo \"Initialization complete at $(date)\" | sudo tee -a /tmp/startup-complete > /dev/null"
    ]

    connection {
      type        = "ssh"
      user        = var.ssh_user
      private_key = tls_private_key.ssh_key.private_key_openssh
      host        = self.network_interface[0].access_config[0].nat_ip
    }
  }
}