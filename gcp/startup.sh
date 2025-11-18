#!/bin/bash
set -e

# Log all output
exec > >(tee /var/log/startup-script.log)
exec 2>&1

echo "Starting instance initialization..."

# Update and upgrade packages
echo "Running apt-get update and upgrade..."
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get upgrade -y

# Install Git
echo "Installing Git..."
apt-get install -y git

# Install Docker
echo "Installing Docker..."
apt-get install -y ca-certificates curl
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Setup Docker to run as non-root user
echo "Configuring Docker for non-root user ${ssh_user}..."
groupadd -f docker
usermod -aG docker ${ssh_user}

# Enable Docker service
systemctl enable docker
systemctl start docker

echo "Instance initialized successfully" > /tmp/startup-complete
echo "Initialization complete at $(date)"