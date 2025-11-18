variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "machine_type" {
  description = "GCP machine type"
  type        = string
  default     = "n2-standard-4"
}

variable "disk_size_gb" {
  description = "Boot disk size in GB"
  type        = number
  default     = 50
}

variable "image" {
  description = "GCP boot disk image"
  type        = string
  default     = "ubuntu-os-cloud/ubuntu-minimal-2510-amd64"
}

variable "instance_name" {
  description = "Name of the compute instance"
  type        = string
  default     = "scarab-instance"
}

variable "ssh_user" {
  description = "SSH username"
  type        = string
  default     = "ubuntu"
}