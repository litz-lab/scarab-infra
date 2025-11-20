output "instance_name" {
  description = "Name of the compute instance"
  value       = google_compute_instance.vm_instance.name
}

output "instance_public_ip" {
  description = "Public IP address of the instance"
  value       = google_compute_instance.vm_instance.network_interface[0].access_config[0].nat_ip
}

output "instance_private_ip" {
  description = "Private IP address of the instance"
  value       = google_compute_instance.vm_instance.network_interface[0].network_ip
}

output "ssh_command" {
  description = "Command to SSH into the instance"
  value       = "ssh -i .ssh/gcp-key ${var.ssh_user}@${google_compute_instance.vm_instance.network_interface[0].access_config[0].nat_ip}"
}

output "ssh_key_path" {
  description = "Path to the SSH private key"
  value       = local_file.private_key.filename
}

output "view_startup_status" {
  description = "Command to check startup completion status"
  value       = "ssh -i .ssh/gcp-key ${var.ssh_user}@${google_compute_instance.vm_instance.network_interface[0].access_config[0].nat_ip} 'cat /tmp/startup-complete'"
}