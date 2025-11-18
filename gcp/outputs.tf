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

output "check_startup_status" {
  description = "Command to check startup script status"
  value       = "./scripts/check-startup.sh ${google_compute_instance.vm_instance.network_interface[0].access_config[0].nat_ip}"
}

output "view_startup_log" {
  description = "Command to view the full startup log"
  value       = "ssh -i .ssh/gcp-key ${var.ssh_user}@${google_compute_instance.vm_instance.network_interface[0].access_config[0].nat_ip} 'sudo cat /var/log/startup-script.log'"
}

output "follow_startup_log" {
  description = "Command to follow startup log in real-time"
  value       = "ssh -i .ssh/gcp-key ${var.ssh_user}@${google_compute_instance.vm_instance.network_interface[0].access_config[0].nat_ip} 'sudo tail -f /var/log/startup-script.log'"
}

output "watch_serial_output" {
  description = "Command to watch serial port output (shows startup script in real-time)"
  value       = "gcloud compute instances tail-serial-port-output ${google_compute_instance.vm_instance.name} --zone ${var.zone} --project ${var.project_id}"
}

output "get_serial_output" {
  description = "Command to get full serial port output"
  value       = "gcloud compute instances get-serial-port-output ${google_compute_instance.vm_instance.name} --zone ${var.zone} --project ${var.project_id}"
}