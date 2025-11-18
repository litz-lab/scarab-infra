# GCP Infrastructure Setup

This guide explains how to set up and authenticate Terraform with Google Cloud Platform (GCP) for deploying Scarab infrastructure.

## Prerequisites

- [Terraform](https://www.terraform.io/downloads) >= 1.0
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
- A GCP project with billing enabled

## Authentication

### Option 1: Service Account Key (Recommended for local development)

1. **Create a service account:**

   ```bash
   gcloud iam service-accounts create terraform-sa \
     --display-name "Terraform Service Account"
   ```

2. **Grant necessary permissions:**

   ```bash
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
     --member="serviceAccount:terraform-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/editor"
   ```

3. **Create and download the key:**

   ```bash
   gcloud iam service-accounts keys create gcp-credentials.json \
     --iam-account=terraform-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com
   ```

4. **Set the environment variable:**

   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/gcp-credentials.json"
   ```

### Option 2: Application Default Credentials (ADC)

For personal development, you can use your own credentials:

```bash
gcloud auth application-default login
```

This is simpler but uses your personal account instead of a service account.

## Setup Steps

1. **Copy the example variables file:**

   ```bash
   cp terraform.tfvars.example terraform.tfvars
   ```

2. **Edit `terraform.tfvars` with your configuration:**

   ```hcl
   project_id = "your-gcp-project-id"
   region     = "us-central1"
   zone       = "us-central1-a"
   ```

3. **Initialize Terraform:**

   ```bash
   terraform init
   ```

4. **Validate your configuration:**

   ```bash
   terraform plan
   ```

5. **Apply the infrastructure:**

   ```bash
   terraform apply
   ```

## Security Notes

- The `.gitignore` file is configured to exclude:
  - `*.tfstate` files (contain sensitive state data)
  - `terraform.tfvars` (contains your project configuration)
  - `gcp-credentials.json` (service account key)
  - `.terraform/` directory

- Never commit credentials or state files to version control
- Use service accounts with minimal required permissions
- Rotate service account keys regularly

## Useful Commands

- **Format Terraform files:** `terraform fmt`
- **Validate configuration:** `terraform validate`
- **Show current state:** `terraform show`
- **Destroy infrastructure:** `terraform destroy`

## Troubleshooting

### Authentication Errors

If you get authentication errors, verify:

1. The `GOOGLE_APPLICATION_CREDENTIALS` environment variable is set
2. The service account has sufficient permissions
3. The credentials file path is correct

### Project ID Issues

Ensure your project ID in `terraform.tfvars` matches your actual GCP project:

```bash
gcloud config get-value project
```
