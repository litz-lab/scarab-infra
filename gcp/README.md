# GCP Infrastructure Setup

This guide explains how to set up and authenticate Terraform with Google Cloud Platform (GCP) for deploying Scarab infrastructure.

## ⚠️ IMPORTANT: Cost Warning

**This infrastructure will incur charges on your Google Cloud Platform account.**

- **Cloud VMs cost money** - The compute instance will continue to accrue charges while it's running, even if you're not actively using it.
- **Pause the VM when not in active use** - To minimize costs, stop the VM instance when you're not actively working:
  ```bash
  gcloud compute instances stop scarab-instance --zone=us-central1-a
  ```
  To start it again:
  ```bash
  gcloud compute instances start scarab-instance --zone=us-central1-a
  ```
- **Clean up after your project is complete** - Always run `terraform destroy` when you're done to avoid ongoing charges:
  ```bash
  terraform destroy
  ```
  This will delete all resources and stop all charges.

**Disclaimer:** I am not responsible for any additional costs incurred if you forget to pause or destroy the infrastructure. Please monitor your GCP billing dashboard regularly and ensure you clean up resources when not in use.

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
   project_id      = "your-gcp-project-id"
   region          = "us-central1"
   zone            = "us-central1-a"
   ```

3. **Prepare cookies.txt for Google Drive downloads (if needed):**

   Google limits bulk downloads from shared folders; providing browser cookies lets gdown reuse your session when Drive throttles direct access.

   If you can still open the files in your browser, exporting cookies.txt may unblock automated downloads.

   For users who need to download traces from the shared Google Drive, prepare cookies.txt before continuing:

   1. On your local machine, open a gdrive folder in a logged-in browser session.
   2. Install a cookies export extension (e.g. 'Get cookies.txt LOCALLY').
   3. Use the extension's 'Export All Cookies' button to save the folder cookies as cookies.txt.
   4. Place cookies.txt in the gcp folder.

4. **Initialize Terraform:**

   ```bash
   terraform init
   ```

5. **Validate your configuration:**

   ```bash
   terraform plan
   ```

6. **Apply the infrastructure:**

   ```bash
   terraform apply
   ```

   **Note:** The startup commands will run automatically during `terraform apply` and you'll see their output in real-time. The commands install Docker, clone repositories, and set up the basic environment. This may take several minutes to complete.

7. **After terraform completes, SSH into the instance and complete the setup:**

   After `terraform apply` completes, you'll see SSH connection details in the output. Connect to the instance:

   ```bash
   ssh -i .ssh/gcp-key ubuntu@<INSTANCE_IP>
   ```

   Or use the command from terraform output:
   ```bash
   terraform output -raw ssh_command
   ```

## Post-Deployment Setup

Once you've SSH'd into the instance, follow these steps to set up and run simulations:

### Step 1: Initialize scarab-infra

Navigate to the scarab-infra directory and run the initialization:

```bash
cd ~/scarab-infra
./sci --init
```

**Important:** When running `./sci --init`, make these choices:
- **Download traces:** Choose "Yes" only for the `cse220` suite, say "No" to all other suites (datacenter, google, etc.)
- **Install Slurm:** Choose "No"
- **Login to ghcr.io:** Choose "No"
- **Setup SSH keys:** Choose "No" (unless you need GitHub access)

The initialization will install Miniconda, create the conda environment, and download CSE220 traces.

### Step 2: Verify the cse220 descriptor

The `cse220.json` descriptor is already present in the repository. You can review it:

```bash
cat ~/scarab-infra/json/cse220.json
```

Make any necessary edits to match your configuration:

```bash
nano ~/scarab-infra/json/cse220.json
```

Key fields to verify:
- `root_dir`: Directory where simulations will be stored
- `scarab_path`: Path to the Scarab repository (should be `~/scarab`)
- `scarab_build`: Build mode (`opt` for optimized, `dbg` for debug)

### Step 3: Build Scarab

Build Scarab for the cse220 descriptor:

```bash
cd ~/scarab-infra
./sci --build-scarab cse220
```

This will build Scarab inside the appropriate Docker container according to the `scarab_build` mode specified in the descriptor. The build process may take several minutes.

### Step 4: Run simulations

Launch the simulations:

```bash
./sci --sim cse220
```

This will run simulations in parallel across all simpoints defined in `cse220.json`. You can monitor progress:

```bash
# Check status of running simulations
./sci --status cse220

# View logs for a specific simulation
ls -la ~/simulations/cse220/baseline/<workload>/<simpoint>/
```

### Step 5: Visualize results

After simulations complete, generate visualizations:

```bash
./sci --visualize cse220
```

This generates bar charts (value and speedup) for each counter listed in the descriptor's `visualize_counters` field. Charts are saved next to `collected_stats.csv` under `<root_dir>/simulations/cse220/`.

### Additional Commands

- **Kill active simulations:**
  ```bash
  ./sci --kill cse220
  ```

- **Clean up containers and temporary state:**
  ```bash
  ./sci --clean cse220
  ```

- **View collected statistics:**
  ```bash
  cat ~/simulations/cse220/collected_stats.csv
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
