# MICRO2025-ATR Workflow

This guide explains how to set up the environment, run simulations, and generate plots using the Scarab infrastructure.

---

## Environment Setup

### Step 1: Install the Slurm
https://github.com/litz-lab/scarab-infra/blob/main/docs/slurm_install_guide.md

### Step 2: Install the Conda Environment
If not already installed, create the Conda environment:
```
conda env create -f environment.yml
```

### Step 3: Activate the Environment
```
conda activate scarabinfra
```

## Running a Simulation

### Step 1: Prepare the Scarab repository
```
git clone -b MICRO2025-ATR git@github.com:litz-lab/scarab.git
```

### Step 2: Prepare the SPEC2017 dataset
```
pip install gdown
gdown --folder --continue --remaining-ok -O ./traces https://drive.google.com/drive/folders/1kU4AyvpnPOfBCQ1ZxSqrp4XXLhyAA5hv
```

### Step 3: Prepare the Simulation Config
Copy the simulation config file (atr.json) into the json/ directory

Update the below values of the json file:
- **`root_dir`**  
  The root directory mounted into the Docker container’s home.
  Update this value to the user’s home directory.
- **`traces_dir`**  
  The local directory containing all simpointed traces.
  Update this value to the path of the traces directory.
- **`scarab_path`**  
  The directory of the Scarab repository that contains the Scarab binary and the PARAM file used to run the experiment. 
  Update this value to the path of the Scarab repository.

### Step 4: Launch the Simulation
```
./run.sh --simulation atr
```

## Plotting and Visualization
### Step 1: Dump Simulation Data
Once the simulation completes, convert the output into CSV format:
```
python3 00_dump_data.py
```

### Step 2: Generate Plots
Use the provided plotting scripts to generate figures:
```
python3 04_lifecycle.py
# Add additional scripts as needed
```
The resulting figures will be saved in the fig/ directory.