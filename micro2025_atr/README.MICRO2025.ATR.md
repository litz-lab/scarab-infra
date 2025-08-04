# MICRO2025-ATR Workflow

This guide explains how to set up the environment, run simulations, and generate plots using the Scarab infrastructure.

---

## Environment Setup

### Step 1: Install the Conda Environment
If not already installed, create the Conda environment:
```
conda env create -f environment.yml
```

### Step 2: Activate the Environment
```
conda activate scarabinfra
```

## Running a Simulation

### Step 1: Prepare the Simulation Config
Copy the simulation config file (atr.json) into the json/ directory

### Step 2: Launch the Simulation
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