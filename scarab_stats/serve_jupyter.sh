#!/bin/bash
# set -x #echo on

# Activate conda environment first
CONDA_BIN=$(command -v conda)
if [[ -z "$CONDA_BIN" ]]; then
    echo "ERR: conda not found on PATH."
    exit 1
fi

CONDA_BASE=$("$CONDA_BIN" info --base 2>/dev/null)
if [[ -z "$CONDA_BASE" || ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    echo "ERR: Unable to locate conda.sh; ensure Conda is installed correctly."
    exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate scarabinfra

GUIDE_PATH="scarab_stats_quick_start.ipynb"

# Check that guide exists
test -f $GUIDE_PATH
GUIDE_EXISTS=$?

if [[ $GUIDE_EXISTS -ne 0 ]]; then
    echo "ERR: Couldn't find $GUIDE_PATH in the current directory."
    exit 1
fi

# Scan for free ports starting at 8889
BASE_PORT=8889
INCREMENT=1

port=$BASE_PORT
isfree=$(ss -tln | grep ":$port ")
hostname=$(hostname)

while [[ -n "$isfree" ]]; do
    port=$[port+INCREMENT]
    isfree=$(ss -tln | grep ":$port ")
done

echo "Using port: $port"

# Set fixed password
password="scarabdev"

# Launch notebook server quietly as child process on porte
python3 -m notebook --no-browser $GUIDE_PATH --ip=0.0.0.0 --port=$port --NotebookApp.password=$(python3 -c "from jupyter_server.auth import passwd; print(passwd('$password'))") > jupyter_log.txt 2>&1 &
pid=$!

# Create stop program
echo "kill $pid" > stop_jupyter.sh
chmod +x stop_jupyter.sh

# Get username to make ssh tunnel command
me=$(whoami)

echo
echo "Run the following command on your local machine to create a ssh tunnel to the server:"
echo "ssh -NfL localhost:$port:localhost:$port $me@$hostname.soe.ucsc.edu"
echo "(Above not requied if using vscode with Remote - SSH extension)"
echo
echo "Visit the following url in the browser on your local machine to access the notebook:"
echo "http://localhost:$port/"
echo "Password: $password"
echo "Open $GUIDE_PATH for an interactive quick start guide for the scarab stats library"
echo
echo "When you are done run the following on $hostname:"
echo "./stop_jupyter.sh"
echo 
echo "To close the ssh tunnel on your local (unix) machine, use the following to get pid to kill:"
echo "ps -axo pid,user,cmd | grep 'ssh -NfL localhost:$port:localhost:$port'"
echo "(Above not requied if using vscode with Remote - SSH extension)"
