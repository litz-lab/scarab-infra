cd /tmp_home/application/cpu2017 && source shrc
# Pick the SPEC bench directory from APPNAME
benchdir=$(ls benchspec/CPU | grep -i "${APPNAME}" | head -1)
[ -n "$benchdir" ] || { echo "No benchdir for $APPNAME"; exit 1; }

# Pick the memtrace run directory
rundir=$(ls "benchspec/CPU/$benchdir/run" | grep -i memtrace | head -1)
[ -n "$rundir" ] || { echo "No memtrace run dir in $benchdir"; exit 1; }

cd "benchspec/CPU/$benchdir/run/$rundir"