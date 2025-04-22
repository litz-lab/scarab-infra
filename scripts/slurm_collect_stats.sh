#!/bin/bash
descriptor_path=$1
root_dir=$2
experiment_name=$3

cd scarab_stats
python3 stat_collector.py -d $descriptor_path -o $root_dir/simulations/$experiment_name/collected_stats.csv