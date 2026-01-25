#!/usr/bin/env python3

import pandas as pd
import sys
import os
from pathlib import Path

INSTRUCTION_THRESHOLD = 15_000_000


def load_opt_p(filepath):
    """Load opt.p file: segment(0-indexed), cluster"""
    return pd.read_csv(filepath, sep=' ', header=None, names=['segment', 'cluster'])


def load_opt_l(filepath):
    """Load opt.l file: cluster(0-indexed), distance from centroid(each line is segment 0, 1 ...)"""
    df = pd.read_csv(filepath, sep=' ', header=None, names=['cluster', 'distance'])
    df['segment'] = df.index
    return df


def load_inscount(filepath):
    """Load inscount file and return as dict {segment_0idx: total_instrs}"""
    df = pd.read_csv(filepath)
    # Convert 1-indexed to 0-indexed
    df['segment_0idx'] = df['segment'] - 1
    return dict(zip(df['segment_0idx'], df['total_instrs_in_seg']))


def find_inscount_file(fingerprint_dir):
    inscount_files = list(Path(fingerprint_dir).glob("*.inscount"))
    if len(inscount_files) == 0:
        raise FileNotFoundError(f"No .inscount file found in {fingerprint_dir}")
    if len(inscount_files) > 1:
        print(f"Warning: Multiple .inscount files found, using {inscount_files[0]}")
    return str(inscount_files[0])


def find_valid_segment(cluster_id, cluster_data, inscount_dict):
    min_ins = float('inf')
    min_seg = None
    min_dist = None
    
    for _, row in cluster_data.iterrows():
        seg_id = int(row['segment'])
        ins_count = inscount_dict.get(seg_id, 0)
        
        if ins_count <= INSTRUCTION_THRESHOLD:
            return seg_id, ins_count, row['distance']
        
        if ins_count < min_ins:
            min_ins = ins_count
            min_seg = seg_id
            min_dist = row['distance']
    
    print(f"WARNING: All segments in cluster {cluster_id} exceed threshold. "
          f"Selecting segment with min ins_count: {min_seg} ({min_ins:,})")
    return min_seg, min_ins, min_dist


def adjust_simpoints(workload_home):
    import shutil
    
    simpoints_dir = os.path.join(workload_home, "simpoints")
    fingerprint_dir = os.path.join(workload_home, "fingerprint")
    
    opt_p_path = os.path.join(simpoints_dir, "opt.p.lpt0.99")
    opt_l_path = os.path.join(simpoints_dir, "opt.l")
    backup_path = os.path.join(simpoints_dir, "opt.p.lpt0.99.original")
    
    print(f"Loading opt.p from {opt_p_path}")
    opt_p = load_opt_p(opt_p_path)
    
    print(f"Loading opt.l from {opt_l_path}")
    opt_l = load_opt_l(opt_l_path)
    
    inscount_path = find_inscount_file(fingerprint_dir)
    print(f"Loading inscount from {inscount_path}")
    inscount_dict = load_inscount(inscount_path)
    
    print(f"Loaded {len(opt_p)} clusters, {len(opt_l)} segments")
    print(f"Instruction threshold: {INSTRUCTION_THRESHOLD:,}")
    
    cluster_segments = {}
    for cluster_id in opt_p['cluster'].unique():
        cluster_data = opt_l[opt_l['cluster'] == cluster_id].copy()
        cluster_data = cluster_data.sort_values('distance')
        cluster_segments[cluster_id] = cluster_data
    
    print(f"\nCluster statistics:")
    for cluster_id in sorted(cluster_segments.keys()):
        cluster_data = cluster_segments[cluster_id]
        total = len(cluster_data)
        
        ins_counts = [inscount_dict.get(int(row['segment']), 0) for _, row in cluster_data.iterrows()]
        valid = sum(1 for ic in ins_counts if ic <= INSTRUCTION_THRESHOLD)
        min_ins = min(ins_counts)
        max_ins = max(ins_counts)
        
        print(f"  Cluster {cluster_id}: {total} segments ({valid} valid, {total - valid} over) | "
            f"ins: {min_ins:,} ~ {max_ins:,}")
    
    results = []
    changes = []
    
    for _, row in opt_p.iterrows():
        original_segment = int(row['segment'])
        cluster_id = int(row['cluster'])
        
        original_ins_count = inscount_dict.get(original_segment, 0)
        
        if original_ins_count <= INSTRUCTION_THRESHOLD:
            orig_dist = opt_l[opt_l['segment'] == original_segment]['distance'].values[0]
            results.append({'segment': original_segment, 'cluster': cluster_id})
            changes.append({
                'cluster': cluster_id,
                'original_segment': original_segment,
                'new_segment': original_segment,
                'original_ins': original_ins_count,
                'new_ins': original_ins_count,
                'original_dist': orig_dist,
                'new_dist': orig_dist,
                'changed': False
            })
        else:
            orig_dist = opt_l[opt_l['segment'] == original_segment]['distance'].values[0]
            cluster_data = cluster_segments[cluster_id]
            new_seg, new_ins, new_dist = find_valid_segment(
                cluster_id, cluster_data, inscount_dict
            )
            
            results.append({'segment': new_seg, 'cluster': cluster_id})
            changes.append({
                'cluster': cluster_id,
                'original_segment': original_segment,
                'new_segment': new_seg,
                'original_ins': original_ins_count,
                'new_ins': new_ins,
                'original_dist': orig_dist,
                'new_dist': new_dist,
                'changed': True
            })
    
    opt_p2 = pd.DataFrame(results)
    
    changes_df = pd.DataFrame(changes)
    num_changed = changes_df['changed'].sum()
    
    print(f"\n{'='*60}")
    print(f"Summary: {num_changed}/{len(changes_df)} clusters adjusted")
    print(f"{'='*60}")
    
    if num_changed > 0:
        changed = changes_df[changes_df['changed']]
        for _, row in changed.iterrows():
            print(f"  Cluster {row['cluster']}: "
                f"segment {row['original_segment']} ({row['original_ins']:,}, dist={row['original_dist']:.4f}) -> "
                f"segment {row['new_segment']} ({row['new_ins']:,}, dist={row['new_dist']:.4f})")
    
    all_valid = True
    for _, row in opt_p2.iterrows():
        seg = int(row['segment'])
        ins = inscount_dict.get(seg, 0)
        if ins > INSTRUCTION_THRESHOLD:
            print(f"WARNING: ALL segments from Cluster {row['cluster']} are over THRESHOLD, "
                  f"Segment {seg} still exceeds threshold ({ins:,} > {INSTRUCTION_THRESHOLD:,})")
            all_valid = False
    
    if all_valid:
        print(f"\nAll segments are under {INSTRUCTION_THRESHOLD:,} instructions")
    
    shutil.copy(opt_p_path, backup_path)
    print(f"\nBacked up original to {backup_path}")
    
    opt_p2.to_csv(opt_p_path, sep=' ', header=False, index=False)
    print(f"Saved adjusted simpoints to {opt_p_path}")
    
    return opt_p_path


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <workload_home>")
        print(f"  workload_home: Path to workload directory (containing fingerprint/)")
        sys.exit(1)
    
    workload_home = sys.argv[1]
    
    if not os.path.isdir(workload_home):
        print(f"Error: {workload_home} is not a directory")
        sys.exit(1)
    
    try:
        adjust_simpoints(workload_home)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()