#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import pandas
from pathlib import Path
from typing import Dict, List, Set

# scarab_stats lives at the repo root (two levels up from this file, under
# profiler/), not on the default import path, so add it explicitly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scarab_stats import stat_aggregator


class Loader:
    """
        Reads a Scarab stats CSV and exposes the workloads/configs/stats it contains.
    """

    def __init__(self, csv_path: Path) -> None:
        # Load the CSV and extract the workloads/configs/stats it contains.
        csv_path = Path(csv_path)
        self.aggregator = stat_aggregator()
        self.experiment = self.aggregator.load_experiment_csv(str(csv_path))

        # Workload identifiers, e.g. "spec2017/rate_int_v2/gcc_r".
        self.workloads: List[str] = sorted(self.experiment.get_workloads())

        # Named run configurations being compared, e.g. "bind_age", "unbind_amd_rr".
        self.configs: List[str] = sorted(self.experiment.get_configurations())

        # Stat names available for lookup, e.g. "IPC".
        self.available_stats: Set[str] = set(self.experiment.get_stats())

        if not self.workloads or not self.configs:
            raise RuntimeError("[ERROR] No workloads/configurations found in stats CSV.")
        print(
            f"[INFO] Loaded {len(self.workloads)} workloads, {len(self.configs)} configs, "
            f"{len(self.available_stats)} stats from {csv_path.name}."
        )

    def retrieve_stat(
        self,
        stat_name: str,
        *,
        configs: List[str],
        workloads: List[str],
    ) -> Dict[str, float]:
        """
            Look up one stat across configs/workloads.
            Returns a flat dict keyed by "{config} {workload} {stat_name}" -> value,
            as produced by scarab_stats' Experiment.retrieve_stats.
        """
        if stat_name not in self.available_stats:
            raise RuntimeError(f"[ERROR] Stat '{stat_name}' is not present.")
        if not workloads:
            raise RuntimeError("[ERROR] No workloads given to retrieve_stat.")

        data = self.experiment.retrieve_stats(configs, [stat_name], workloads)
        if data is None:
            raise RuntimeError(f"[ERROR] Failed to retrieve stat '{stat_name}'.")
        return data
