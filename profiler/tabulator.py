#!/usr/bin/env python3
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

from loader import Loader


class Tabulator:
    def __init__(self, loader: Loader) -> None:
        self.loader = loader

    def filter_workloads(self, prefix: str) -> List[str]:
        """Select the loaded workloads that start with the given prefix."""
        return [wl for wl in self.loader.workloads if wl.startswith(prefix)]

    @staticmethod
    def display_workload(workload: str, prefix: str) -> str:
        """Shorten a workload identifier for display in labels/table rows."""
        return workload.removeprefix(prefix)

    @staticmethod
    def base_workload(workload: str) -> str:
        """Strip a trailing numeric simpoint suffix, e.g. '..._2' -> '...'."""
        return re.sub(r"_\d+$", "", workload)

    @classmethod
    def _workload_groups(cls, workloads: List[str], *, combined: bool) -> List[Tuple[str, List[str]]]:
        """Return (label, member_workloads) pairs.

        combined=False: one row per workload (label == workload).
        combined=True: workloads sharing a base name (e.g. gcc_r, gcc_r_2,
        gcc_r_3 are separate simpoints of the same benchmark) are merged into
        a single row labeled by the base name.
        """
        if not combined:
            return [(wl, [wl]) for wl in workloads]

        groups: Dict[str, List[str]] = {}
        for wl in workloads:
            groups.setdefault(cls.base_workload(wl), []).append(wl)
        return list(groups.items())

    @staticmethod
    def format_numeric(value: Optional[float], *, as_percent: bool = False) -> str:
        if value is None:
            return "N/A"
        if not isinstance(value, (int, float)):
            return str(value)
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"

        formatted = f"{value:.4f}" if not as_percent else f"{value:.2f}"
        formatted = formatted.rstrip("0").rstrip(".")
        return f"{formatted}%" if as_percent else formatted

    @staticmethod
    def average(values: List[float], *, use_geomean: bool) -> Optional[float]:
        if not values:
            return None
        if use_geomean:
            product = 1.0
            for val in values:
                product *= val
            return product ** (1 / len(values))
        return sum(values) / len(values)

    @staticmethod
    def speedup(cur: Optional[float], base: Optional[float]) -> Optional[float]:
        if cur is None or base is None:
            return None
        if base == 0:
            return math.inf if cur > 0 else None
        return (cur / base - 1.0) * 100.0

    def resolve_baseline(self, baseline: str) -> Tuple[str, List[str]]:
        """Validate the baseline and order configs with it first.

        Callers must only invoke this when a baseline was actually given —
        a null/absent baseline means "don't compute speedup" and should be
        handled before reaching here.
        """
        configs = self.loader.configs
        if baseline not in configs:
            raise RuntimeError(f"[ERROR] Baseline config '{baseline}' is not present.")

        # Baseline goes first so comparison tables list it as the leftmost column.
        configs_ordered = [baseline] + [cfg for cfg in configs if cfg != baseline]
        return baseline, configs_ordered

    def _average_row(
        self,
        row_labels: List[str],
        matrix: Dict[str, Dict[str, Optional[float]]],
        columns: List[str],
        use_geomean: Dict[str, bool],
    ) -> Dict[str, Optional[float]]:
        """Average each column across the given rows (e.g. to derive "Avg")."""
        return {
            col: self.average(
                [matrix[label].get(col) for label in row_labels if matrix[label].get(col) is not None],
                use_geomean=use_geomean.get(col, False),
            )
            for col in columns
        }

    def _build_workload_matrix(
        self,
        stat_name: str,
        *,
        baseline: Optional[str],
        configs: Optional[List[str]],
        workloads: str,
        combined: bool,
    ) -> Tuple[
        str,
        List[str],
        List[str],
        List[str],
        Dict[str, Dict[str, Optional[float]]],
        Dict[str, Dict[str, Optional[float]]],
    ]:
        loader = self.loader
        has_baseline = baseline is not None
        if has_baseline:
            resolved_baseline, all_configs = self.resolve_baseline(baseline)
            configs_used = configs if configs else [cfg for cfg in all_configs if cfg != resolved_baseline]
            value_configs = [resolved_baseline] + configs_used
        else:
            resolved_baseline = None
            configs_used = configs if configs else loader.configs
            value_configs = configs_used

        workloads_used = self.filter_workloads(workloads)

        data = loader.retrieve_stat(stat_name, configs=value_configs, workloads=workloads_used)
        use_geomean = stat_name.split("_")[-1] == "pct"
        groups = self._workload_groups(workloads_used, combined=combined)

        def group_average(cfg: str, members: List[str]) -> Optional[float]:
            values = [data.get(f"{cfg} {wl} {stat_name}") for wl in members]
            return self.average([v for v in values if v is not None], use_geomean=use_geomean)

        row_labels: List[str] = []
        value_matrix: Dict[str, Dict[str, Optional[float]]] = {}
        for label, members in groups:
            row_labels.append(label)
            value_matrix[label] = {cfg: group_average(cfg, members) for cfg in value_configs}

        # Derive "Avg" from the displayed rows (not the raw per-simpoint
        # workloads), so a benchmark with more simpoints doesn't outweigh
        # one with fewer once combined.
        use_geomean_by_col = {cfg: use_geomean for cfg in value_configs}
        value_matrix["Avg"] = self._average_row(row_labels, value_matrix, value_configs, use_geomean_by_col)
        row_labels = row_labels + ["Avg"]

        speedup_matrix: Dict[str, Dict[str, Optional[float]]] = {}
        for label in row_labels:
            if has_baseline:
                baseline_val = value_matrix[label][resolved_baseline]
                speedup_matrix[label] = {
                    cfg: self.speedup(value_matrix[label][cfg], baseline_val) for cfg in configs_used
                }
            else:
                speedup_matrix[label] = {}

        speedup_configs = configs_used if has_baseline else []
        return resolved_baseline, row_labels, value_configs, speedup_configs, value_matrix, speedup_matrix

    def build_workload_matrix(
        self,
        stat_name: str,
        *,
        baseline: Optional[str] = None,
        configs: Optional[List[str]] = None,
        workloads: str,
    ) -> Tuple[
        str,
        List[str],
        List[str],
        List[str],
        Dict[str, Dict[str, Optional[float]]],
        Dict[str, Dict[str, Optional[float]]],
    ]:
        """Per-workload values and speedup-vs-baseline for one stat across configs.

        `workloads` is a required prefix (e.g. "spec2017/rate_int_v2/") used
        to select which loaded workloads to include.

        Returns (resolved_baseline, row_labels, value_configs, speedup_configs,
        value_matrix, speedup_matrix) where value_matrix[row][config] is the
        raw value (baseline + configs_used) and speedup_matrix[row][config]
        is the speedup percentage vs baseline (configs_used only). row_labels
        ends with an aggregate "Avg" row.
        """
        return self._build_workload_matrix(
            stat_name, baseline=baseline, configs=configs, workloads=workloads, combined=False
        )

    def build_combined_workload_matrix(
        self,
        stat_name: str,
        *,
        baseline: Optional[str] = None,
        configs: Optional[List[str]] = None,
        workloads: str,
    ) -> Tuple[
        str,
        List[str],
        List[str],
        List[str],
        Dict[str, Dict[str, Optional[float]]],
        Dict[str, Dict[str, Optional[float]]],
    ]:
        """Same as build_workload_matrix, but simpoints of the same benchmark
        (e.g. gcc_r, gcc_r_2, gcc_r_3) are averaged into a single row."""
        return self._build_workload_matrix(
            stat_name, baseline=baseline, configs=configs, workloads=workloads, combined=True
        )

    @staticmethod
    def _as_percent_of(value: Optional[float], total: Optional[float]) -> Optional[float]:
        if value is None or total is None or total == 0:
            return None
        return value / total * 100.0

    def _build_counter_matrix(
        self,
        counters: List[str],
        *,
        baseline: Optional[str],
        config: str,
        workloads: str,
        combined: bool,
        ratio: Optional[str] = None,
    ) -> Tuple[str, List[str], Dict[str, Dict[str, Optional[float]]], Dict[str, Dict[str, Optional[float]]]]:
        loader = self.loader
        has_baseline = baseline is not None
        resolved_baseline = self.resolve_baseline(baseline)[0] if has_baseline else None
        workloads_used = self.filter_workloads(workloads)
        groups = self._workload_groups(workloads_used, combined=combined)

        row_labels: List[str] = [label for label, _ in groups]
        value_matrix: Dict[str, Dict[str, Optional[float]]] = {label: {} for label in row_labels}
        baseline_matrix: Dict[str, Dict[str, Optional[float]]] = {label: {} for label in row_labels}
        use_geomean_by_col: Dict[str, bool] = {}

        for counter in counters:
            use_geomean = counter.split("_")[-1] == "pct"
            use_geomean_by_col[counter] = use_geomean
            configs_needed = [config, resolved_baseline] if has_baseline else [config]
            data = loader.retrieve_stat(counter, configs=configs_needed, workloads=workloads_used)

            def group_average(cfg: str, members: List[str]) -> Optional[float]:
                values = [data.get(f"{cfg} {wl} {counter}") for wl in members]
                return self.average([v for v in values if v is not None], use_geomean=use_geomean)

            for label, members in groups:
                value_matrix[label][counter] = group_average(config, members)
                if has_baseline:
                    baseline_matrix[label][counter] = group_average(resolved_baseline, members)

        # Derive "Avg" from the displayed rows, same as _build_workload_matrix.
        value_matrix["Avg"] = self._average_row(row_labels, value_matrix, counters, use_geomean_by_col)
        if has_baseline:
            baseline_matrix["Avg"] = self._average_row(row_labels, baseline_matrix, counters, use_geomean_by_col)
        row_labels = row_labels + ["Avg"]

        if ratio is not None:
            # Normalize each counter's (raw) value by a separate "total"
            # counter's value for the same rows/config, as a percentage.
            # Derived from the same raw-averaged rows used for "Avg" above,
            # so it's "ratio of averages" rather than "average of ratios".
            ratio_configs = [config, resolved_baseline] if has_baseline else [config]
            ratio_data = loader.retrieve_stat(ratio, configs=ratio_configs, workloads=workloads_used)

            def ratio_group_average(cfg: str, members: List[str]) -> Optional[float]:
                values = [ratio_data.get(f"{cfg} {wl} {ratio}") for wl in members]
                return self.average([v for v in values if v is not None], use_geomean=False)

            ratio_by_label = {label: ratio_group_average(config, members) for label, members in groups}
            ratio_by_label["Avg"] = self.average(
                [v for v in ratio_by_label.values() if v is not None], use_geomean=False
            )
            for label in row_labels:
                ratio_val = ratio_by_label[label]
                for counter in counters:
                    value_matrix[label][counter] = self._as_percent_of(value_matrix[label].get(counter), ratio_val)

            if has_baseline:
                ratio_baseline_by_label = {
                    label: ratio_group_average(resolved_baseline, members) for label, members in groups
                }
                ratio_baseline_by_label["Avg"] = self.average(
                    [v for v in ratio_baseline_by_label.values() if v is not None], use_geomean=False
                )
                for label in row_labels:
                    ratio_baseline_val = ratio_baseline_by_label[label]
                    for counter in counters:
                        baseline_matrix[label][counter] = self._as_percent_of(
                            baseline_matrix[label].get(counter), ratio_baseline_val
                        )

        speedup_matrix: Dict[str, Dict[str, Optional[float]]] = {}
        for label in row_labels:
            if has_baseline:
                speedup_matrix[label] = {
                    c: self.speedup(value_matrix[label].get(c), baseline_matrix[label].get(c)) for c in counters
                }
            else:
                speedup_matrix[label] = {}

        return resolved_baseline, row_labels, value_matrix, speedup_matrix

    def build_counter_matrix(
        self,
        counters: List[str],
        *,
        baseline: Optional[str] = None,
        config: str,
        workloads: str,
        ratio: Optional[str] = None,
    ) -> Tuple[str, List[str], Dict[str, Dict[str, Optional[float]]], Dict[str, Dict[str, Optional[float]]]]:
        """Per-workload values and speedup-vs-baseline for one config across
        multiple counters (stats).

        `workloads` is a required prefix (e.g. "spec2017/rate_int_v2/") used
        to select which loaded workloads to include. When `ratio` (a counter
        name) is given, each counter's value is expressed as a percentage of
        that counter's value instead of its raw value.

        Returns (resolved_baseline, row_labels, value_matrix, speedup_matrix)
        where value_matrix[row][counter] is the raw (or ratio %) value for
        `config`, and speedup_matrix[row][counter] is the speedup percentage
        of `config` vs baseline for that counter. row_labels ends with an
        aggregate "Avg" row.
        """
        return self._build_counter_matrix(
            counters, baseline=baseline, config=config, workloads=workloads, combined=False, ratio=ratio
        )

    def build_combined_counter_matrix(
        self,
        counters: List[str],
        *,
        baseline: Optional[str] = None,
        config: str,
        workloads: str,
        ratio: Optional[str] = None,
    ) -> Tuple[str, List[str], Dict[str, Dict[str, Optional[float]]], Dict[str, Dict[str, Optional[float]]]]:
        """Same as build_counter_matrix, but simpoints of the same benchmark
        (e.g. gcc_r, gcc_r_2, gcc_r_3) are averaged into a single row."""
        return self._build_counter_matrix(
            counters, baseline=baseline, config=config, workloads=workloads, combined=True, ratio=ratio
        )
