#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from tabulator import Tabulator

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "profiler" / "output"

# Matplotlib's tab20/tab20b/tab20c combined: 60 mutually distinct colors,
# shared by all bar charts, so even large counters/configs lists rarely
# have to repeat a color.
BAR_PALETTE = [
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
    "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
    "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
    "#17becf", "#9edae5",
    "#393b79", "#5254a3", "#6b6ecf", "#9c9ede", "#637939", "#8ca252",
    "#b5cf6b", "#cedb9c", "#8c6d31", "#bd9e39", "#e7ba52", "#e7cb94",
    "#843c39", "#ad494a", "#d6616b", "#e7969c", "#7b4173", "#a55194",
    "#ce6dbd", "#de9ed6",
    "#3182bd", "#6baed6", "#9ecae1", "#c6dbef", "#e6550d", "#fd8d3c",
    "#fdae6b", "#fdd0a2", "#31a354", "#74c476", "#a1d99b", "#c7e9c0",
    "#756bb1", "#9e9ac8", "#bcbddc", "#dadaeb", "#636363", "#969696",
    "#bdbdbd", "#d9d9d9",
]

# Cap legend columns so a long series list wraps into multiple rows instead
# of running off the side of the figure.
MAX_LEGEND_COLS = 4

# When "group_combined" is set, series contributing less than this share of
# the grand total are merged into a single "Other" series.
SMALL_GROUP_THRESHOLD = 0.02

# Bar charts are saved in both formats.
BAR_EXTENSIONS = ["png", "pdf"]


class Plotter:
    def __init__(self, tabulator: Tabulator, *, output_dir: Path = OUTPUT_DIR) -> None:
        self.tabulator = tabulator
        self.output_dir = Path(output_dir)
        self._handlers = {
            ("table", "markdown"): self._plot_markdown_table,
            ("table", "tsv"): self._plot_tsv_table,
            ("bar", "multi"): self._plot_bar_fig,
            ("bar", "stacked"): self._plot_stacked_bar_fig,
        }

    def plot(self, outputs: List[dict]) -> None:
        for output in outputs:
            if "subfig" in output:
                print(f"[INFO] Plotting subfig ({len(output['subfig'])} panels): {output.get('name')}")
                self._plot_subfig(output)
                continue

            output_type = output.get("type")
            output_format = output.get("format")
            handler = self._handlers.get((output_type, output_format))
            if handler is None:
                print(f"[ERROR] Unsupported output: type={output_type!r} format={output_format!r}, skipping.")
                continue
            counters = output.get("counters") or []
            detail = counters[0] if len(counters) == 1 else f"{len(counters)} counters" if counters else None
            print(f"[INFO] Plotting {output_type}/{output_format}" + (f" ({detail})" if detail else ""))
            handler(output)

    @staticmethod
    def _validate_group(group: str, *, counters: List[str], configs: Optional[List[str]]) -> None:
        """Every output is grouped by either "configs" or "counters": the
        grouping dimension may list many values, but the other one must
        name exactly one — a table/chart always varies along a single axis.
        """
        if group == "configs":
            if len(counters) != 1:
                raise RuntimeError(
                    f'[ERROR] group="configs" requires exactly one counter, got {counters!r}.'
                )
        elif group == "counters":
            if not configs or len(configs) != 1:
                raise RuntimeError(
                    f'[ERROR] group="counters" requires exactly one config, got {configs!r}.'
                )
        else:
            raise RuntimeError(f"[ERROR] Unsupported group: {group!r}.")

    @staticmethod
    def _simplify_labels(names: List[str]) -> List[str]:
        """Strip the longest common whole "_"-separated token prefix/suffix
        shared by all names, so legend entries show only what differs
        between them (e.g. "OP_ISSUED_count"/"OP_RETIRED_count" become
        "ISSUED"/"RETIRED", dropping the shared "OP" / "count" tokens).

        Only whole tokens are stripped — never a partial word — so e.g.
        "STALL_SHORT"/"STALLED_LONG" are left untouched instead of being
        cut into "_SHORT"/"ED_LONG".
        """
        if len(names) < 2:
            return list(names)

        token_lists = [name.split("_") for name in names]

        prefix_len = 0
        for tokens in zip(*token_lists):
            if len(set(tokens)) > 1:
                break
            prefix_len += 1

        suffix_len = 0
        for tokens in zip(*(t[::-1] for t in token_lists)):
            if len(set(tokens)) > 1:
                break
            suffix_len += 1

        shortest = min(len(tokens) for tokens in token_lists)
        prefix_len = min(prefix_len, shortest)
        suffix_len = min(suffix_len, shortest - prefix_len)

        stripped = [
            "_".join(tokens[prefix_len: len(tokens) - suffix_len]) for tokens in token_lists
        ]
        return names if any(not s for s in stripped) else stripped

    @staticmethod
    def _slug(name: str) -> str:
        """Turn a human-readable output name into a filename-safe slug
        (spaces -> underscores)."""
        return "_".join(name.split())

    @staticmethod
    def _combine_small_series(
        series_names: List[str],
        series_values: dict,
        *,
        threshold: float = SMALL_GROUP_THRESHOLD,
        other_label: str = "Other",
    ) -> Tuple[List[str], dict]:
        """Merge series whose share of the grand total falls below
        `threshold` into a single `other_label` series, so a long tail of
        negligible categories doesn't clutter the legend/colors."""
        def total(name: str) -> float:
            return sum(v for v in series_values[name] if v is not None)

        grand_total = sum(total(name) for name in series_names)
        if grand_total <= 0:
            return list(series_names), series_values

        kept, small = [], []
        for name in series_names:
            (kept if total(name) / grand_total >= threshold else small).append(name)

        if len(small) < 2:
            # Nothing meaningful to combine.
            return list(series_names), series_values

        num_categories = len(next(iter(series_values.values())))
        combined = [sum(series_values[name][i] or 0.0 for name in small) for i in range(num_categories)]

        new_values = {name: series_values[name] for name in kept}
        new_values[other_label] = combined
        return kept + [other_label], new_values

    def _write_output(
        self,
        name: str,
        output_type: str,
        output_format: str,
        *,
        extension: str,
        content: str,
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{output_type}_{output_format}_{name}.{extension}"
        output_path.write_text(content)
        print(f"[INFO] Wrote {output_path.name}")

    # ---- table (markdown/tsv share one matrix->rows pipeline) ----

    def _build_table_data(
        self, output: dict
    ) -> Tuple[str, Optional[str], List[str], List[List[str]], List[str], List[List[str]]]:
        """Resolve an output spec's group into a generic table.

        Returns (stat_name, resolved_baseline, value_headers, value_rows,
        speedup_headers, speedup_rows). `stat_name` is the counter
        (group=configs) or config (group=counters) the table is about.
        """
        group = output.get("group", "configs")
        counters = output.get("counters", [])
        configs = output.get("configs")
        self._validate_group(group, counters=counters, configs=configs)

        baseline = output.get("baseline")
        workloads = output.get("workloads")
        simpoints_combined = output.get("simpoints_combined", False)

        if group == "configs":
            counter = counters[0]
            build = (
                self.tabulator.build_combined_workload_matrix
                if simpoints_combined
                else self.tabulator.build_workload_matrix
            )
            resolved_baseline, row_labels, value_columns, speedup_columns, value_matrix, speedup_matrix = build(
                counter, baseline=baseline, configs=configs, workloads=workloads
            )
            stat_name = counter
            column_label = counter
        else:
            config = configs[0]
            ratio = output.get("ratio")
            build = (
                self.tabulator.build_combined_counter_matrix
                if simpoints_combined
                else self.tabulator.build_counter_matrix
            )
            resolved_baseline, row_labels, value_matrix, speedup_matrix = build(
                counters, baseline=baseline, config=config, workloads=workloads, ratio=ratio
            )
            value_columns = counters
            speedup_columns = counters
            stat_name = config
            column_label = f"{config}, % of {ratio}" if ratio else config

        display_rows = [self.tabulator.display_workload(label, workloads) for label in row_labels]

        value_headers = ["Workload"] + [f"{col} ({column_label})" for col in value_columns]
        value_rows = [
            [display] + [self.tabulator.format_numeric(value_matrix[label].get(col)) for col in value_columns]
            for label, display in zip(row_labels, display_rows)
        ]

        speedup_headers = ["Workload"]
        speedup_rows: List[List[str]] = []
        if resolved_baseline is not None:
            speedup_headers += [f"{col} speedup vs {resolved_baseline} (%)" for col in speedup_columns]
            speedup_rows = [
                [display] + [
                    self.tabulator.format_numeric(speedup_matrix[label].get(col), as_percent=True)
                    for col in speedup_columns
                ]
                for label, display in zip(row_labels, display_rows)
            ]

        return stat_name, resolved_baseline, value_headers, value_rows, speedup_headers, speedup_rows

    # ---- markdown ----

    def _format_markdown_table(self, headers: List[str], rows: List[List[str]]) -> str:
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    def _plot_markdown_table(self, output: dict) -> None:
        stat_name, baseline, value_headers, value_rows, speedup_headers, speedup_rows = self._build_table_data(output)

        value_table = self._format_markdown_table(value_headers, value_rows)
        content = f"# {stat_name}\n\n## Values\n\n{value_table}\n"

        if baseline is not None:
            speedup_table = self._format_markdown_table(speedup_headers, speedup_rows)
            content += f"\n## Speedup vs {baseline} (%)\n\n{speedup_table}\n"

        name = self._slug(output.get("name"))
        self._write_output(name, output.get("type"), output.get("format"), extension="md", content=content)

    # ---- tsv ----

    def _format_tsv_table(self, headers: List[str], rows: List[List[str]]) -> str:
        lines = ["\t".join(headers)]
        lines.extend("\t".join(row) for row in rows)
        return "\n".join(lines)

    def _plot_tsv_table(self, output: dict) -> None:
        stat_name, baseline, value_headers, value_rows, speedup_headers, speedup_rows = self._build_table_data(output)

        value_table = self._format_tsv_table(value_headers, value_rows)
        content = f"{stat_name}\n\nValues\n{value_table}\n"

        if baseline is not None:
            speedup_table = self._format_tsv_table(speedup_headers, speedup_rows)
            content += f"\nSpeedup vs {baseline} (%)\n{speedup_table}\n"

        name = self._slug(output.get("name"))
        self._write_output(name, output.get("type"), output.get("format"), extension="tsv", content=content)

    # ---- bar ----

    def _plot_bar_fig(self, output: dict) -> None:
        for chart in self._bar_charts(output):
            self._render_bar_fig(chart, output_type=output.get("type"), output_format=output.get("format"))

    def _plot_subfig(self, output: dict) -> None:
        """Render several bar output specs as subplot panels of one
        combined figure, instead of one file per spec."""
        subs = output.get("subfig", [])
        panels = [self._bar_charts(sub)[0] for sub in subs]

        output_type = subs[0].get("type", "bar") if subs else "bar"
        output_format = subs[0].get("format", "multi") if subs else "multi"

        self._render_bar_grid(
            panels,
            title=output.get("name"),
            output_type=output_type,
            output_format=output_format,
            suffix=self._slug(output.get("name")),
        )

    def _bar_charts(self, output: dict) -> List[dict]:
        """Compute (without rendering) the chart dict(s) for a bar output
        spec: one for "value", plus one for "speedup" if a baseline is set.
        Used both for standalone bar outputs and for each panel of a
        combined "subfig" grid."""
        group = output.get("group", "configs")
        counters = output.get("counters", [])
        configs = output.get("configs")
        self._validate_group(group, counters=counters, configs=configs)

        if group == "configs":
            return self._bar_charts_by_configs(output, counter=counters[0])
        return self._bar_charts_by_counters(output, config=configs[0])

    def _bar_charts_by_configs(self, output: dict, *, counter: str) -> List[dict]:
        baseline = output.get("baseline")
        configs = output.get("configs")
        workloads = output.get("workloads")
        build = (
            self.tabulator.build_combined_workload_matrix
            if output.get("simpoints_combined", False)
            else self.tabulator.build_workload_matrix
        )

        resolved_baseline, row_labels, value_configs, speedup_configs, value_matrix, speedup_matrix = (
            build(counter, baseline=baseline, configs=configs, workloads=workloads)
        )

        title = output.get("name")
        charts = [self._prepare_bar_chart(
            output, matrix=value_matrix, categories=row_labels, series_names=value_configs,
            kind="value", ylabel=counter, title=title,
        )]
        if resolved_baseline is not None:
            charts.append(self._prepare_bar_chart(
                output, matrix=speedup_matrix, categories=row_labels, series_names=speedup_configs,
                kind="speedup", ylabel="Speedup (%)", title=title,
            ))
        return charts

    def _bar_charts_by_counters(self, output: dict, *, config: str) -> List[dict]:
        baseline = output.get("baseline")
        counters = output.get("counters", [])
        workloads = output.get("workloads")
        ratio = output.get("ratio")
        build = (
            self.tabulator.build_combined_counter_matrix
            if output.get("simpoints_combined", False)
            else self.tabulator.build_counter_matrix
        )

        resolved_baseline, row_labels, value_matrix, speedup_matrix = build(
            counters, baseline=baseline, config=config, workloads=workloads, ratio=ratio
        )

        title = output.get("name")
        value_ylabel = "Percentage (%)" if ratio else "Value"
        charts = [self._prepare_bar_chart(
            output, matrix=value_matrix, categories=row_labels, series_names=counters,
            kind="value", ylabel=value_ylabel, title=title,
        )]
        if resolved_baseline is not None:
            charts.append(self._prepare_bar_chart(
                output, matrix=speedup_matrix, categories=row_labels, series_names=counters,
                kind="speedup", ylabel="Speedup (%)", title=title,
            ))
        return charts

    def _prepare_bar_chart(
        self,
        output: dict,
        *,
        matrix: dict,
        categories: List[str],
        series_names: List[str],
        kind: str,
        ylabel: str,
        title: str,
    ) -> dict:
        # categories already ends with the matrix's own aggregate "Avg" row.
        series_values = {name: [matrix[wl].get(name) for wl in categories] for name in series_names}

        if output.get("group_combined", False):
            series_names, series_values = self._combine_small_series(series_names, series_values)

        workloads = output.get("workloads")
        display_categories = [self.tabulator.display_workload(wl, workloads) for wl in categories]
        name_slug = self._slug(output.get("name"))

        return {
            "categories": display_categories,
            "series_names": series_names,
            "series_values": series_values,
            "ylabel": ylabel,
            "title": title,
            "suffix": f"{name_slug}_{kind}",
        }

    def _draw_grouped_bars(self, ax, chart: dict) -> Tuple[list, list]:
        """Draw one grouped-bar panel onto `ax`. Returns (handles, labels)
        for the caller to build a legend from (a standalone figure draws its
        own directly below; a subfig grid shares one legend for all panels)."""
        import math

        categories = chart["categories"]
        series_names = chart["series_names"]
        series_values = chart["series_values"]
        display_names = self._simplify_labels(series_names)

        num_series = max(len(series_names), 1)
        group_width = 0.45
        width = group_width / num_series
        # Groups sit closer together than one unit apart, so the empty gap
        # between them shrinks along with the thinner bars.
        group_spacing = 0.6
        x = [i * group_spacing for i in range(len(categories))]

        for i, name in enumerate(series_names):
            values = [v if v is not None and not math.isnan(v) else 0.0 for v in series_values[name]]
            offsets = [xi + i * width for xi in x]
            ax.bar(
                offsets, values, width, label=display_names[i],
                color=BAR_PALETTE[i % len(BAR_PALETTE)], edgecolor="black", linewidth=0.6,
            )

        tick_positions = [xi + (num_series - 1) * width / 2 for xi in x]
        ax.set_xticks(list(tick_positions))
        ax.set_xticklabels(categories, rotation=30, ha="right")
        ax.set_ylabel(chart["ylabel"], fontsize=13)
        ax.set_title(chart["title"], fontsize=16)

        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        ax.xaxis.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("black")
        ax.spines["bottom"].set_linewidth(1.0)
        ax.tick_params(axis="y", length=0)
        ax.axhline(0, color="black", linewidth=1.0)

        return ax.get_legend_handles_labels()

    def _render_bar_fig(self, chart: dict, *, output_type: str, output_format: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with plt.rc_context({"font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42}):
            fig, ax = plt.subplots(figsize=(max(3.5, len(chart["categories"]) * 0.55), 7))
            handles, labels = self._draw_grouped_bars(ax, chart)

            ncol = min(len(labels), MAX_LEGEND_COLS)
            ax.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=ncol)
            # Fixed margins (not fig.tight_layout(), which would shrink the
            # axes to make room for the legend) so the chart's own plotting
            # area stays a constant size regardless of how many legend rows
            # there are. bbox_inches="tight" below just grows the saved
            # canvas to fully include the legend, without touching the axes.
            fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.28)

            self.output_dir.mkdir(parents=True, exist_ok=True)
            for ext in BAR_EXTENSIONS:
                output_path = self.output_dir / f"{output_type}_{output_format}_{chart['suffix']}.{ext}"
                fig.savefig(output_path, bbox_inches="tight")
                print(f"[INFO] Wrote {output_path.name}")
            plt.close(fig)

    def _render_bar_grid(
        self, panels: List[dict], *, title: str, output_type: str, output_format: str, suffix: str
    ) -> None:
        """Render several precomputed chart dicts as subplot panels of one
        combined figure (2 columns, as many rows as needed), sharing a
        single legend for the whole grid."""
        import math

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ncols = 2
        nrows = math.ceil(len(panels) / ncols)

        with plt.rc_context({"font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42}):
            fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows), squeeze=False)
            used_axes = []
            handles, labels = [], []
            for idx, chart in enumerate(panels):
                ax = axes[idx // ncols][idx % ncols]
                used_axes.append(ax)
                panel_handles, panel_labels = self._draw_grouped_bars(ax, chart)
                if not handles:
                    handles, labels = panel_handles, panel_labels

            for idx in range(len(panels), nrows * ncols):
                axes[idx // ncols][idx % ncols].axis("off")

            # Same y-axis range on every panel so bar heights are directly
            # comparable across panels, not just within one.
            ylims = [ax.get_ylim() for ax in used_axes]
            shared_ylim = (min(y[0] for y in ylims), max(y[1] for y in ylims))
            for ax in used_axes:
                ax.set_ylim(shared_ylim)

            fig.suptitle(title, fontsize=18)
            ncol = min(len(labels), MAX_LEGEND_COLS)
            fig.legend(handles, labels, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=ncol)
            fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.16, hspace=0.3, wspace=0.15)

            self.output_dir.mkdir(parents=True, exist_ok=True)
            for ext in BAR_EXTENSIONS:
                output_path = self.output_dir / f"{output_type}_{output_format}_{suffix}.{ext}"
                fig.savefig(output_path, bbox_inches="tight")
                print(f"[INFO] Wrote {output_path.name}")
            plt.close(fig)

    # ---- stacked bar ----

    def _plot_stacked_bar_fig(self, output: dict) -> None:
        group = output.get("group", "configs")
        baseline = output.get("baseline")
        configs = output.get("configs")
        counters = output.get("counters", [])
        workloads = output.get("workloads")

        self._validate_group(group, counters=counters, configs=configs)
        if baseline is not None:
            raise RuntimeError('[ERROR] bar/stacked requires baseline to be null.')

        config = configs[0]
        build = (
            self.tabulator.build_combined_counter_matrix
            if output.get("simpoints_combined", False)
            else self.tabulator.build_counter_matrix
        )
        _, row_labels, value_matrix, _ = build(
            counters, baseline=baseline, config=config, workloads=workloads
        )

        def as_percentages(row: dict) -> dict:
            total = sum(v for v in row.values() if v is not None)
            return {c: (row.get(c) or 0.0) / total * 100 if total else 0.0 for c in counters}

        # row_labels already ends with the matrix's own aggregate "Avg" row.
        percentages = {label: as_percentages(value_matrix[label]) for label in row_labels}
        categories = [self.tabulator.display_workload(label, workloads) for label in row_labels]
        series_values = {c: [percentages[label][c] for label in row_labels] for c in counters}
        series_names = counters

        if output.get("group_combined", False):
            series_names, series_values = self._combine_small_series(series_names, series_values)

        self._render_stacked_bar_fig(
            categories=categories,
            series_names=series_names,
            series_values=series_values,
            ylabel="Percentage (%)",
            title=output.get("name"),
            output_type=output.get("type"),
            output_format=output.get("format"),
            suffix=self._slug(output.get("name")),
        )

    def _render_stacked_bar_fig(
        self,
        *,
        categories: List[str],
        series_names: List[str],
        series_values: dict,
        ylabel: str,
        title: str,
        output_type: str,
        output_format: str,
        suffix: str,
    ) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        display_names = self._simplify_labels(series_names)

        width = 0.4
        x = [i * 0.8 for i in range(len(categories))]

        with plt.rc_context({"font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42}):
            fig, ax = plt.subplots(figsize=(max(3.5, len(categories) * 0.5), 7))
            bottoms = [0.0] * len(categories)
            for i, name in enumerate(series_names):
                values = [v if v is not None else 0.0 for v in series_values[name]]
                ax.bar(
                    x, values, width, bottom=bottoms, label=display_names[i],
                    color=BAR_PALETTE[i % len(BAR_PALETTE)], edgecolor="black", linewidth=0.6,
                )
                bottoms = [b + v for b, v in zip(bottoms, values)]

            ax.set_xticks(x)
            ax.set_xticklabels(categories, rotation=30, ha="right")
            ax.set_ylabel(ylabel, fontsize=13)
            ax.set_title(title, fontsize=16)
            ax.set_ylim(0, 100)

            ax.set_axisbelow(True)
            ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
            ax.xaxis.grid(False)
            ncol = min(len(series_names), MAX_LEGEND_COLS)
            ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=ncol)
            # Fixed margins (see _render_bar_fig) so the axes keep a constant
            # size regardless of how many legend rows there are.
            fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.3)

            self.output_dir.mkdir(parents=True, exist_ok=True)
            for ext in BAR_EXTENSIONS:
                output_path = self.output_dir / f"{output_type}_{output_format}_{suffix}.{ext}"
                fig.savefig(output_path, bbox_inches="tight")
                print(f"[INFO] Wrote {output_path.name}")
            plt.close(fig)
