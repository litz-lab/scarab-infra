# 🧩 McPAT–Scarab Power Pipeline

End-to-end power and energy estimation for Scarab simulations.
Converts each simpoint's Scarab output into McPAT input XML, runs
McPAT, and aggregates the per-simpoint results into a
simpoint-weighted whole-workload summary.

---

## 🚀 Usage

```
./sci --init           # builds McPAT 1.3 from source if not already present
./sci --power <DESCRIPTOR>
```

`./sci --init` includes an optional step that runs
`install_mcpat.sh`, which clones `HewlettPackard/mcpat` at v1.3.0
and builds the binary at `scarab_stats/power/mcpat`. The script can
also be run directly. Set `MCPAT_BIN=/path/to/mcpat` to point at a
pre-built binary instead.


For each (suite, subsuite, workload, configuration) under the
descriptor's experiment directory, the pipeline:

1. Walks per-simpoint Scarab outputs.
2. For each simpoint: parses `PARAMS.out` and `power.stat.0.csv`,
   builds the McPAT input XML, runs the `mcpat` binary, and parses
   the Processor block (Peak Dynamic, Subthreshold Leakage, Gate
   Leakage, Runtime Dynamic, Total). Results are cached at
   `<sim_dir>/power/power_summary.json` for idempotent re-runs.
3. Aggregates per-simpoint summaries using simpoint weights from
   `workloads/workloads_db.json`.
4. Writes `<root_dir>/simulations/<experiment>/power_summary.csv`.

Output columns: `suite, subsuite, workload, config,
n_simpoints_with_power, weight_covered, avg_runtime_dynamic_W,
avg_leakage_W, avg_total_W, weighted_cycles, sim_time_s, energy_J`.

Energy is `avg_total_W × sim_time_s`, where `sim_time_s =
weighted_cycles / clock_freq` (default 4.0 GHz).

---

## 📁 Structure

```
.
├── converter.py                    # Scarab↔McPAT field-mapping (Converter class)
├── parse_power_stat.py             # Reads scarab power.stat.0.csv totals
├── aggregate_workload_power.py     # Per-simpoint pipeline + weighted aggregation
├── install_mcpat.sh                # Clone + build McPAT 1.3 (HewlettPackard/mcpat v1.3.0)
└── xml/
    └── template.xml                # McPAT XML template
```

`Converter.generate_json_from_template()` parses `xml/template.xml` and emits the field-mapping JSON tables (`mcpat_structure.json`, `params_table.json`, `stats_table.json`) into a per-simpoint working dir on every run, so no static JSON tables need to be checked in.
