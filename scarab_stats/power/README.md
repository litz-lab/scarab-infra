# 🧩 McPAT–Scarab Power Pipeline

End-to-end power and energy estimation for Scarab simulations.
Converts Scarab simulation outputs into McPAT input XML, runs McPAT,
and aggregates per-simpoint results into simpoint-weighted
whole-workload power and energy.

---

## 🚀 Quick usage

```
./sci --power <DESCRIPTOR>
```

For each (suite, subsuite, workload, configuration) under the
descriptor's experiment directory, walks per-simpoint Scarab outputs,
runs McPAT once per simpoint (cached), and writes a simpoint-weighted
summary at `<root_dir>/simulations/<experiment>/power_summary.csv`.

Columns: `suite, subsuite, workload, config, n_simpoints_with_power,
weight_covered, avg_runtime_dynamic_W, avg_leakage_W, avg_total_W,
weighted_cycles, sim_time_s, energy_J`.

Energy = `avg_total_W * sim_time_s`, where `sim_time_s =
weighted_cycles / clock_freq` (clock defaults to 4.0 GHz; pass
`--clock-ghz` to override the underlying aggregator).

---

## 📁 Structure

```
.
├── converter.py                    # Conversion logic (Converter class)
├── parse_power_stat.py             # Reads scarab power.stat.0.csv totals
├── client.py                       # Manual 4-step CLI (lower-level)
├── aggregate_workload_power.py     # Simpoint-weighted whole-workload pipeline
├── xml/
│   └── template.xml                # McPAT XML template
├── template/                       # Field-mapping templates
│   ├── mcpat_structure.json
│   ├── params_table.json
│   └── stats_table.json
└── mcpat                           # McPAT binary
```

---

## 🧱 Pipeline

`./sci --power` invokes `aggregate_workload_power.py`, which for each
simpoint output dir does:

1. **Parse** `PARAMS.out` and `power.stat.0.csv` (both auto-emitted by
   Scarab in each simpoint sim dir).
2. **Build XML** via `Converter`: `template.xml` → JSON tables →
   merged with parsed PARAMS + stats → final `mcpat_infile.xml`.
3. **Run McPAT** on the XML.
4. **Cache** the parsed Processor-block power numbers (Peak Dynamic,
   Subthreshold Leakage, Gate Leakage, Runtime Dynamic, Total) at
   `<sim_dir>/power/power_summary.json` for idempotent re-runs.

After all simpoints are processed, weights from
`workloads/workloads_db.json` aggregate the per-simpoint summaries
into one whole-workload row per (suite, config).

---

## 🛠 Manual lower-level usage

`client.py` exposes the conversion as four discrete steps for
debugging or one-off use against an arbitrary scarab output dir.
Either pass paths explicitly to each step, or symlink your sim dir at
`./scarab_output/` so the default paths resolve.

```
python client.py all
```

### 1️⃣ Generate initial JSON files from McPAT XML template
```
python client.py generate-json
```
 - input: ./xml/template.xml
 - output: ./json/

Parses the McPAT template and emits three JSON files:
`mcpat_structure.json` (hierarchical tree), `params_table.json`
(static architectural parameters), and `stats_table.json` (dynamic
runtime statistics).

### 2️⃣ Update JSON with Scarab PARAMS file
```
python client.py update-params
```
 - input: ./scarab_output/PARAMS.out
 - output: ./json/params_table.json

Reads `PARAMS.out` from a scarab simulation dir and updates the
corresponding fields in `params_table.json`.

### 3️⃣ Update JSON with Scarab STATS file
```
python client.py update-stats
```
 - input: ./scarab_output/power.pkl
 - output: ./json/stats_table.json

Updates dynamic power statistics in `stats_table.json`. The
production pipeline (`./sci --power`) reads
`scarab_output/power.stat.0.csv` directly via
`parse_power_stat.scarab_csv_to_power_dict()`; this manual step is
retained only for cases where a pre-built dict is preferred.

### 4️⃣ Generate final McPAT XML file
```
python client.py generate-xml
```
 - input: ./json
 - output: ./xml/mcpat_infile.xml

Combines `params_table.json` and `stats_table.json` according to
`mcpat_structure.json` and regenerates the complete McPAT XML input
file.

---

## 🔬 Direct McPAT invocation

After step ④ (or after `./sci --power` has produced
`<sim_dir>/power/mcpat_infile.xml`):

```
./mcpat -dump_design -infile xml/mcpat_infile.xml -print_level 5 > scarab.out
```
