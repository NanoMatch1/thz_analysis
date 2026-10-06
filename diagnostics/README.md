# diagnostics/

Scripts for **instrument and bench tests** — measurements of the setup itself rather than of a
sample: beam profile, purge settling, drift during a run. Each is a `*_run_me.py` with a
module-level `config` dict, run from the repo root. Reusable logic sits in a sibling module, so it
can be tested and imported without running the script.

| Script | Measures | Logic |
|---|---|---|
| `knife_edge_run_me.py` | Beam radius against frequency: spectral amplitude/intensity vs blade position (`..._<pos>mm...` in the filename), erf fit per frequency, CSV of the flattened columns | `knife_edge.py` |
| `acquisition_drift_run_me.py` | Drift scan by scan inside a run, and whether a purge transient has settled (exponential settling, wait-time verdict) | `dataset_core/adapters/acquisition_tracking.py` |

```
python diagnostics/knife_edge_run_me.py [<data directory>] [--no-graph]
python diagnostics/acquisition_drift_run_me.py        # edit data_dir in the script
```

Outputs go beside the data (`<data directory>/knife_edge_analysis/`) unless the config says
otherwise. The tools for the ellipsometry bench (wire-grid null, live view, HWP walk) stay in
`thz_ellipsometry_bench_run_me.py`, because they belong to that package.
