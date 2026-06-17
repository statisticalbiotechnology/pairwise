# denovo_benchmarks — `algorithms/pairwise`

Verbatim copy of the harness files that drive the PA container, taken from
[bittremieuxlab/denovo_benchmarks](https://github.com/bittremieuxlab/denovo_benchmarks/tree/main/algorithms/pairwise).
They are mirrored here so the container build steps documented in the top-level
[README.md](../../README.md) and [docs/backend_notes.md](../../docs/backend_notes.md)
can be read alongside this repo.

| File | Role |
| --- | --- |
| `create_lance.py` | Benchmark version of the spectra→Lance converter. The container copies this **over** `src/data/create_lance.py` (only difference vs the repo copy: `min_peaks=1` instead of `0`). |
| `make_predictions.sh` | Container runscript pipeline: build Lance → run `src/main.py` → map output. |
| `output_mapper.py` | Converts PA's mzTab into the benchmark common `outputs.csv` (needs the harness `algorithms/base` package). |
| `versions.log` | Harness bookkeeping (container version, upstream commit). |

The container definition itself is at [../container.def](../container.def). Its
`%files` section expects the harness layout (`algorithms/pairwise`,
`algorithms/base`), so it is meant to be built from inside a `denovo_benchmarks`
checkout, not from this repo's root.
