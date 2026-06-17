# PA backend notes

Integration and reproducibility notes for using PA as a de-novo backend (e.g. the
[denovo_benchmarks](https://github.com/bittremieuxlab/denovo_benchmarks) harness, or
the *borgonovo* reference-free assembler). See the top-level
[README.md](../README.md) for the container and the high-level I/O contract; this
file collects the lower-level gotchas. Every claim is cited to `file:line`.

## Environment / build gotchas

- **Submodule URL is SSH.** [`.gitmodules`](../.gitmodules) points
  `depthcharge` at `git@github.com:Alfred-N/depthcharge.git`, so a recursive
  clone fails without an SSH key. The container works around this by rewriting
  the URL to HTTPS before `git submodule update`
  ([`misc/container.def:19-21`](../misc/container.def)); do the same (or clone the
  fork over HTTPS) outside the container.
- **`pkg_resources` / setuptools.** [`environment.yml:11`](../environment.yml)
  pins `pytorch-lightning==2.1.1`, which still imports the deprecated
  `pkg_resources`. Recent setuptools (≥ 81) no longer ship it, and
  `environment.yml` does not pin setuptools — if you hit
  `ModuleNotFoundError: pkg_resources`, add `setuptools<81` to the env.
- **mzML support is untested in the bundled env.** depthcharge parses mzML/mzXML
  through `pyteomics.mzml`/`pyteomics.mzxml`
  (depthcharge `depthcharge/data/parsers.py:14-15`), which typically also needs
  `psims` for PSI-MS CV terms. `environment.yml` only exercises the MGF path, so
  expect to install extra pyteomics dependencies before using `--suffix .mzML`.

## Running `src/main.py` directly

- **Run from the repo root.** Config paths are relative — `configs/master_bm.yaml`
  references `configs/downstream/casanovo_bm.yaml`
  ([`configs/master_bm.yaml:38`](../configs/master_bm.yaml)) — and the container
  `cd`s into the repo before invoking it
  ([`misc/denovo_benchmarks_pairwise/make_predictions.sh:15-20`](../misc/denovo_benchmarks_pairwise/make_predictions.sh)).
- **`--predict_only` takes a value.** It is `type=int, default=0`
  ([`src/parse_args.py:221-226`](../src/parse_args.py)) cast to bool
  (`parse_args.py:306`), i.e. `--predict_only=1`, not a bare flag. (In the
  container it comes from `predict_only: 1` in `configs/master_bm.yaml:30`.)
  Prediction then runs through `trainer.predict` (`src/main.py:207-209`).
- **Output-dir creation aborts if `outs/` exists.** `create_output_dirs` calls
  `os.makedirs(..., exist_ok=False)` for both `output_dir` and `log_dir`
  ([`src/parse_args.py:391-392`](../src/parse_args.py)). With `fixed_output_dir: 1`
  (`configs/master_bm.yaml:31`) the paths are not uniquified, so a leftover `outs/`
  from a previous run will crash the next one — delete it first.

## Driving the model without the Lightning Trainer

- **`setup()` needs a Trainer.** `BenchmarkDataModule.setup` (and
  `LanceDataModule.setup`) read `self.trainer.global_rank` / `self.trainer.world_size`
  to build the `ShardedBatchSampler`
  ([`src/data/lance_data_module.py:119-125`](../src/data/lance_data_module.py)).
  To build a dataloader without a real `pl.Trainer`, attach a dummy, e.g.
  `dm.trainer = SimpleNamespace(global_rank=0, world_size=1)` before calling
  `dm.setup("predict")`.

## Calling the decoder directly (per-step distributions)

Neither the container nor `make_predictions.sh` emits per-step probability
distributions over the token vocabulary (the mzTab carries only scalar per-residue
confidences, and `outputs.csv` simulates even those —
[`output_mapper.py:191`](../misc/denovo_benchmarks_pairwise/output_mapper.py)).
A borgonovo-style backend that needs the distributions must call the model
directly:

- The decoder behind `decoder_model: casanovo_decoder` is
  `PeptideDecoder.forward(tokens, precursors, memory, memory_key_padding_mask)`
  ([`src/models/casanovo/decoder_interface.py:109,166`](../src/models/casanovo/decoder_interface.py)).
- It returns `(scores, tokens)`, with `scores` of shape `(B, L, V)` (raw logits;
  softmax for probabilities) — `return self.final(preds), tokens`
  (`decoder_interface.py:163`).
- The **first** step uses `decoder(None, precursors, memory, mem_masks)`; `None`
  tokens means only the precursor embedding is fed in
  (`decoder_interface.py:137-141`; called at
  [`src/wrappers/beam_search.py:84`](../src/wrappers/beam_search.py), then
  iteratively at `beam_search.py:121-126`).
- `precursors` is `[mass, charge, mz]`, where `mass = mz·charge` (protonated
  cluster mass — see the precursor-mass note in the README)
  (`src/wrappers/casanovo_trainer_wrapper.py:150-154`).
- `decoder.reverse` controls decoding direction; the released checkpoint runs with
  `reverse: True` (`configs/downstream/casanovo_bm.yaml`), set onto the decoder at
  `src/wrappers/casanovo_trainer_wrapper.py:99`, which means **C→N** order (stop
  token `$` appears first; `beam_search.py:232`).
- Greedy decoding is just beam search with `n_beams: 1`
  (`configs/downstream/casanovo_bm.yaml`; wrapper default at
  `src/wrappers/casanovo_trainer_wrapper.py:76`).
- Token labels are `C+57.021`, `M+15.995`, … ([`src/utils.py:33,49`](../src/utils.py)),
  not Casanovo's `C[Carbamidomethyl]` / ProForma `C[UNIMOD:4]`.
