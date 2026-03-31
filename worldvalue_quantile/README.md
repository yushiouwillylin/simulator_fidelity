# WorldValue Quantile Reproduction

This folder packages the notebooks and archived figures needed to reproduce the WorldValueBench results used in the paper.

## What This Reproduces

- Human-response cleaning and retained-question filtering for the final 235-question subset.
- Synthetic-response post-processing for the LLM simulators.
- The WorldValue quantile plots and confidence-band analyses used in the final paper figures.

## Folder Contents

- `WV_datacleaning.ipynb`: builds cleaned human-response artifacts, prompt inputs, and the uniform baseline.
- `WV_llmcalls.ipynb`: merges raw LLM output shards and converts raw responses into numeric artifacts. The default published path does not make API calls.
- `WV_quantile_construction.ipynb`: reproduces the quantile analyses and links each main plotting block to the corresponding paper figure.
- `WV_quantile_embedding_benchmark.ipynb`: adds learned baseline simulators to the calibrated WorldValue quantile pipeline. It builds question-level features from survey text and metadata, cross-fits a main question-embedding predictor using question features only and an optional simulator-augmented predictor using question features plus simulator-side `qhat` features, reinterprets those out-of-fold predictions as simulator-side outputs `q_tilde`, and then benchmarks them under the same calibrated discrepancy pipeline used in `WV_quantile_construction.ipynb`.
- `wvs_notebook_helpers.py`: shared helpers for retained-question loading, filtering, and pickle compatibility.
- `wvs_data_preparation.py`: local preprocessing helpers used by the cleaning notebook.
- `simfidelity_utils.py`: local copy of the quantile utility module used by the quantile notebook.
- `figures/`: archived PNG copies of the main paper figures plus the benchmark comparison figures, and a figure manifest.
- `../datasets/worldvalue_data.zip`: source archive for the WorldValue bundle-local data tree.

## How To Use It

1. Install the core runtime dependencies:
   `pip install -r ../requirements.txt`
2. Run a preflight check that does not modify the environment:
   `python ../check_repro_environment.py --dataset worldvalue`
3. Restore the bundle-local data tree:
   `python ../datasets/unpack_reproduction_data.py --dataset worldvalue --worldvalue-layout minimal`
4. Run `WV_datacleaning.ipynb` if you want to regenerate the cleaned human-side artifacts or the retained-question uniform baseline.
5. Run `WV_llmcalls.ipynb` to merge raw shard files and convert synthetic answers into numeric artifacts.
6. If you want to experiment with fresh provider calls, set `RUN_GENERATION = True` in `WV_llmcalls.ipynb`, provide credentials in `.env.local`, and install `../requirements-optional-llm.txt`.
7. Run `WV_quantile_construction.ipynb` to reproduce the paper plots. The notebook includes links from each major plotting block to the final figure used in the manuscript.
8. Run `WV_quantile_embedding_benchmark.ipynb` if you want the additional learned-baseline benchmark analysis. That notebook fits a main question-embedding baseline using question features only, an optional simulator-augmented baseline using question features plus simulator-side `qhat` features, and a kernel-ridge-vs-kNN method comparison for both setups. It stores the learned out-of-fold prediction vectors as simulator outputs and then evaluates them inside the same robust discrepancy and calibrated quantile-curve pipeline as the LLM simulators. When executed, it now keeps a minimal output set under `worldvalue_quantile/output_embedding_benchmark/`: one feature snapshot, one prediction file per baseline variant, one fold-selection file per variant, one master `benchmark_qhat_dataframe.csv`, one master `calibrated_delta_dataframe.csv`, one master `curve_dataframe.csv`, and compact summary CSVs. The canonical benchmark PNGs are archived in `worldvalue_quantile/figures/`.

## Required Data And Inputs

- `data/worldvalue/`
- `data/worldvaluesbench/`

For the paper reproduction, only a minimal subset of `data/worldvaluesbench/` is required:

- `data/worldvaluesbench/dataset_construction/question_metadata.json`
- `data/worldvaluesbench/dataset_construction/codebook.json`
- `data/worldvaluesbench/dataset_construction/answer_adjustment.json`
- `data/worldvaluesbench/F00011356-WVS_Cross-National_Wave_7_csv_v6_0.zip`

The larger `data/worldvaluesbench/WorldValuesBench/` benchmark splits, `data/worldvaluesbench/output/` exports, auxiliary documentation, and Python helpers are not required for the quantile-figure reproduction and can be restored later with the full layout if needed.

In the trimmed reproduction bundle, most of the remaining `data/worldvaluesbench/` size comes from `F00011356-WVS_Cross-National_Wave_7_csv_v6_0.zip`, which is already the compressed raw WVS archive.

These paths are resolved inside the reproduction root. If they are missing, restore them from `../datasets/worldvalue_data.zip` with:

```bash
python ../datasets/unpack_reproduction_data.py --dataset worldvalue --worldvalue-layout minimal
```

The data-cleaning notebook checks these extracted paths explicitly and raises an error with the same restore command if they are missing.

## Figure Archive

The `figures/` directory stores PNG copies of both the paper figures and the benchmark comparison figures so the final plots can be inspected without rerunning the expensive cells. See `figures/README.md` for the figure-to-notebook mapping.

## Additional Benchmark Notebook

`WV_quantile_embedding_benchmark.ipynb` is not part of the original paper figure set. Its purpose is to evaluate how the calibrated WorldValue discrepancy curves compare with a learned baseline simulator fit on real survey question data. Concretely, it:

- reuses the retained question set, simulator bundle, and calibrated `qhat` pipeline from `WV_quantile_construction.ipynb`
- builds question representations from local survey text, answer options, and metadata
- cross-fits a question-only predictor on held-in folds of the finite-sample human target used in the calibrated benchmark setting
- stores the out-of-fold predictions as a baseline simulator-side output `q_tilde`
- feeds that learned baseline through the same confidence-set pseudo-discrepancy and calibrated quantile-curve code used for the LLM simulators
- emits diagnostic CSVs under `output_embedding_benchmark/`
- archives the main question-embedding baseline figure under `figures/`, while keeping the simulator-augmented comparison as an optional secondary artifact

The generated `output_embedding_benchmark/` directory is treated as run output rather than source, so it is expected to be recreated locally when the notebook is executed. The notebook prunes overlapping legacy CSV and PNG artifacts at the end of a clean run so the directory stays readable.
