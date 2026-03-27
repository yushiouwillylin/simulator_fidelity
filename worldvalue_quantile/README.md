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
- `wvs_notebook_helpers.py`: shared helpers for retained-question loading, filtering, and pickle compatibility.
- `wvs_data_preparation.py`: local preprocessing helpers used by the cleaning notebook.
- `simfidelity_utils.py`: local copy of the quantile utility module used by the quantile notebook.
- `figures/`: archived PNG copies of the main paper figures and a figure manifest.
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

The `figures/` directory stores PNG copies of the paper figures so the final plots can be inspected without rerunning the expensive cells, and the notebook now saves archived figures directly there instead of the repo root. See `figures/README.md` for the figure-to-notebook mapping.
