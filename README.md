# Model-Free Assessment of Simulator Fidelity via Quantile Curves

This directory is a self-contained reproduction bundle for the paper figures. The notebooks, archived figures, helper modules, and extracted datasets all live inside this folder.

## Contents

- `worldvalue_quantile/`: WorldValueBench preprocessing, simulator post-processing, and quantile-plot reproduction.
- `eedi_opinionqa_quantile/`: EEDI and OpinionQA quantile-plot reproduction from prepared dataset artifacts.
- `datasets/`: zip archives plus an unpack script for the underlying datasets used by the reproduction notebooks.
- `data/`: extracted bundle-local datasets used by the notebooks after unpacking.

## Canonical Output Locations

- WorldValue figures are archived in `worldvalue_quantile/figures/`.
- EEDI and OpinionQA figures are archived in `eedi_opinionqa_quantile/figures/`.
- Rebuilt WorldValue derived datasets stay under `data/worldvalue/`.

## Typical Use

1. Install the core runtime dependencies:
   `pip install -r requirements.txt`
2. Run a non-invasive preflight check:
   `python check_repro_environment.py --dataset all`
3. Restore the extracted bundle-local datasets:
   From the repository root:
   `python datasets/unpack_reproduction_data.py`
4. Open the notebook for the dataset you want to reproduce.
5. Run the notebook from top to bottom to regenerate the archived figures shown in each folder.

## Fresh VM Or Colab

- The bundle is designed to run in an isolated VM, Codespace, or Colab runtime without machine-specific paths.
- `python check_repro_environment.py ...` only validates the runtime and prints next steps; it does not install packages or rewrite configuration.
- `python datasets/unpack_reproduction_data.py --dataset worldvalue --worldvalue-layout minimal` restores only the WorldValuesBench pieces needed for the paper figures. Use `--worldvalue-layout full` only if you also want the upstream benchmark splits and auxiliary outputs.
- In the minimal WorldValue layout, almost all remaining `data/worldvaluesbench/` size comes from `F00011356-WVS_Cross-National_Wave_7_csv_v6_0.zip`, which is the raw WVS archive already kept compressed on disk.
- Fresh provider API calls in `WV_llmcalls.ipynb` are optional. Only install `requirements-optional-llm.txt` if you plan to rerun generation.

Each notebook auto-detects the reproduction root before loading inputs. The expected extracted directories are:

- `data/worldvalue/` and the minimal `data/worldvaluesbench/` subset for WorldValueBench.
- `data/eedi/` for EEDI.
- `data/opinionqa/` for OpinionQA.
