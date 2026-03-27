# EEDI and OpinionQA Quantile Reproduction

This folder reproduces the appendix EEDI and OpinionQA quantile-fidelity figures directly from the prepared artifacts in `data/eedi/` and `data/opinionqa/`.

## Contents

- `EEDI_OpinionQA_quantile_construction.ipynb`: loads the prepared artifacts, regenerates the paper's EEDI and OpinionQA quantile-fidelity plots, saves them to `figures/`, and displays the archived outputs in the notebook.
- `figures/`: archived PNG copies of the final EEDI and OpinionQA plots.

## How To Use It

1. Install the core runtime dependencies:
   `pip install -r ../requirements.txt`
2. Run a preflight check that does not modify the environment:
   `python ../check_repro_environment.py --dataset all`
3. Restore the bundle-local datasets:
   From the repository root:
   `python datasets/unpack_reproduction_data.py --dataset eedi`
   `python datasets/unpack_reproduction_data.py --dataset opinionqa`
4. Open `EEDI_OpinionQA_quantile_construction.ipynb`.
5. Run the notebook top to bottom to regenerate the two archived figures.

The notebook intentionally skips the upstream data-cleaning pipeline and starts from the prepared survey and simulator artifacts used for the final paper figures.
