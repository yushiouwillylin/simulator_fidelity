# UMAR Quantile Reproduction

This folder is a self-contained UMAR instantiation of the paper's quantile-fidelity method. Instead of survey questions, the units here are coarse building-operation scenarios, and instead of survey response rates, the parameter of interest is scenario-level temperature variability.

## How This Maps To The Overall Method

The same paper logic is used here, just with a different domain:

- scenario unit: `psi = h={hour}|w={is_weekend}|tq={temp_q}`
- real-side parameter `p_j`: variance of the future 30-minute room temperature within scenario `psi_j`
- simulator-side parameter `q_j`: variance induced by emulator draws within the same scenario
- uncertainty coverage: scenario-specific `gamma_j = 1 - n_j^{-1/3}`
- discrepancy target: pseudo-discrepancies between the simulator variance and a confidence interval for the real-side variance
- final summary object: the adjusted upper quantile curve built from the scenario-level discrepancies

So the UMAR experiment should be read as "apply the calibrated quantile-curve framework to a short-horizon thermal emulator," not as "pick the best predictor by RMSE alone."

Relative to the paper text, UMAR is best viewed as an instance of the generic continuous/bounded-functional version of the framework: the latent parameter is a scalar variance functional rather than a multinomial mean, and the per-scenario confidence set is implemented with a bootstrap interval rather than the paper's survey-specific multinomial example.

## What This Folder Reproduces

- a cleaned 30-minute UMAR causal table for one room
- cross-fitted simulator outputs for a small, readable set of ML emulator families
- scenario-level variance discrepancy tables
- the main adjusted upper quantile curve used for the UMAR reproduction figure

## Folder Contents

- `umar_ml_simulator_construction.ipynb`: builds the cleaned 30-minute table, cross-fits the emulator families, samples simulator-side draws, and writes the core artifacts.
- `umar_quantile_inference.ipynb`: consumes those saved artifacts and reproduces the calibrated scenario-level variance curve.
- `umar_utils.py`: local helper module for confidence bounds, pseudo-discrepancies, quantile curves, and the default emulator definitions.
- `figures/`: archived PNG copies of the UMAR figures. See `figures/README.md`.
- `output/`: generated CSV/parquet artifacts written by the notebooks. See `output/README.md`.

## Fixed Experiment Used Here

This folder intentionally fixes one concrete UMAR run so the reproduction is easy to rerun and interpret:

- target room temperature: `temp_275`
- ambient temperature: `temp_amb`
- optional setpoint: `setp_275`
- optional irradiance: `irrad`
- resampling rule: `30min`
- target variable: next-step 30-minute mean room temperature `y_future`
- artifact filter: remove rows where `room_temp_t`, `lag1`, `lag2`, or `y_future` exceeds `30C`
- simulator budget: `k_j = 200` draws per scenario
- model families: linear regression, a shallow decision tree, and a multilayer perceptron emulator

These choices are meant to keep the folder aligned with the paper's fidelity-analysis goal: compare standard simulator families on calibrated scenario-level behavior, not maximize task-specific predictive performance with heavy tuning.

## Data Layout

All required UMAR inputs live inside this reproduction repo:

- `../data/umar/raw/`
- `../data/umar/metadata/umar_metadata.csv`

See `../data/umar/README.md` for the expected files.

## How To Run It

1. Install the core runtime dependencies:
   `pip install -r ../requirements.txt`
2. Run a preflight check:
   `python ../check_repro_environment.py --dataset umar`
3. Run `umar_ml_simulator_construction.ipynb`.
4. Run `umar_quantile_inference.ipynb`.

`pyarrow` is part of the core requirements because this workflow writes and reads parquet artifacts under `output/`.

## Main Figure

The main archived UMAR figure is:

- `figures/umar_main_variance_adjusted_upper_curves.png`

It reports the adjusted upper quantile curves for the scenario-level variance discrepancy after the fixed artifact filter is applied.
