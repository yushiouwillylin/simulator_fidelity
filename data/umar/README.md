# UMAR Data Layout

The UMAR reproduction expects all of its local inputs under this directory.

Required files:

- `metadata/umar_metadata.csv`
- `raw/umar_2019-07-01_2020-07-01_wide_descId.csv`
- `raw/umar_2020-07-01_2021-07-01_wide_descId.csv`
- `raw/umar_2021-07-01_2022-07-01_wide_descId.csv`
- `raw/umar_2022-07-01_2023-07-01_wide_descId.csv`

These files are consumed by `umar_quantile/umar_ml_simulator_construction.ipynb`, which aggregates them into the cleaned 30-minute causal table and the downstream simulator artifacts used by the UMAR quantile analysis.
