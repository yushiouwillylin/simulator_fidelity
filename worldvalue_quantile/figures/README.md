# WorldValue Figure Archive

This directory stores archived PNG copies of the WorldValue paper figures and links them back to the notebook blocks that generate them.

| Paper Figure | Archived PNG | Notebook Block | Notes |
| --- | --- | --- | --- |
| Figure 3 | `worldvalue_calibrated_quantile_curve.png` | `Paper Figure 3 — Calibrated Quantile Curve` in `WV_quantile_construction.ipynb` | Main calibrated quantile curve across simulators. |
| Figure 4 | `worldvalue_robustness_all_models_n_grid.png` | `Paper Figure 4 — Robustness Across Human Sample Sizes` in `WV_quantile_construction.ipynb` | Four-panel robustness plot over `n in {50, 500, 5000, 10000}`. |
| Figure 5 | `worldvalue_tightness_adaptive_gamma_gpt4o.png` | `Paper Figures 5 and 6 — Tightness Analysis` in `WV_quantile_construction.ipynb` | Tightness analysis under adaptive `gamma_j`. |
| Figure 6 | `worldvalue_tightness_fixed_gamma_gpt4o.png` | `Paper Figures 5 and 6 — Tightness Analysis` in `WV_quantile_construction.ipynb` | Tightness analysis under fixed `gamma`. |
| Figure 7 | `worldvalue_band_panels_gpt4o_vs_llama_n1000_n5000.png` | `Paper Figure 7 — Confidence Bands for GPT-4o vs Llama 3.3` in `WV_quantile_construction.ipynb` | Two-panel confidence-band comparison. |

Supplementary archived plots in this folder include:

- `worldvalue_adaptive_band_gpt4o_n5000.png`
- `worldvalue_adaptive_band_gpt4o_vs_llama_n5000.png`
- `worldvalue_band_panels_gpt4o_n1000_n5000.png`
- `worldvalue_beta_sensitivity_gpt4o_n50_n200.png` for the schedule `\gamma_j = 1 - n_j^{-\beta}` with `\beta \in {1/5, 1/4, 1/3, 1/2}`
