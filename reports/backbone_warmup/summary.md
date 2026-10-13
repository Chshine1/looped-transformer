# Backbone warmup checkpoint diagnostics

Dropout off; all six depths collected through the production forward. FP32, no autocast. Depths are one-based. Oracle metrics use labels and are optimistic bounds, not deployable policies.

| Epoch | Gamma | CE by depth | Accuracy by depth (%) | Any-depth correct (%) |
| --- | --- | --- | --- | --- |
| 1 | 0.5756 | 0.9753, 0.9678, 0.9606, 0.9574, 0.9574, 0.9533 | 51.04, 52.94, 53.74, 53.95, 53.77, 53.89 | 64.21 |
| 2 | 0.4730 | 0.9296, 0.8839, 0.8782, 0.8773, 0.8770, 0.8775 | 55.63, 58.96, 59.22, 59.31, 59.46, 59.15 | 67.68 |
| 3 | 0.4580 | 0.8947, 0.8499, 0.8471, 0.8485, 0.8489, 0.8480 | 57.89, 61.47, 61.60, 61.61, 61.56, 61.45 | 69.04 |
| 4 | 0.4604 | 0.8833, 0.8387, 0.8367, 0.8366, 0.8365, 0.8359 | 59.30, 62.12, 62.23, 62.36, 62.36, 62.40 | 69.51 |

Full typed measurements, sample indices, source hashes, gradient matrices, probes and intervention results are in report.json.

Adjacent change statistics use the same examples at every depth. Improved/worsened means a CE change exceeding 0.01. Relative representation changes are per-example norms, then averaged; classifier-visible energy is a ratio of summed squared norms in normalized pooled space. Gradient cancellation is norm(mean gradient) / mean(norm gradient), measured on the recorded fixed subset with dropout off. Interventions are inference-time distribution shifts and do not establish what retraining would achieve. Probe optimization uses a fixed budget without validation tuning; independently fitted heads have more parameters.
