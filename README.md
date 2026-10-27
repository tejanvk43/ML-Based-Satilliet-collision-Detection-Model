# LEO Satellite Collision Risk Prediction

A B.Tech CSE minor project that explores machine-learning methods for prioritizing satellite conjunction events using data associated with the ESA Kelvins Collision Avoidance Challenge.

## Project Overview

As the number of satellites in low Earth orbit increases, operators must review predicted close approaches, known as conjunctions. This project builds event-level features from conjunction-message history and compares a simple last-risk baseline with machine-learning models at several prediction lead times.

**Project scope:** This is an academic minor project and an experimental risk-ranking workflow. It is not an operational collision-avoidance system and its metrics are not an official ESA competition score.

## Dataset

- **Dataset archive:** `Collision_Avoidance_Challenge_Dataset.zip`
- **Raw data member:** `raw_data_2015-2019.gz`
- **Archive URL:** https://drive.google.com/file/d/1MESuFeOgnZ9tfX64IZZNttfgJCZr_qbm/view
- **Reported raw dimensions:** 199,082 rows and 103 columns
- **Reported common cohort:** 8,300 events across five lead times, yielding 41,500 event/lead-time rows
- **Reported missions:** 19
- **Lead times:** 5, 4, 3, 2, and 1 day before the time of closest approach (TCA)

The statistics above are transcribed from the project notes and supplied notebook output. Verify them against the final `cohort_report.csv` before using them as definitive submission figures.

See [`dataset_source.txt`](dataset_source.txt) for dataset fields, preprocessing notes, and the target-definition caution.

## Methodology

1. Load the compressed dataset from the downloaded archive.
2. Clean invalid numeric values by replacing infinities and extreme numeric magnitudes with missing values.
3. Build event-level features using only conjunction messages available at each prediction lead-time cutoff.
4. Aggregate history-derived features such as message count and risk statistics, where available.
5. Keep `mission_id` for grouping and validation, rather than using it as a predictive feature.
6. Evaluate the last-risk baseline and machine-learning models using out-of-fold predictions and mission-level bootstrap analysis where available.

The exact positive-label rule should be documented from the final source code. Project notes mention `HIGH_RISK = -6.0` and `FINAL_TCA_MAX = 1.0`; do not infer the full label rule from these constants alone.

## Reported Results

The supplied Paper 5 output reports these pooled out-of-fold estimates:

| Model | PR-AUC | Precision | Recall | F2 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Last-risk baseline | 0.4008 | 0.3476 | 0.8079 | 0.6387 | 229 | 29 |
| Hist Gradient Boosting | 0.5007 | 0.4413 | 0.7219 | 0.6404 | 138 | 42 |
| LightGBM | 0.4923 | 0.3459 | 0.7881 | 0.6276 | 225 | 32 |

### Interpretation

- Hist Gradient Boosting has the highest reported PR-AUC among the three models in this experiment.
- Its paired PR-AUC difference versus the last-risk baseline is reported as `+0.0999`, with a 95% mission-bootstrap interval of `[0.0111, 0.1697]`.
- Its F2 difference versus the baseline is only `+0.0017`, with a 95% interval of `[-0.0254, 0.0355]`; this does not establish a clear F2 improvement.
- Hist Gradient Boosting has lower recall than the baseline at the reported threshold and has 42 false negatives compared with 29 for the baseline.
- LightGBM's reported PR-AUC difference versus the baseline has a 95% interval of `[-0.0021, 0.1717]`, which includes zero.

These results apply to the reported evaluation setup and threshold. They should not be presented as official ESA competition scores or as proof of real-world collision-avoidance capability.

## Result Files

The existing MoLab output folders may contain the following relevant CSV files:

- `cohort_report.csv` — cohort and dataset summary.
- `model_comparison.csv` — primary model comparison.
- `mission_holdout_pooled_metrics.csv` — pooled mission-held-out metrics.
- `mission_holdout_fold_metrics.csv` — metrics by held-out fold.
- `mission_holdout_predictions.csv` — mission-held-out predictions.
- `event_level_bootstrap_comparison.csv` — event-level bootstrap comparison.
- `paper3_model_comparison.csv` — Paper 3 model comparison.
- `paper3_fold_stability_summary.csv` — fold stability summary.
- `paper4_pooled_summary.csv` — Paper 4 pooled summary.
- `paper5_bootstrap_confidence_intervals.csv` — bootstrap confidence intervals.
- `paper5_paired_model_comparisons.csv` — paired model comparisons.
- `paper5_point_estimates.csv` — point estimates.

These files come from different experimental stages. Do not combine their metrics in one table unless the datasets, splits, target definition, and evaluation procedures are compatible.

## Execution Evidence

- `run_log.txt` contains a formatted transcription of the Paper 5 output supplied during project preparation; it is not a direct terminal capture.
- Save screenshots from the actual MoLab run showing the completed cell and result tables.
- Keep the original CSV outputs alongside screenshots and documentation.

## Reproducibility

Record the following for the final submitted run:

- Python version and package versions.
- Final source-code filename and run date.
- Dataset source and exact cohort size.
- Target/label definition and decision threshold.
- Validation strategy and random seed, if applicable.
- Exact commands or notebook steps required to reproduce the outputs.

## Limitations

- The positive class is rare, so PR-AUC and recall should be interpreted alongside precision and false-negative counts.
- Performance varies across missions; bootstrap intervals describe uncertainty under the reported resampling scheme.
- A model with higher PR-AUC does not necessarily have the best operational trade-off at a chosen threshold.
- Feature availability and leakage audits should be reviewed before claiming that all predictors are operationally available at the prediction cutoff.
- Results are experimental and are not a substitute for orbital-dynamics analysis or expert operational decisions.

## Suggested Submission Structure

```text
LEO_Collision_Risk_Minor_Project/
├── README.md
├── dataset_source.txt
├── run_log.txt
├── source_code/
├── results/
│   ├── model_comparison.csv
│   ├── mission_holdout_pooled_metrics.csv
│   ├── paper5_bootstrap_confidence_intervals.csv
│   └── project_graphs/
└── evidence/
    ├── execution_complete.png
    └── paper5_results.png
```

Only include files that actually exist. Do not create placeholder CSVs or screenshots.

## Credits and Attribution

Credit the ESA Kelvins Collision Avoidance Challenge and preserve the dataset's original attribution and license/terms when preparing the final submission. Confirm the exact terms from the dataset's source before redistributing the raw data.
