"""
Machine Learning-Based Spacecraft Collision Risk Classification
Reproducible minor-project pipeline.

What this script does:
1. Downloads the ESA Kelvins Collision Avoidance Challenge archive from Google Drive.
2. Reads raw_data_2015-2019.gz from the ZIP.
3. Builds event-level features available at 5, 4, 3, 2, and 1 days before TCA.
4. Evaluates models with 5-fold GroupKFold by mission (unseen missions in test folds).
5. Saves fold metrics, out-of-fold predictions, pooled metrics, bootstrap intervals,
   a lead-time plot, and a text summary.

Important:
- This is a research/educational classifier, not an operational collision-avoidance system.
- It does not reproduce the official ESA competition scoring protocol.
- Mission-held-out results may differ from prior random/repeated stratified CV results.
"""

from __future__ import annotations

import gzip
import json
import warnings
import zipfile
from pathlib import Path

import gdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    fbeta_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("default")

# -----------------------------
# Configuration
# -----------------------------
DRIVE_FILE_ID = "1MESuFeOgnZ9tfX64IZZNttfgJCZr_qbm"
ZIP_NAME = "Collision_Avoidance_Challenge_Dataset.zip"
RAW_MEMBER_SUFFIX = "raw_data_2015-2019.gz"

LEAD_DAYS = [5.0, 4.0, 3.0, 2.0, 1.0]
HIGH_RISK_THRESHOLD = -6.0
FINAL_TCA_MAX = 1.0
N_GROUP_SPLITS = 5
RANDOM_STATE = 42
BOOTSTRAP_REPETITIONS = 1000
OUTPUT_DIR = Path("outputs/final_minor_project")
DATA_DIR = Path("data")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = DATA_DIR / ZIP_NAME


def download_dataset() -> Path:
    """Download the public dataset archive if a usable local copy is absent."""
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 200_000_000:
        print(f"Using existing archive: {ZIP_PATH.resolve()}")
    else:
        print("Downloading ESA Kelvins Collision Avoidance Challenge dataset...")
        result = gdown.download(
            url=f"https://drive.google.com/uc?id={DRIVE_FILE_ID}",
            output=str(ZIP_PATH),
            quiet=False,
            fuzzy=True,
        )
        if result is None or not ZIP_PATH.exists():
            raise RuntimeError(
                "Dataset download failed. Check the Google Drive file access "
                "and your internet connection, then retry."
            )

    if not zipfile.is_zipfile(ZIP_PATH):
        raise ValueError(
            f"{ZIP_PATH} is not a valid ZIP archive. Delete it and retry download."
        )

    print(
        f"Archive ready: {ZIP_PATH.resolve()} "
        f"({ZIP_PATH.stat().st_size / 1024**2:.1f} MiB)"
    )
    return ZIP_PATH


def load_raw_data(archive_path: Path) -> pd.DataFrame:
    """Read the historical compressed CSV directly from the ZIP archive."""
    with zipfile.ZipFile(archive_path, "r") as archive:
        members = [
            member for member in archive.namelist()
            if member.lower().endswith(RAW_MEMBER_SUFFIX)
        ]
        if not members:
            raise FileNotFoundError(
                f"Could not find {RAW_MEMBER_SUFFIX} inside {archive_path}."
            )

        member_name = members[0]
        print(f"Reading archive member: {member_name}")
        with archive.open(member_name) as compressed_member:
            with gzip.open(compressed_member, "rt", encoding="utf-8-sig") as stream:
                raw = pd.read_csv(stream, sep=None, engine="python")

    raw.columns = raw.columns.astype(str).str.strip()
    required = {"event_id", "time_to_tca", "risk"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"Required columns are missing: {sorted(missing)}. "
            "Inspect the dataset schema before proceeding."
        )

    print(f"Raw data shape: {raw.shape}")
    print(f"Columns: {raw.shape[1]}")
    print(f"Unique events: {raw['event_id'].nunique():,}")
    return raw


def clean_raw_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index, pd.Series, pd.Series]:
    """Clean required columns and derive each event's final target."""
    data = raw.copy()
    for col in ["time_to_tca", "risk"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["event_id", "time_to_tca", "risk"]).copy()

    # Larger time_to_tca means an earlier CDM. The smallest time_to_tca
    # is the final/closest-to-TCA observation for each event.
    data = data.sort_values(
        ["event_id", "time_to_tca"],
        ascending=[True, False],
    ).copy()

    final_cdm = data.groupby("event_id", sort=False).tail(1).set_index("event_id")
    eligible_final = final_cdm.index[
        final_cdm["time_to_tca"] < FINAL_TCA_MAX
    ]
    final_risk = final_cdm["risk"]
    max_time = data.groupby("event_id")["time_to_tca"].max()

    print(f"Valid CDM rows after cleaning: {len(data):,}")
    print(f"Events after cleaning: {data['event_id'].nunique():,}")
    print(f"Events with final observation under {FINAL_TCA_MAX} day: {len(eligible_final):,}")
    return data, final_cdm, pd.Index(eligible_final), max_time, final_risk


def build_event_dataset(
    data: pd.DataFrame,
    eligible_final: pd.Index,
    max_time: pd.Series,
    final_risk: pd.Series,
    lead_days: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build cutoff-specific event features and labels for one lead time."""
    # Use the common cohort: every event must have enough history to support
    # the largest requested lead time, allowing fair comparisons across leads.
    common_min_lead = max(LEAD_DAYS)
    eligible = eligible_final.intersection(
        max_time.index[max_time >= common_min_lead]
    )

    history = data[
        data["event_id"].isin(eligible)
        & (data["time_to_tca"] >= lead_days)
    ].copy()

    grouped = history.groupby("event_id", sort=False)
    latest = grouped.tail(1).set_index("event_id")
    earliest = grouped.head(1).set_index("event_id")

    aggregates = grouped.agg(
        n_cdms=("time_to_tca", "size"),
        risk_max=("risk", "max"),
        risk_min=("risk", "min"),
        risk_mean=("risk", "mean"),
    )

    # Keep mission IDs only as group metadata, never as model predictors.
    if "mission_id" not in latest.columns:
        raise KeyError(
            "The raw dataset has no 'mission_id' column. Mission-held-out "
            "validation requires a documented mission mapping; do not substitute "
            "event IDs or random folds."
        )

    groups = latest["mission_id"].astype(str).rename("mission_id")

    features = latest.rename(
        columns={
            "risk": "last_risk",
            "time_to_tca": "tca_of_last_cdm",
        }
    ).join(aggregates)

    features["risk_delta"] = latest["risk"] - earliest["risk"]
    features["tca_span"] = (
        earliest["time_to_tca"] - latest["time_to_tca"]
    )

    if "miss_distance" in latest.columns:
        latest_md = pd.to_numeric(latest["miss_distance"], errors="coerce")
        earliest_md = pd.to_numeric(earliest["miss_distance"], errors="coerce")
        features["miss_distance_delta"] = latest_md - earliest_md

    # Preserve event IDs as index; remove all raw identifier columns from X.
    features = features.drop(
        columns=["event_id", "mission_id"],
        errors="ignore",
    )

    categorical_cols = list(
        features.select_dtypes(exclude="number").columns
    )
    if categorical_cols:
        features = pd.get_dummies(
            features,
            columns=categorical_cols,
            dummy_na=True,
            dtype=float,
        )

    features = features.replace([np.inf, -np.inf], np.nan)
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        features.loc[features[col].abs() > 1e30, col] = np.nan

    features = features.dropna(axis=1, how="all")
    features = features.loc[:, features.nunique(dropna=True) > 1]

    # Align target labels to the exact feature rows.
    labels = (
        final_risk.reindex(features.index) >= HIGH_RISK_THRESHOLD
    ).astype(int).to_numpy()

    group_ids = groups.reindex(features.index).astype(str).to_numpy()

    if len(features) != len(labels) or len(labels) != len(group_ids):
        raise AssertionError("Feature, label, and mission lengths do not match.")
    if pd.isna(group_ids).any():
        raise ValueError("Missing mission IDs found.")

    return features, labels, group_ids


def build_models(seed: int) -> dict:
    """Return classifiers with preprocessing fitted only on training folds."""
    models = {
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        "Extra Trees": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            ExtraTreesClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        "Hist Gradient Boosting": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            HistGradientBoostingClassifier(
                max_iter=180,
                learning_rate=0.08,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=seed,
            ),
        ),
    }

    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            LGBMClassifier(
                n_estimators=250,
                learning_rate=0.04,
                num_leaves=15,
                class_weight="balanced",
                verbosity=-1,
                random_state=seed,
                n_jobs=-1,
            ),
        )
    except ImportError:
        print("LightGBM is not installed; continuing with sklearn models.")

    return models


def main() -> None:
    archive = download_dataset()
    raw = load_raw_data(archive)
    data, final_cdm, eligible_final, max_time, final_risk = clean_raw_data(raw)

    # Save a compact dataset audit.
    audit = {
        "raw_rows": int(len(raw)),
        "raw_columns": int(raw.shape[1]),
        "clean_cdm_rows": int(len(data)),
        "unique_events": int(data["event_id"].nunique()),
        "eligible_final_events": int(len(eligible_final)),
        "lead_days": LEAD_DAYS,
        "high_risk_log10_threshold": HIGH_RISK_THRESHOLD,
        "final_tca_max_exclusive_days": FINAL_TCA_MAX,
        "validation": f"{N_GROUP_SPLITS}-fold GroupKFold by mission",
        "random_state": RANDOM_STATE,
    }

    dataset_tables = {}
    cohort_records = []

    for lead in LEAD_DAYS:
        print(f"\nBuilding features for {lead:g}-day lead...")
        X, y, groups = build_event_dataset(
            data, eligible_final, max_time, final_risk, lead
        )
        dataset_tables[lead] = (X, y, groups)
        cohort_records.append({
            "lead_days": lead,
            "events": len(y),
            "missions": len(np.unique(groups)),
            "positive_events": int(y.sum()),
            "positive_percent": round(100 * float(y.mean()), 3),
            "predictor_count": int(X.shape[1]),
        })

    cohort_df = pd.DataFrame(cohort_records)
    cohort_df.to_csv(OUTPUT_DIR / "cohort_report.csv", index=False)
    with open(OUTPUT_DIR / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print("\n=== COHORT REPORT ===")
    print(cohort_df.to_string(index=False))

    fold_records = []
    prediction_records = []

    for lead, (X, y, groups) in dataset_tables.items():
        unique_missions = np.unique(groups)
        if len(unique_missions) < N_GROUP_SPLITS:
            raise ValueError(
                f"Only {len(unique_missions)} missions at {lead} days; "
                f"cannot use {N_GROUP_SPLITS}-fold GroupKFold."
            )

        if len(np.unique(y)) < 2:
            raise ValueError(f"Only one target class at {lead} days.")

        print(f"\n=== MISSION-HELD-OUT EVALUATION: {lead:g} DAYS ===")
        cv = GroupKFold(n_splits=N_GROUP_SPLITS)

        for fold, (train_idx, test_idx) in enumerate(
            cv.split(X, y, groups=groups), start=1
        ):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            if train_groups & test_groups:
                raise AssertionError("Mission leakage detected.")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if len(np.unique(y_train)) < 2:
                raise ValueError(
                    f"Training fold {fold} at lead {lead} contains one class."
                )

            # Baseline ranking score: observed log-risk at the cutoff.
            baseline_scores = pd.to_numeric(
                X_test["last_risk"], errors="coerce"
            ).to_numpy(dtype=float)
            train_risk = pd.to_numeric(
                X_train["last_risk"], errors="coerce"
            ).to_numpy(dtype=float)
            finite_train_risk = train_risk[np.isfinite(train_risk)]
            fill_value = (
                float(np.median(finite_train_risk))
                if len(finite_train_risk) else -100.0
            )
            baseline_scores = np.where(
                np.isfinite(baseline_scores), baseline_scores, fill_value
            )

            fold_scores = {"Last-risk baseline": baseline_scores}
            model_factory = build_models(RANDOM_STATE + int(lead * 100) + fold)

            for model_name, model in model_factory.items():
                model.fit(X_train, y_train)
                fold_scores[model_name] = model.predict_proba(X_test)[:, 1]

            for model_name, scores in fold_scores.items():
                valid_pr = int(y_test.sum()) > 0
                valid_roc = len(np.unique(y_test)) == 2

                pr_auc = (
                    float(average_precision_score(y_test, scores))
                    if valid_pr else np.nan
                )
                roc_auc = (
                    float(roc_auc_score(y_test, scores))
                    if valid_roc else np.nan
                )

                fold_records.append({
                    "lead_days": lead,
                    "fold": fold,
                    "model": model_name,
                    "train_events": len(train_idx),
                    "test_events": len(test_idx),
                    "train_missions": len(train_groups),
                    "test_missions": len(test_groups),
                    "test_positives": int(y_test.sum()),
                    "PR_AUC": pr_auc,
                    "ROC_AUC": roc_auc,
                })

                for position, row_index in enumerate(test_idx):
                    prediction_records.append({
                        "lead_days": lead,
                        "fold": fold,
                        "event_id": str(X.index[row_index]),
                        "mission_id": str(groups[row_index]),
                        "y_true": int(y[row_index]),
                        "model": model_name,
                        "score": float(scores[position]),
                    })

            print(
                f"Fold {fold}/{N_GROUP_SPLITS}: "
                f"{len(test_idx)} test events, "
                f"{len(test_groups)} held-out missions"
            )

    folds_df = pd.DataFrame(fold_records)
    predictions_df = pd.DataFrame(prediction_records)

    folds_df.to_csv(OUTPUT_DIR / "mission_holdout_fold_metrics.csv", index=False)
    predictions_df.to_csv(OUTPUT_DIR / "mission_holdout_predictions.csv", index=False)

    # Summary of fold-level means and standard deviations.
    summary_df = (
        folds_df.groupby(["lead_days", "model"])
        .agg(
            mean_PR_AUC=("PR_AUC", "mean"),
            std_PR_AUC=("PR_AUC", "std"),
            mean_ROC_AUC=("ROC_AUC", "mean"),
            std_ROC_AUC=("ROC_AUC", "std"),
            folds_with_defined_PR_AUC=("PR_AUC", "count"),
        )
        .reset_index()
    )
    summary_df.to_csv(OUTPUT_DIR / "mission_holdout_summary.csv", index=False)

    # Pooled out-of-fold scores for each lead/model.
    pooled_records = []
    for (lead, model_name), subset in predictions_df.groupby(["lead_days", "model"]):
        y_true = subset["y_true"].to_numpy(dtype=int)
        scores = subset["score"].to_numpy(dtype=float)
        pooled_records.append({
            "lead_days": float(lead),
            "model": model_name,
            "events": len(subset),
            "missions": subset["mission_id"].nunique(),
            "positives": int(y_true.sum()),
            "positive_rate": float(y_true.mean()),
            "pooled_PR_AUC": (
                float(average_precision_score(y_true, scores))
                if np.unique(y_true).size == 2 else np.nan
            ),
            "pooled_ROC_AUC": (
                float(roc_auc_score(y_true, scores))
                if np.unique(y_true).size == 2 else np.nan
            ),
        })

    pooled_df = pd.DataFrame(pooled_records)
    pooled_df.to_csv(OUTPUT_DIR / "mission_holdout_pooled_metrics.csv", index=False)

    # Mission-cluster bootstrap of pooled OOF PR-AUC differences.
    # These intervals are conditional on the saved fold predictions.
    bootstrap_rng = np.random.default_rng(RANDOM_STATE + 1000)
    bootstrap_records = []

    for lead, lead_frame in predictions_df.groupby("lead_days"):
        event_labels = (
            lead_frame.drop_duplicates("event_id")
            .set_index("event_id")["y_true"]
        )
        event_missions = (
            lead_frame.drop_duplicates("event_id")
            .set_index("event_id")["mission_id"]
        )
        event_ids = event_labels.index.to_numpy()
        mission_values = event_missions.to_numpy()
        unique_missions = np.unique(mission_values)

        model_scores = {}
        for model_name, model_frame in lead_frame.groupby("model"):
            model_scores[model_name] = (
                model_frame.set_index("event_id")["score"]
                .reindex(event_ids)
                .to_numpy(dtype=float)
            )

        base_name = "Last-risk baseline"
        if base_name not in model_scores:
            continue

        valid_boot = {model_name: [] for model_name in model_scores}

        mission_to_indices = {
            mission: np.flatnonzero(mission_values == mission)
            for mission in unique_missions
        }

        for _ in range(BOOTSTRAP_REPETITIONS):
            sampled_missions = bootstrap_rng.choice(
                unique_missions, size=len(unique_missions), replace=True
            )
            sampled_indices = []
            for mission in sampled_missions:
                sampled_indices.extend(mission_to_indices[mission].tolist())
            sampled_indices = np.asarray(sampled_indices, dtype=int)

            y_boot = event_labels.to_numpy(dtype=int)[sampled_indices]
            if np.unique(y_boot).size < 2:
                continue

            for model_name, scores in model_scores.items():
                valid_boot[model_name].append(
                    average_precision_score(y_boot, scores[sampled_indices])
                )

        base_point = pooled_df.loc[
            (pooled_df["lead_days"] == lead)
            & (pooled_df["model"] == base_name),
            "pooled_PR_AUC"
        ]
        if base_point.empty:
            continue
        base_point_value = float(base_point.iloc[0])
        base_boot_values = np.asarray(valid_boot[base_name], dtype=float)

        for model_name in model_scores:
            point_row = pooled_df.loc[
                (pooled_df["lead_days"] == lead)
                & (pooled_df["model"] == model_name)
            ]
            if point_row.empty:
                continue

            model_boot_values = np.asarray(valid_boot[model_name], dtype=float)
            model_point_value = float(point_row["pooled_PR_AUC"].iloc[0])

            if len(model_boot_values):
                ci_low, ci_high = np.percentile(
                    model_boot_values, [2.5, 97.5]
                )
            else:
                ci_low, ci_high = np.nan, np.nan

            if model_name == base_name:
                gain_low, gain_high, gain_point = 0.0, 0.0, 0.0
            elif len(model_boot_values) and len(base_boot_values):
                paired_length = min(len(model_boot_values), len(base_boot_values))
                # Each model's bootstrap scores are generated from the same
                # sampled mission sequence and therefore are paired by iteration.
                gains = (
                    model_boot_values[:paired_length]
                    - base_boot_values[:paired_length]
                )
                gain_low, gain_high = np.percentile(gains, [2.5, 97.5])
                gain_point = model_point_value - base_point_value
            else:
                gain_low, gain_high = np.nan, np.nan
                gain_point = model_point_value - base_point_value

            bootstrap_records.append({
                "lead_days": float(lead),
                "model": model_name,
                "pooled_PR_AUC": model_point_value,
                "PR_AUC_CI95_low": ci_low,
                "PR_AUC_CI95_high": ci_high,
                "PR_AUC_gain_vs_baseline": gain_point,
                "gain_CI95_low": gain_low,
                "gain_CI95_high": gain_high,
                "gain_CI_excludes_zero_positive": bool(
                    np.isfinite(gain_low) and gain_low > 0
                ),
                "valid_bootstrap_repetitions": len(model_boot_values),
            })

    bootstrap_df = pd.DataFrame(bootstrap_records)
    bootstrap_df.to_csv(
        OUTPUT_DIR / "mission_bootstrap_pr_auc_intervals.csv", index=False
    )

    # Plot pooled PR-AUC by lead time.
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, model_frame in pooled_df.groupby("model"):
        model_frame = model_frame.sort_values("lead_days")
        ax.plot(
            model_frame["lead_days"],
            model_frame["pooled_PR_AUC"],
            marker="o",
            label=model_name,
        )
    ax.invert_xaxis()
    ax.set_xlabel("Lead time before closest approach (days)")
    ax.set_ylabel("Pooled PR-AUC (higher is better)")
    ax.set_title("Mission-Held-Out Collision Risk Classification")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "lead_time_pooled_pr_auc.png", dpi=200)
    plt.close(fig)

    # Create a plain-text summary for project documentation.
    best_by_lead = (
        pooled_df.sort_values(
            ["lead_days", "pooled_PR_AUC"],
            ascending=[False, False],
        )
        .groupby("lead_days", as_index=False)
        .first()
    )

    summary_lines = [
        "SPACECRAFT COLLISION RISK CLASSIFICATION — RUN SUMMARY",
        "=" * 62,
        f"Dataset archive: {ZIP_PATH.resolve()}",
        f"Raw rows: {len(raw):,}",
        f"Raw columns: {raw.shape[1]}",
        f"Clean CDM rows: {len(data):,}",
        f"Unique cleaned events: {data['event_id'].nunique():,}",
        f"High-risk threshold: final log-risk >= {HIGH_RISK_THRESHOLD}",
        f"Final observation criterion: time_to_tca < {FINAL_TCA_MAX} day",
        f"Validation: {N_GROUP_SPLITS}-fold GroupKFold by mission",
        "",
        "Best pooled PR-AUC model per lead time (descriptive):",
        best_by_lead.to_string(index=False),
        "",
        "Interpretation boundary:",
        "- Results estimate generalization to held-out missions in this dataset.",
        "- PR-AUC is the primary ranking metric because the target is imbalanced.",
        "- Bootstrap intervals are conditional on saved out-of-fold predictions.",
        "- This project does not reproduce the official ESA challenge score.",
        "- The classifier is a research prototype, not an operational alert system.",
    ]
    (OUTPUT_DIR / "project_run_summary.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )

    print("\n=== MISSION-HELD-OUT POOLED RESULTS ===")
    print(
        pooled_df.sort_values(
            ["lead_days", "pooled_PR_AUC"],
            ascending=[False, False],
        ).round(4).to_string(index=False)
    )

    print("\n=== FOLD-LEVEL SUMMARY ===")
    print(summary_df.round(4).to_string(index=False))

    print("\n=== BOOTSTRAP PR-AUC INTERVALS ===")
    if not bootstrap_df.empty:
        print(bootstrap_df.round(4).to_string(index=False))
    else:
        print("No bootstrap results were generated.")

    print(f"\nAll outputs saved under: {OUTPUT_DIR.resolve()}")
    print("This run is complete.")


if __name__ == "__main__":
    main()
