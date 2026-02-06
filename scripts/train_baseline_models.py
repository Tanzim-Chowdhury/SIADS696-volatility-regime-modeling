#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import json

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import joblib


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--train-end", required=True, help="Inclusive train end date, e.g. 2022-12-31")
    p.add_argument("--test-start", required=True, help="Inclusive test start date, e.g. 2023-01-01")
    p.add_argument("--outdir", default="artifacts/baseline", help="Where to save models/metrics")
    return p.parse_args()


def time_split(df: pd.DataFrame, train_end: str, test_start: str):
    train_end = pd.to_datetime(train_end)
    test_start = pd.to_datetime(test_start)

    # Train: <= train_end
    train = df[df["date"] <= train_end].copy()

    # Test: >= test_start
    test = df[df["date"] >= test_start].copy()

    # Guard against overlap (can happen if test_start <= train_end)
    overlap = (df["date"] >= test_start) & (df["date"] <= train_end)
    if overlap.any():
        n_overlap = int(overlap.sum())
        raise ValueError(
            f"Time split overlap detected: test_start ({test_start.date()}) "
            f"<= train_end ({train_end.date()}). Overlapping rows: {n_overlap}"
        )

    return train, test


def safe_roc_auc(y_true, probas, labels):
    """
    Handles binary vs multiclass AUC in a consistent way.
    - probas: ndarray shape (n, n_classes)
    - labels: list/array of class labels in model order
    """
    n_classes = len(labels)
    if n_classes == 2:
        # Use positive class probability (second column in sklearn order)
        return roc_auc_score(y_true, probas[:, 1])
    else:
        return roc_auc_score(y_true, probas, multi_class="ovr")


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data)
    df["date"] = pd.to_datetime(df["date"])

    # ===== Feature selection =====
    feature_cols = [
        c for c in df.columns
        if (
            c.startswith("rv_")
            or c.startswith("vol_")
            or c.startswith("vix_")
            or c == "log_return"
        )
    ]

    target = "regime"

    # ===== Time split =====
    train, test = time_split(df, args.train_end, args.test_start)

    X_train = train[feature_cols]
    y_train = train[target]

    X_test = test[feature_cols]
    y_test = test[target]

    print("Train size:", len(X_train))
    print("Test size:", len(X_test))

    print("\nTrain regime distribution:")
    print(y_train.value_counts(dropna=False))
    print("\nTest regime distribution:")
    print(y_test.value_counts(dropna=False))

    metrics = {
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "feature_count": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "train_end": str(args.train_end),
        "test_start": str(args.test_start),
        "models": {},
    }

    # ===== Scaling (for Logistic Regression only) =====
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ===== Logistic Regression =====
    print("\n=== Logistic Regression ===")
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train_scaled, y_train)

    lr_preds = lr.predict(X_test_scaled)
    lr_probs = lr.predict_proba(X_test_scaled)
    lr_labels = list(lr.classes_)

    lr_f1 = f1_score(y_test, lr_preds, average="macro")
    lr_auc = safe_roc_auc(y_test, lr_probs, lr_labels)
    lr_cm = confusion_matrix(y_test, lr_preds)

    print("F1 (macro):", lr_f1)
    print("ROC AUC:", lr_auc)
    print(lr_cm)
    print(classification_report(y_test, lr_preds))

    metrics["models"]["logreg"] = {
        "f1_macro": float(lr_f1),
        "roc_auc": float(lr_auc),
        "classes": [str(x) for x in lr_labels],
        "confusion_matrix": lr_cm.tolist(),
        "classification_report": classification_report(y_test, lr_preds, output_dict=True),
    }

    # ===== Random Forest =====
    print("\n=== Random Forest ===")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",  # helpful if regimes are imbalanced
    )
    rf.fit(X_train, y_train)

    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)
    rf_labels = list(rf.classes_)

    rf_f1 = f1_score(y_test, rf_preds, average="macro")
    rf_auc = safe_roc_auc(y_test, rf_probs, rf_labels)
    rf_cm = confusion_matrix(y_test, rf_preds)

    print("F1 (macro):", rf_f1)
    print("ROC AUC:", rf_auc)
    print(rf_cm)
    print(classification_report(y_test, rf_preds))

    metrics["models"]["rf"] = {
        "f1_macro": float(rf_f1),
        "roc_auc": float(rf_auc),
        "classes": [str(x) for x in rf_labels],
        "confusion_matrix": rf_cm.tolist(),
        "classification_report": classification_report(y_test, rf_preds, output_dict=True),
    }

    # ===== Feature Importance =====
    print("\nTop Random Forest Features:")
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(importances.head(15))

    metrics["models"]["rf"]["top_features"] = importances.head(50).to_dict()

    # ===== Save artifacts =====
    joblib.dump(scaler, outdir / "scaler.joblib")
    joblib.dump(lr, outdir / "logreg.joblib")
    joblib.dump(rf, outdir / "rf.joblib")

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved artifacts to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
