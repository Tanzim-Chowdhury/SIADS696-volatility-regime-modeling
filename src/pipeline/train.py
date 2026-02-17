from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from itertools import product
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SplitSpec:
    train_end: str
    val_start: str
    val_end: str
    test_start: str


def time_split_3way(df: pd.DataFrame, split: SplitSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None)

    train_end_dt = pd.to_datetime(split.train_end)
    val_start_dt = pd.to_datetime(split.val_start)
    val_end_dt = pd.to_datetime(split.val_end)
    test_start_dt = pd.to_datetime(split.test_start)

    train = d[d["date"] <= train_end_dt].copy()
    val = d[(d["date"] >= val_start_dt) & (d["date"] <= val_end_dt)].copy()
    test = d[d["date"] >= test_start_dt].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError("Train/val/test split has an empty partition. Check split dates and dataset range.")

    # Safety: enforce ordering, no overlap
    if not (train["date"].max() < val["date"].min() <= val["date"].max() < test["date"].min()):
        raise ValueError(
            "Split ordering/overlap issue. Ensure: train_end < val_start <= val_end < test_start "
            "and that dataset has dates in each range."
        )

    return train, val, test


def select_numeric_features(df: pd.DataFrame, target: str, leakage_prefixes: tuple[str, ...] = ("fwd_rv_",)) -> list[str]:
    blacklist = {"date", "ticker", target}
    cols = []
    for c in df.columns:
        if c in blacklist:
            continue
        if any(c.startswith(p) for p in leakage_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    if not cols:
        raise ValueError("No numeric features found.")
    return cols


def _make_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    prep = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([("num", prep, feature_cols)], remainder="drop")


def _eval(model, X: pd.DataFrame, y: pd.Series) -> dict:
    pred = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "classification_report": classification_report(y, pred, output_dict=True),
    }


def train_baselines_3way(df: pd.DataFrame, target: str, split: SplitSpec) -> dict:
    train_df, val_df, test_df = time_split_3way(df, split)
    features = select_numeric_features(df, target=target)

    X_train, y_train = train_df[features], train_df[target].astype(int)
    X_val, y_val = val_df[features], val_df[target].astype(int)
    X_test, y_test = test_df[features], test_df[target].astype(int)

    preprocessor = _make_preprocessor(features)

    logreg = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    rf = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )),
    ])

    out = {
        "split": {
            "train": {"start": str(train_df["date"].min().date()), "end": str(train_df["date"].max().date()), "n": int(len(train_df))},
            "val":   {"start": str(val_df["date"].min().date()),   "end": str(val_df["date"].max().date()),   "n": int(len(val_df))},
            "test":  {"start": str(test_df["date"].min().date()),  "end": str(test_df["date"].max().date()),  "n": int(len(test_df))},
        },
        "features": features,
        "models": {},
    }

    for name, model in {"logreg": logreg, "random_forest": rf}.items():
        model.fit(X_train, y_train)
        out["models"][name] = {
            "val": _eval(model, X_val, y_val),
            "test": _eval(model, X_test, y_test),
        }

    return out

def tune_random_forest(
    df: pd.DataFrame,
    target: str,
    split: SplitSpec,
    n_estimators: int = 400,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Tune RF hyperparams using ONLY the validation set (macro-F1).
    Returns:
      results_df sorted by val_macro_f1 desc
      best_params dict (for the RF classifier)
    """
    train_df, val_df, test_df = time_split_3way(df, split)  # test_df not used for tuning
    features = select_numeric_features(df, target=target)

    X_train, y_train = train_df[features], train_df[target].astype(int)
    X_val, y_val = val_df[features], val_df[target].astype(int)

    preprocessor = _make_preprocessor(features)

    # Small, sensible grid (fast enough, still meaningful)
    grid = {
        "max_depth": [3, 5, 8, None],
        "min_samples_leaf": [1, 5, 20],
        "max_features": ["sqrt", 0.5, None],
    }

    rows = []
    for max_depth, min_samples_leaf, max_features in product(
        grid["max_depth"], grid["min_samples_leaf"], grid["max_features"]
    ):
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )

        model = Pipeline([
            ("prep", preprocessor),
            ("clf", clf),
        ])

        model.fit(X_train, y_train)
        pred_val = model.predict(X_val)

        val_macro_f1 = float(f1_score(y_val, pred_val, average="macro"))
        val_acc = float(accuracy_score(y_val, pred_val))

        rows.append({
            "val_macro_f1": val_macro_f1,
            "val_accuracy": val_acc,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
        })

    results = pd.DataFrame(rows).sort_values(
        ["val_macro_f1", "val_accuracy"], ascending=False
    ).reset_index(drop=True)

    best = results.iloc[0].to_dict()
    best_max_depth = best["max_depth"]
    if pd.isna(best_max_depth):
        best_max_depth = None
    else:
        best_max_depth = int(best_max_depth)
    best_params = {
        "n_estimators": n_estimators,
        "random_state": random_state,
        "n_jobs": -1,
        "class_weight": "balanced_subsample",
        "max_depth": best["max_depth"],
        "min_samples_leaf": int(best["min_samples_leaf"]),
        "max_features": best["max_features"],
    }

    return results, best_params


def train_best_random_forest(
    df: pd.DataFrame,
    target: str,
    split: SplitSpec,
    best_params: dict,
) -> dict:
    """
    Fit best RF on train+val, then evaluate once on test.
    """
    train_df, val_df, test_df = time_split_3way(df, split)
    features = select_numeric_features(df, target=target)

    trainval = pd.concat([train_df, val_df], axis=0, ignore_index=True)

    X_trainval, y_trainval = trainval[features], trainval[target].astype(int)
    X_test, y_test = test_df[features], test_df[target].astype(int)

    preprocessor = _make_preprocessor(features)

    clf = RandomForestClassifier(**best_params)
    model = Pipeline([
        ("prep", preprocessor),
        ("clf", clf),
    ])

    model.fit(X_trainval, y_trainval)
    test_metrics = _eval(model, X_test, y_test)

    return {
        "best_params": best_params,
        "test": test_metrics,
        "n_trainval": int(len(trainval)),
        "n_test": int(len(test_df)),
        "features": features,
    }