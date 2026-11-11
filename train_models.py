"""
Train the two models and save them to models/.

    python train_models.py              full run (cross-validation + final fit)
    python train_models.py --skip-cv    quick run (final fit only)
    python train_models.py --only svm   train only one model

Pipelines are the ones from genre_ml.py. The SVM now has probability=True
so that predict_proba is available.
"""
import argparse
import json
import os
import time
from datetime import datetime
from sklearn.svm import SVC
from xgb_wrapper import LabelledXGBClassifier

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer, StandardScaler
from sklearn.svm import SVC

import config
from feature_extractor import FEATURE_COLUMNS


# ----------------------------------------------------------------- models
def build_logistic_regression():
    return Pipeline([
        ("power", PowerTransformer()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.01, max_iter=2000)),
    ])


def build_svm():
    return make_pipeline(
        PowerTransformer(),
        SVC(kernel="rbf", C=3, gamma=0.03,
            probability=True, random_state=config.RANDOM_STATE),
    )


def build_xgboost():
    # settings from genre_ml_final.py (cell 39); no PowerTransformer needed for trees
    return LabelledXGBClassifier(
        max_depth=6, learning_rate=0.1, n_estimators=800,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )


MODEL_SPECS = {
    "logistic_regression": {"name": "Logistic Regression", "builder": build_logistic_regression},
    "svm": {"name": "Support Vector Machine (RBF)", "builder": build_svm},
    # serial_cv: XGBoost already uses all CPU cores, so folds run one after another
    "xgboost": {"name": "XGBoost", "builder": build_xgboost, "serial_cv": True},
}


# ------------------------------------------------------------------- data
def load_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise SystemExit(f"CSV not found: {csv_path}\nPut your file at data/music_features.csv")

    df = pd.read_csv(csv_path)

    for needed in ("song_id", "label"):
        if needed not in df.columns:
            raise SystemExit(f"Column '{needed}' is missing from the CSV.\nColumns found: {list(df.columns)}")

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(
            "These feature columns are missing from your CSV:\n"
            f"  {missing}\n\nColumns your CSV has:\n  {list(df.columns)}\n\n"
            "Send me this message and I will align the names."
        )

    df = df.dropna(subset=FEATURE_COLUMNS + ["label", "song_id"]).reset_index(drop=True)
    X = df[FEATURE_COLUMNS].astype(float)
    y = df["label"].astype(str)
    groups = df["song_id"]
    return X, y, groups


# ---------------------------------------------------------- cross-validation
def _run_fold(builder, X, y, train_idx, test_idx):
    model = builder()
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    proba = model.predict_proba(X.iloc[test_idx])
    return test_idx, proba, model.classes_


def cross_validate(builder, X, y, groups, n_splits, n_jobs):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                              random_state=config.RANDOM_STATE)
    splits = list(cv.split(X, y, groups))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_fold)(builder, X, y, tr, te) for tr, te in splits
    )

    classes = results[0][2]
    proba = np.zeros((len(X), len(classes)))
    for test_idx, p, _ in results:
        proba[test_idx] = p

    # segment-level accuracy
    segment_acc = accuracy_score(y, classes[proba.argmax(axis=1)])

    # song-level accuracy: average the probabilities of a song's segments
    song_proba = pd.DataFrame(proba, columns=classes).groupby(groups.values).mean()
    song_true = y.groupby(groups.values).first().loc[song_proba.index]
    song_acc = accuracy_score(song_true, classes[song_proba.values.argmax(axis=1)])

    return {"segment_accuracy": float(segment_acc), "song_accuracy": float(song_acc)}


# --------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Train and save the genre models.")
    parser.add_argument("--csv", default=str(config.DEFAULT_CSV), help="path to music_features.csv")
    parser.add_argument("--only", choices=list(MODEL_SPECS), help="train just one model")
    parser.add_argument("--skip-cv", action="store_true", help="skip cross-validation")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=min(5, os.cpu_count() or 1),
                        help="parallel workers for cross-validation")
    args = parser.parse_args()

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.csv} ...")
    X, y, groups = load_dataset(args.csv)
    print(f"  {len(X)} segments | {groups.nunique()} songs | {y.nunique()} genres")
    print(f"  genres: {sorted(y.unique())}\n")

    keys = [args.only] if args.only else list(MODEL_SPECS)

    for key in keys:
        spec = MODEL_SPECS[key]
        print("=" * 60)
        print(spec["name"])
        print("=" * 60)

        metrics = {}
        if not args.skip_cv:
            t0 = time.time()
            cv_jobs = 1 if spec.get("serial_cv") else args.jobs
            print(f"Cross-validating ({args.folds} folds, {cv_jobs} workers) ...")
            metrics = cross_validate(spec["builder"], X, y, groups, args.folds, cv_jobs)
            print(f"  segment accuracy: {metrics['segment_accuracy']:.4f}")
            print(f"  song accuracy:    {metrics['song_accuracy']:.4f}")
            print(f"  ({time.time() - t0:.0f} s)")

        print("Fitting final model on 100% of the data ...")
        t0 = time.time()
        model = spec["builder"]()
        model.fit(X, y)
        print(f"  done ({time.time() - t0:.0f} s)")

        joblib.dump(model, config.MODELS_DIR / f"{key}.joblib")
        meta = {
            "key": key,
            "name": spec["name"],
            "classes": [str(c) for c in model.classes_],
            "n_features": len(FEATURE_COLUMNS),
            "feature_columns": FEATURE_COLUMNS,
            "n_segments_trained": int(len(X)),
            "n_songs_trained": int(groups.nunique()),
            "sklearn_version": sklearn.__version__,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "cv_metrics": metrics,
        }
        with open(config.MODELS_DIR / f"{key}.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        print(f"  saved models/{key}.joblib\n")

    print("All done. Now start the app with:  python app.py")


if __name__ == "__main__":
    main()