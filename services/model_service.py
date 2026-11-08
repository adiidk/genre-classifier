"""Loads the saved models and turns features into genre probabilities."""
import json
import threading
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import sklearn

from services.errors import AppError

PREFERRED_ORDER = ["logistic_regression", "svm"]


class ModelRegistry:
    def __init__(self, models_dir):
        self.models_dir = Path(models_dir)
        self._meta = {}
        self._issues = []
        self._cache = {}          # key -> (file modified time, model)
        self._lock = threading.Lock()
        self.refresh()

    # ----------------------------------------------------------- discovery
    def refresh(self):
        """Scan models/ for *.joblib files (any model dropped in is picked up)."""
        meta, issues = {}, []

        for path in sorted(self.models_dir.glob("*.joblib")):
            key = path.stem
            info = {
                "key": key,
                "name": key.replace("_", " ").title(),
                "classes": [],
                "cv_metrics": {},
                "trained_at": None,
                "n_segments_trained": None,
                "path": path,
            }
            json_path = path.with_suffix(".json")
            if json_path.exists():
                try:
                    with open(json_path, encoding="utf-8") as fh:
                        data = json.load(fh)
                    info["name"] = data.get("name", info["name"])
                    info["classes"] = data.get("classes", [])
                    info["cv_metrics"] = data.get("cv_metrics", {})
                    info["trained_at"] = data.get("trained_at")
                    info["n_segments_trained"] = data.get("n_segments_trained")
                    trained_with = data.get("sklearn_version")
                    if trained_with and trained_with != sklearn.__version__:
                        issues.append({
                            "level": "warning",
                            "code": "version_mismatch",
                            "message": (f"'{info['name']}' was trained with scikit-learn {trained_with} "
                                        f"but {sklearn.__version__} is installed. If predictions fail, "
                                        "re-run: python train_models.py"),
                        })
                except (OSError, ValueError):
                    issues.append({
                        "level": "warning",
                        "code": "bad_metadata",
                        "message": f"Could not read {json_path.name}. The model may still work.",
                    })
            meta[key] = info

        if not meta:
            issues.append({
                "level": "error",
                "code": "no_models",
                "message": "No trained models were found. Run  python train_models.py  "
                           "in the project folder, then reload this page.",
            })

        with self._lock:
            self._meta = meta
            self._issues = issues

    @property
    def issues(self):
        return list(self._issues)

    def list_models(self):
        keys = sorted(
            self._meta,
            key=lambda k: (PREFERRED_ORDER.index(k) if k in PREFERRED_ORDER else 99, k),
        )
        public = ("key", "name", "classes", "cv_metrics", "trained_at", "n_segments_trained")
        return [{field: self._meta[k][field] for field in public} for k in keys]

    # -------------------------------------------------------------- loading
    def _get(self, key):
        with self._lock:
            info = self._meta.get(key)
            if info is None:
                raise AppError(
                    "The selected model was not found. Run  python train_models.py  to create it.",
                    "model_missing", 404,
                )
            path = info["path"]
            try:
                mtime = path.stat().st_mtime
            except OSError:
                raise AppError("The model file is missing. Please re-run  python train_models.py.",
                               "model_missing", 404)

            cached = self._cache.get(key)
            if cached and cached[0] == mtime:
                return cached[1], info

            try:
                model = joblib.load(path)
            except Exception as exc:
                raise AppError(
                    "The model file could not be loaded. It may be damaged or made with another "
                    "library version. Please re-run  python train_models.py.",
                    "model_load_failed", 500,
                ) from exc

            if not hasattr(model, "predict_proba"):
                raise AppError("This model cannot output probabilities. Please retrain it.",
                               "model_invalid", 500)

            self._cache[key] = (mtime, model)
            return model, info

    # ----------------------------------------------------------- prediction
    def predict(self, key, frame):
        model, info = self._get(key)

        try:
            proba = model.predict_proba(frame)
        except Exception as exc:
            raise AppError(
                "The model could not make a prediction from these features. "
                "Try re-running  python train_models.py.",
                "prediction_failed", 500,
            ) from exc

        classes = [str(c) for c in model.classes_]

        # the song-level result is the mean of the per-segment probabilities
        mean = proba.mean(axis=0)
        mean = mean / mean.sum()
        order = np.argsort(mean)[::-1]
        ranking = [{"genre": classes[i], "probability": float(mean[i])} for i in order]

        segment_top = [classes[i] for i in proba.argmax(axis=1)]
        segment_predictions = [
            {"index": i, "genre": segment_top[i], "confidence": float(proba[i].max())}
            for i in range(len(segment_top))
        ]
        votes = dict(Counter(segment_top).most_common())

        return {
            "model": {"key": key, "name": info["name"]},
            "predicted_genre": ranking[0]["genre"],
            "confidence": ranking[0]["probability"],
            "ranking": ranking,
            "segment_predictions": segment_predictions,
            "votes": votes,
            "n_segments": len(segment_top),
        }