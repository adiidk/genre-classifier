"""XGBoost classifier that accepts genre names and returns genre names.

Wraps XGBClassifier + LabelEncoder (as in genre_ml_final.py, cell 39) so the
model can be saved with joblib and used like the other models.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class LabelledXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=6, learning_rate=0.1, n_estimators=800,
                 subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                 random_state=42):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder_.classes_
        self.model_ = XGBClassifier(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self.model_.fit(X, self.label_encoder_.transform(y))
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]