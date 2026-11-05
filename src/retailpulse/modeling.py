"""Phase 3: time-based churn modelling (baselines, 4 models, tuning, evaluation).

Split design (monthly cutoffs, label = no purchase in the next 90 days):
  train      : cutoffs whose 90-day label is fully known by the validation cutoff
               (2010-12-01 .. 2011-03-01)
  validation : 2011-06-01  -> hyper-parameter tuning, model + threshold selection
  test       : 2011-09-01  -> touched once, for the final report
  final refit: every cutoff whose label is known by the test cutoff
               (2010-12-01 .. 2011-06-01)
All preprocessing lives inside sklearn Pipelines, so imputation medians,
scaling statistics and one-hot categories are learned from training rows only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, PredefinedSplit, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE
from .features import CATEGORICAL_FEATURES, HORIZON_DAYS, NUMERIC_FEATURES, TARGET

ALL_CUTOFFS = pd.date_range("2010-12-01", "2011-09-01", freq="MS")
VAL_CUTOFF = pd.Timestamp("2011-06-01")
TEST_CUTOFF = pd.Timestamp("2011-09-01")
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def labels_known_by(cutoffs, when: pd.Timestamp, horizon_days: int = HORIZON_DAYS):
    """Cutoffs whose whole label window has ended by `when` (usable for training then)."""
    return [c for c in cutoffs if c + pd.Timedelta(days=horizon_days) <= when]


TRAIN_CUTOFFS = labels_known_by(ALL_CUTOFFS, VAL_CUTOFF)
REFIT_CUTOFFS = labels_known_by(ALL_CUTOFFS, TEST_CUTOFF)


def split(snaps: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train": snaps[snaps["cutoff"].isin(TRAIN_CUTOFFS)],
        "val": snaps[snaps["cutoff"] == VAL_CUTOFF],
        "refit": snaps[snaps["cutoff"].isin(REFIT_CUTOFFS)],
        "test": snaps[snaps["cutoff"] == TEST_CUTOFF],
    }


def make_preprocessor(scale: bool) -> ColumnTransformer:
    """Median imputation (+ missing flag); log1p + standardisation for scale-sensitive models."""
    num = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num += [("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("scale", StandardScaler())]
    return ColumnTransformer([
        ("num", Pipeline(num), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ], verbose_feature_names_out=False)


def model_specs() -> dict[str, dict]:
    """Estimator, whether it needs scaling, and the hyper-parameter search space."""
    rs = RANDOM_STATE
    return {
        "Logistic Regression": dict(
            est=LogisticRegression(max_iter=5000, class_weight="balanced"), scale=True, n_iter=20,
            params={"clf__C": loguniform(1e-3, 1e2)}),
        "Decision Tree": dict(
            est=DecisionTreeClassifier(class_weight="balanced", random_state=rs), scale=False, n_iter=40,
            params={"clf__max_depth": [3, 4, 5, 6, 8, 10, 12, None],
                    "clf__min_samples_leaf": [5, 10, 20, 50, 100, 200],
                    "clf__criterion": ["gini", "entropy"]}),
        "Random Forest": dict(
            est=RandomForestClassifier(class_weight="balanced", random_state=rs, n_jobs=-1),
            scale=False, n_iter=25,
            params={"clf__n_estimators": [200, 400],
                    "clf__max_depth": [6, 8, 12, 16, None],
                    "clf__min_samples_leaf": [1, 5, 10, 20, 50],
                    "clf__max_features": ["sqrt", 0.5, 0.8]}),
        "Gradient Boosting": dict(
            est=HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False,
                                               random_state=rs),
            scale=False, n_iter=30,
            params={"clf__learning_rate": loguniform(0.02, 0.3),
                    "clf__max_iter": [100, 200, 400],
                    "clf__max_leaf_nodes": [7, 15, 31, 63],
                    "clf__min_samples_leaf": [20, 50, 100, 200],
                    "clf__l2_regularization": [0.0, 0.1, 1.0, 10.0]}),
    }


def make_pipeline(spec: dict) -> Pipeline:
    from sklearn.base import clone
    return Pipeline([("prep", make_preprocessor(spec["scale"])), ("clf", clone(spec["est"]))])


def tune(spec: dict, train: pd.DataFrame, val: pd.DataFrame) -> RandomizedSearchCV:
    """Randomised search scored on the *time-based* validation month (PredefinedSplit)."""
    X = pd.concat([train[FEATURES], val[FEATURES]])
    y = pd.concat([train[TARGET], val[TARGET]])
    fold = np.r_[np.full(len(train), -1), np.zeros(len(val), dtype=int)]  # -1 = always train
    search = RandomizedSearchCV(make_pipeline(spec), spec["params"], n_iter=spec["n_iter"],
                                scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
                                refit=False, cv=PredefinedSplit(fold), random_state=RANDOM_STATE,
                                n_jobs=-1)
    return search.fit(X, y)


def best_params(search: RandomizedSearchCV, metric: str = "roc_auc") -> dict:
    res = search.cv_results_
    return res["params"][int(np.argmax(res[f"mean_test_{metric}"]))]


def group_cv_auc(pipe: Pipeline, train: pd.DataFrame, n_splits: int = 5) -> np.ndarray:
    """5-fold CV on the training months, grouped by customer (no customer in two folds)."""
    return cross_val_score(pipe, train[FEATURES], train[TARGET], groups=train["Customer ID"],
                           cv=GroupKFold(n_splits=n_splits), scoring="roc_auc", n_jobs=-1)


def best_threshold(y, proba) -> float:
    """Probability threshold that maximises macro-F1 (chosen on validation data only)."""
    grid = np.round(np.arange(0.20, 0.81, 0.01), 2)
    scores = [f1_score(y, (proba >= t).astype(int), average="macro") for t in grid]
    return float(grid[int(np.argmax(scores))])


def evaluate(y, proba, threshold: float = 0.5) -> dict:
    y = np.asarray(y)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y, proba) if len(np.unique(proba)) > 1 else 0.5
    return {
        "threshold": threshold, "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision_churn": precision_score(y, pred, zero_division=0),
        "recall_churn": recall_score(y, pred, zero_division=0),
        "f1_churn": f1_score(y, pred, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "roc_auc": auc, "pr_auc": average_precision_score(y, proba),
        "brier": brier_score_loss(y, np.clip(proba, 0, 1)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


class RecencyRule:
    """Business-rule baseline: 'no order in the last 90 days => churn'. Score = recency."""

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        r = np.asarray(X["recency_days"], dtype=float)
        p = np.clip(r / 365.0, 0, 1)                 # monotone score in [0, 1]
        return np.c_[1 - p, p]


RECENCY_RULE_THRESHOLD = 90 / 365.0


def make_dummy() -> Pipeline:
    return Pipeline([("prep", make_preprocessor(False)),
                     ("clf", DummyClassifier(strategy="most_frequent"))])


def lift_table(y, proba, monetary, bins: int = 10) -> pd.DataFrame:
    """Customers ranked by churn risk into deciles: churn rate, lift and revenue."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(proba), "spend": np.asarray(monetary)})
    df["decile"] = pd.qcut(df["p"].rank(method="first", ascending=False), bins, labels=range(1, bins + 1))
    t = df.groupby("decile", observed=True).agg(customers=("y", "size"), churners=("y", "sum"),
                                                churn_rate=("y", "mean"), mean_risk=("p", "mean"),
                                                past_365d_spend=("spend", "sum"))
    t["lift"] = t["churn_rate"] / df["y"].mean()
    t["cum_share_of_churners"] = t["churners"].cumsum() / df["y"].sum()
    return t
