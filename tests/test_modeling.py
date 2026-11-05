import numpy as np
import pandas as pd

from src.retailpulse import modeling as M
from src.retailpulse.features import NUMERIC_FEATURES


def test_time_split_never_uses_unknown_labels():
    # every training cutoff's 90-day label window ends before the validation cutoff
    assert all(c + pd.Timedelta(days=90) <= M.VAL_CUTOFF for c in M.TRAIN_CUTOFFS)
    assert all(c + pd.Timedelta(days=90) <= M.TEST_CUTOFF for c in M.REFIT_CUTOFFS)
    assert M.VAL_CUTOFF not in M.TRAIN_CUTOFFS and M.TEST_CUTOFF not in M.REFIT_CUTOFFS
    assert max(M.TRAIN_CUTOFFS) == pd.Timestamp("2011-03-01")


def test_pipeline_handles_missing_values_and_unseen_category():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(1, 100, size=(60, len(NUMERIC_FEATURES))), columns=NUMERIC_FEATURES)
    X.loc[:9, "avg_days_between_orders"] = np.nan
    X["country_group"] = ["United Kingdom", "Germany"] * 30
    y = (X["recency_days"] > 50).astype(int)
    spec = M.model_specs()["Logistic Regression"]
    pipe = M.make_pipeline(spec).fit(X, y)
    new = X.head(3).copy()
    new["country_group"] = "Atlantis"          # category never seen in training
    proba = pipe.predict_proba(new)[:, 1]
    assert proba.shape == (3,) and np.all((proba >= 0) & (proba <= 1))


def test_evaluate_and_threshold():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    res = M.evaluate(y, p, threshold=0.5)
    assert res["tp"] == 1 and res["fn"] == 1 and res["tn"] == 2
    assert res["roc_auc"] == 0.75
    assert 0.2 <= M.best_threshold(y, p) <= 0.8
