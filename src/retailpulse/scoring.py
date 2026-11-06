"""Phase 4: score customers with the saved model and export small files for the dashboard.

Run from the repository root (after notebook 03 has saved the model):
    python -m src.retailpulse.scoring
Outputs (committed, a few hundred KB, so the dashboard runs without the raw data):
    app/data/scores_2011-09-01.csv  - test cohort, with the actual outcome (lets users check the model)
    app/data/scores_2011-12-01.csv  - latest cutoff in the data, no outcome yet ("who to contact now")
    app/data/feature_reference.json - medians and ranges used by the what-if form
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from .cleaning import load_clean
from .config import MODELS, ROOT
from .features import NUMERIC_FEATURES, TARGET, build_snapshot

APP_DATA = ROOT / "app" / "data"
MODEL_PATH = MODELS / "churn_model.joblib"
RISK_BANDS = [0.0, 0.4, 0.6, 0.8, 1.0001]
RISK_LABELS = ["Low", "Medium", "High", "Very high"]


def load_bundle(path=MODEL_PATH) -> dict:
    return joblib.load(path)


def score(bundle: dict, snap: pd.DataFrame) -> pd.DataFrame:
    """Add churn probability, risk band, predicted label and value-weighted risk."""
    out = snap.copy()
    p = bundle["pipeline"].predict_proba(out[bundle["features"]])[:, 1]
    out["churn_probability"] = p.round(4)
    out["predicted_churn"] = (p >= bundle["threshold"]).astype(int)
    out["risk_band"] = pd.cut(p, RISK_BANDS, labels=RISK_LABELS, right=False).astype(str)
    out["revenue_at_risk"] = (p * out["monetary"]).round(2)  # p(churn) x past-365-day spend
    return out


def export(bundle: dict, tx: pd.DataFrame, cutoff: str, with_label: bool) -> pd.DataFrame:
    snap = build_snapshot(tx, cutoff, with_label=with_label)
    scored = score(bundle, snap)
    cols = (["Customer ID", "cutoff", "country"] + NUMERIC_FEATURES
            + ["churn_probability", "predicted_churn", "risk_band", "revenue_at_risk"]
            + ([TARGET] if with_label else []))
    scored = scored[cols].sort_values("revenue_at_risk", ascending=False)
    scored["cutoff"] = scored["cutoff"].dt.date.astype(str)
    num = scored.select_dtypes("number").columns
    scored[num] = scored[num].round(4)
    path = APP_DATA / f"scores_{cutoff}.csv"
    scored.to_csv(path, index=False)
    print(f"{path.name}: {len(scored):,} customers, mean risk {scored['churn_probability'].mean():.3f}")
    return scored


def main() -> None:
    bundle = load_bundle()
    tx = load_clean()
    APP_DATA.mkdir(parents=True, exist_ok=True)
    test = export(bundle, tx, "2011-09-01", with_label=True)
    export(bundle, tx, "2011-12-01", with_label=False)
    ref = {c: {"median": float(np.nanmedian(test[c])), "p01": float(np.nanpercentile(test[c], 1)),
               "p99": float(np.nanpercentile(test[c], 99))} for c in NUMERIC_FEATURES}
    ref["_meta"] = {"model_name": bundle["model_name"], "threshold": bundle["threshold"],
                    "trained_on_cutoffs": bundle["trained_on_cutoffs"],
                    "sklearn_version": bundle["sklearn_version"]}
    (APP_DATA / "feature_reference.json").write_text(json.dumps(ref, indent=2), encoding="utf-8")
    print("Saved feature_reference.json")


if __name__ == "__main__":
    main()
