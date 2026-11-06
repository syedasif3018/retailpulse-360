"""Build dist/retailpulse-app.zip: only the files the dashboard needs at runtime.

Run from the repository root:  python deploy/make_package.py
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "retailpulse-app.zip"
FILES = [
    "app/streamlit_app.py",
    "app/data/scores_2011-09-01.csv",
    "app/data/scores_2011-12-01.csv",
    "app/data/feature_reference.json",
    "models/churn_model.joblib",
    "reports/model_results.json",
    "reports/figures/fig13_test_roc_pr.png",
    "reports/figures/fig14_confusion_matrix.png",
    "reports/figures/fig15_permutation_importance.png",
    "reports/figures/fig16_cumulative_gains.png",
    ".streamlit/config.toml",
]

OUT.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        z.write(ROOT / f, f)
    # the platform installs dependencies from requirements.txt at the package root
    z.write(ROOT / "requirements-app.txt", "requirements.txt")
print(f"{OUT} ({OUT.stat().st_size / 1e6:.2f} MB, {len(FILES) + 1} files)")
