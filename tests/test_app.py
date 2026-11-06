from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
MODEL = APP.parents[1] / "models" / "churn_model.joblib"


@pytest.mark.skipif(not MODEL.exists(), reason="model not trained yet (run notebook 03)")
def test_dashboard_renders_without_errors():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 5
    assert at.metric[0].value  # customers KPI rendered
    # switch to the test cohort (has actual outcomes) and re-run
    at.sidebar.radio[0].set_value(at.sidebar.radio[0].options[1]).run()
    assert not at.exception, at.exception
