from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_ui_starts_and_switches_models(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    app=AppTest.from_file(str(root/'app/app.py'),default_timeout=25).run()
    assert not app.exception
    assert len(app.tabs)==9
    selector=next(w for w in app.selectbox if w.label=='Regression model')
    selector.set_value('mlp').run()
    assert not app.exception
    params=next(w for w in app.text_area if w.label=='Model-specific hyperparameters (YAML)')
    assert 'hidden:' in params.value
