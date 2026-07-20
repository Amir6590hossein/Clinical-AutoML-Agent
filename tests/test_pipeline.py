import pytest, tempfile, pandas as pd, os, sys, shutil
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.execution.pipeline_services import run_automl_pipeline
from src.utils.reproducibility import set_seed

@pytest.fixture
def data_path():
    df = pd.DataFrame({'text': [f"Patient {i} reports symptom {i%3}" for i in range(60)],
                       'label': [i % 3 for i in range(60)]})
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f, index=False); path = f.name
    yield path; os.unlink(path)

def test_pipeline_runs(data_path):
    set_seed(42)
    for d in ['experiments','mlruns']:
        if os.path.exists(d): shutil.rmtree(d)
    try:
        r = run_automl_pipeline(data_path, train_pct=0.6, val_pct=0.2, num_trials=2, 
                                max_epochs=1, api_key=None)
        assert r['status'] == 'completed' and r['num_trials_completed'] == 2
        assert len(list(Path('experiments').glob('model_trial_*'))) == 2
    finally:
        for d in ['experiments','mlruns']:
            if os.path.exists(d): shutil.rmtree(d)

def test_pipeline_reproducibility(data_path):
    def run(): return run_automl_pipeline(data_path, train_pct=0.6, val_pct=0.2, 
                                          num_trials=1, max_epochs=1, api_key=None)
    set_seed(42); r1 = run(); shutil.rmtree('experiments', ignore_errors=True)
    set_seed(42); r2 = run(); shutil.rmtree('experiments', ignore_errors=True)
    assert r1['num_trials_completed'] == r2['num_trials_completed']

def test_pipeline_error_handling():
    with pytest.raises(FileNotFoundError):
        run_automl_pipeline("/nonexistent/data.csv", num_trials=1, max_epochs=1)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        pd.DataFrame({'wrong':[1,2]}).to_csv(f, index=False); p = f.name
    try:
        with pytest.raises((ValueError, KeyError)): run_automl_pipeline(p, num_trials=1, max_epochs=1)
    finally: os.unlink(p)