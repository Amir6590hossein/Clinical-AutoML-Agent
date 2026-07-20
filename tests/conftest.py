import os, sys, pytest, pandas as pd, numpy as np, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def sample_data_path():
    data = {'text': [f"patient symptom {i}" for i in range(30)],
            'label': [i % 3 for i in range(30)]}
    df = pd.DataFrame(data)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f, index=False)
        yield f.name
    os.unlink(f.name)

@pytest.fixture(autouse=True)
def setup_test_env():
    import random, torch
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    os.environ['MLFLOW_TRACKING_URI'] = ''
    yield
    torch.cuda.empty_cache() if torch.cuda.is_available() else None