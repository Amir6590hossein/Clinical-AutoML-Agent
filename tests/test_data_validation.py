import pytest, pandas as pd, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_layer.dataset_loader import validate_dataset

def test_healthy_data():
    df = pd.DataFrame({'text': ['a','b','c','d','e','f'], 'label': [0,0,0,1,1,1]})
    assert not any("CRITICAL" in i for i in validate_dataset(df))
def test_missing_columns():
    assert any("CRITICAL" in i for i in validate_dataset(pd.DataFrame({'x':[1]})))
    assert any("CRITICAL" in i for i in validate_dataset(pd.DataFrame({'text':['a']})))

def test_null_values():
    assert any("null" in i.lower() for i in validate_dataset(
        pd.DataFrame({'text':['a',None,'c'], 'label':[0,1,0]})))

def test_single_class():
    assert any("CRITICAL" in i for i in validate_dataset(
        pd.DataFrame({'text':['a','b','c'], 'label':[1,1,1]})))

def test_imbalance_warning():
    assert any("imbalance" in i.lower() for i in validate_dataset(
        pd.DataFrame({'text':['x']*105, 'label':[0]*100+[1]*5})))