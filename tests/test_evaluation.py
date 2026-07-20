import pytest, torch, numpy as np, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.execution.evaluator import Evaluator

def test_perfect_accuracy():
    out = torch.tensor([[10.,.1,.1],[.1,10.,.1],[.1,.1,10.]])
    tgt = torch.tensor([0,1,2])
    assert Evaluator.compute(out,tgt,tgt)['accuracy'] == 1.0

def test_all_metrics_returned():
    m = Evaluator.compute(torch.randn(10,3), torch.randint(0,3,(10,)), torch.randint(0,3,(10,)))
    for k in ['accuracy','f1_macro','entropy','ece','conf_matrix']:
        assert k in m

def test_metric_ranges():
    m = Evaluator.compute(torch.randn(20,5), torch.randint(0,5,(20,)), torch.randint(0,5,(20,)))
    assert 0 <= m['accuracy'] <= 1 and 0 <= m['f1_macro'] <= 1
    assert m['entropy'] >= 0 and 0 <= m['ece'] <= 1

def test_uncertainty_levels():
    assert "Low" in Evaluator.get_uncertainty_statement(0.3)
    assert "Moderate" in Evaluator.get_uncertainty_statement(0.8)
    assert "High" in Evaluator.get_uncertainty_statement(1.5)

def test_ece_calibration():
    perfect = torch.tensor([[.9,.05,.05],[.05,.9,.05],[.05,.05,.9]])
    assert Evaluator.calculate_ece(perfect, torch.tensor([0,1,2])) < 0.3

def test_confusion_matrix():
    cm = np.array(Evaluator.compute(torch.randn(15,4), torch.randint(0,4,(15,)), torch.randint(0,4,(15,)))['conf_matrix'])
    assert cm.shape == (4,4) and cm.sum() == 15