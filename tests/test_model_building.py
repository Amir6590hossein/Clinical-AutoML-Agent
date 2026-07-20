import pytest, torch, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model_layer.model_factory import get_base_model
from src.model_layer.tuners import apply_tuning_strategy

@pytest.fixture
def model():
    try: return get_base_model('bert', num_classes=3, pretrained=False)
    except: pytest.skip("Model loading failed")

def test_output_shape():
    for nc in [2, 5, 24]:
        m = get_base_model('bert', num_classes=nc, pretrained=False)
        with torch.no_grad():
            out = m(input_ids=torch.randint(0,1000,(2,32)), attention_mask=torch.ones(2,32))
        assert out.logits.shape == (2, nc)

def test_invalid_model_raises():
    with pytest.raises(Exception): get_base_model('xyz_invalid', num_classes=3)

def test_strategy_trainable_params(model):
    if model is None: pytest.skip()
    counts = {}
    for s in ['head_only', 'full_ft']:
        m = apply_tuning_strategy(get_base_model('bert',3,False) if s!='head_only' else model, s)
        counts[s] = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert counts['full_ft'] > counts['head_only']

def test_lora_adapter_more_than_head(model):
    if model is None: pytest.skip()
    head_count = sum(p.numel() for p in apply_tuning_strategy(model, 'head_only').parameters() if p.requires_grad)
    for s in ['lora', 'adapter']:
        m = apply_tuning_strategy(get_base_model('bert',3,False), s)
        assert sum(p.numel() for p in m.parameters() if p.requires_grad) > head_count

def test_invalid_strategy_raises(model):
    if model is None: pytest.skip()
    with pytest.raises(ValueError): apply_tuning_strategy(model, 'invalid')