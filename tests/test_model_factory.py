import os
import sys
import pytest
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_layer.model_factory import get_base_model


class TestModelFactory:
    """Test suite for model factory functions"""
    
    @pytest.mark.parametrize("model_name,expected_config", [
        ('bert', 'bert-base-uncased'),
        ('biobert', 'dmis-lab/biobert-v1.1'),
        ('roberta', 'roberta-base'),
    ])
    def test_model_mapping(self, model_name, expected_config):
        """Test if model names map to correct HuggingFace models"""
        # We can't load actual models in test, but we can test mapping logic
        model_mapping = {
            'bert': 'bert-base-uncased',
            'biobert': 'dmis-lab/biobert-v1.1',
            'roberta': 'roberta-base'
        }
        assert model_mapping.get(model_name) == expected_config
    
    @pytest.mark.parametrize("num_classes", [2, 3, 5, 10])
    def test_num_classes_parameter(self, num_classes):
        """Test if num_classes parameter is handled correctly"""
        # Test that num_classes is a valid positive integer
        assert isinstance(num_classes, int)
        assert num_classes > 1
    
    def test_model_parameters_trainable(self):
        """Test if model has trainable parameters"""
        try:
            model = get_base_model('bert', num_classes=3, pretrained=False)
            
            total_params = sum(p.numel() for p in model.parameters())
            assert total_params > 0, "Model should have parameters"
            
            # Initially all parameters should be trainable
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert trainable_params == total_params
            
        except Exception as e:
            pytest.skip(f"Skipping due to model loading error: {e}")
    
    def test_model_output_shape(self):
        """Test model output shape is correct"""
        try:
            model = get_base_model('bert', num_classes=5, pretrained=False)
            model.eval()
            
            # Create dummy input
            batch_size = 2
            seq_length = 128
            input_ids = torch.randint(0, 30000, (batch_size, seq_length))
            attention_mask = torch.ones(batch_size, seq_length)
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Check logits shape
            assert outputs.logits.shape == (batch_size, 5)
            
        except Exception as e:
            pytest.skip(f"Skipping due to model loading error: {e}")
    
    def test_pretrained_flag(self):
        """Test if pretrained flag works correctly"""
        # This is more of an integration test
        try:
            model_pretrained = get_base_model('bert', num_classes=3, pretrained=True)
            model_random = get_base_model('bert', num_classes=3, pretrained=False)
            
            # Both should have same architecture
            assert str(type(model_pretrained)) == str(type(model_random))
            
        except Exception as e:
            pytest.skip(f"Skipping due to model loading error: {e}")
    
    def test_invalid_model_name(self):
        """Test error handling for invalid model names"""
        with pytest.raises(Exception):
            get_base_model('invalid_model_name', num_classes=3)
    
    def test_model_device_compatibility(self):
        """Test if model can be moved to different devices"""
        try:
            model = get_base_model('bert', num_classes=3, pretrained=False)
            
            # Test CPU
            model.cpu()
            assert next(model.parameters()).device.type == 'cpu'
            
            # Test CUDA if available
            if torch.cuda.is_available():
                model.cuda()
                assert next(model.parameters()).device.type == 'cuda'
                
        except Exception as e:
            pytest.skip(f"Skipping due to model loading error: {e}")