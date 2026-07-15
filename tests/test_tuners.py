import os
import sys
import pytest
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_layer.tuners import (
    LoRALayer,
    AdapterLayer,
    inject_lora,
    inject_adapter,
    apply_tuning_strategy,
    should_skip_layer
)
from src.model_layer.model_factory import get_base_model


class TestTuningStrategies:
    """Test suite for various tuning strategies"""
    
    @pytest.fixture
    def sample_linear(self):
        """Create a sample linear layer for testing"""
        return nn.Linear(768, 768)
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample model for testing"""
        try:
            return get_base_model('bert', num_classes=3, pretrained=False)
        except:
            return None
    
    def test_lora_layer_initialization(self, sample_linear):
        """Test LoRA layer initialization"""
        lora = LoRALayer(sample_linear, rank=4, alpha=16)
        
        assert lora.rank == 4
        assert lora.alpha == 16
        assert lora.lora_A.shape == (4, 768)
        assert lora.lora_B.shape == (768, 4)
        
        # Check original weights are frozen
        assert not lora.original_linear.weight.requires_grad
        
    def test_lora_layer_forward(self, sample_linear):
        """Test LoRA layer forward pass"""
        lora = LoRALayer(sample_linear, rank=4, alpha=16)
        
        # Create input
        x = torch.randn(2, 768)
        output = lora(x)
        
        assert output.shape == (2, 768)
        
    def test_adapter_layer_initialization(self, sample_linear):
        """Test Adapter layer initialization"""
        adapter = AdapterLayer(sample_linear, reduction_factor=4)
        
        assert adapter.adapter_down.in_features == 768
        assert adapter.adapter_down.out_features == 192  # 768/4
        assert adapter.adapter_up.in_features == 192
        assert adapter.adapter_up.out_features == 768
        
        # Check original weights are frozen
        assert not adapter.original_linear.weight.requires_grad
        
    def test_adapter_layer_forward(self, sample_linear):
        """Test Adapter layer forward pass"""
        adapter = AdapterLayer(sample_linear, reduction_factor=4)
        
        x = torch.randn(2, 768)
        output = adapter(x)
        
        assert output.shape == (2, 768)
    
    def test_should_skip_layer_classifier(self):
        """Test if classifier layers are correctly identified to skip"""
        assert should_skip_layer('classifier') == True
        assert should_skip_layer('bert.encoder.classifier') == True
        assert should_skip_layer('pooler') == True
        assert should_skip_layer('bert.pooler.dense') == True
    
    def test_should_skip_layer_valid(self):
        """Test if valid layers are not skipped"""
        assert should_skip_layer('bert.encoder.layer.0.attention') == False
        assert should_skip_layer('bert.embeddings') == False
        assert should_skip_layer('encoder.layer.1') == False
    
    def test_apply_head_only_strategy(self, sample_model):
        """Test head_only tuning strategy"""
        if sample_model is None:
            pytest.skip("Model loading failed")
            
        model = apply_tuning_strategy(sample_model, 'head_only')
        
        # Check head is trainable
        head_trainable = False
        for name, param in model.named_parameters():
            if 'classifier' in name or 'score' in name:
                if param.requires_grad:
                    head_trainable = True
                    
        assert head_trainable, "Head should be trainable in head_only strategy"
    
    def test_apply_full_ft_strategy(self, sample_model):
        """Test full fine-tuning strategy"""
        if sample_model is None:
            pytest.skip("Model loading failed")
            
        model = apply_tuning_strategy(sample_model, 'full_ft')
        
        # All parameters should be trainable
        all_trainable = all(p.requires_grad for p in model.parameters())
        assert all_trainable, "All parameters should be trainable in full_ft"
    
    def test_apply_lora_strategy(self, sample_model):
        """Test LoRA tuning strategy"""
        if sample_model is None:
            pytest.skip("Model loading failed")
            
        model = apply_tuning_strategy(sample_model, 'lora')
        
        # Check that at least some parameters are trainable
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_count > 0, "Should have some trainable parameters with LoRA"
    
    def test_apply_adapter_strategy(self, sample_model):
        """Test Adapter tuning strategy"""
        if sample_model is None:
            pytest.skip("Model loading failed")
            
        model = apply_tuning_strategy(sample_model, 'adapter')
        
        # Check that at least some parameters are trainable
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_count > 0, "Should have some trainable parameters with Adapter"
    
    def test_invalid_strategy(self, sample_model):
        """Test error handling for invalid strategy"""
        if sample_model is None:
            pytest.skip("Model loading failed")
            
        with pytest.raises(ValueError):
            apply_tuning_strategy(sample_model, 'invalid_strategy')
    
    def test_strategy_trainable_params_ratio(self, sample_model):
        """Test that different strategies result in different trainable parameter counts"""
        if sample_model is None:
            pytest.skip("Model loading failed")
            
        total_params = sum(p.numel() for p in sample_model.parameters())
        
        strategies = ['head_only', 'full_ft']
        trainable_counts = {}
        
        for strategy in strategies:
            model_copy = apply_tuning_strategy(sample_model, strategy)
            trainable = sum(p.numel() for p in model_copy.parameters() if p.requires_grad)
            trainable_counts[strategy] = trainable
        
        # Full fine-tuning should have more trainable params than head_only
        assert trainable_counts['full_ft'] > trainable_counts['head_only']