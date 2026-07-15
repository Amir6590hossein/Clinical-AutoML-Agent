import os
import sys
import pytest
import torch
import torch.nn.functional as F
import numpy as np
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.execution.evaluator import Evaluator


class TestEvaluator:
    """Test suite for Evaluator class"""
    
    @pytest.fixture
    def sample_outputs(self):
        """Create sample model outputs"""
        # Simulate 10 samples, 3 classes
        logits = torch.tensor([
            [2.0, 0.1, 0.1],  # High confidence class 0
            [1.0, 1.5, 0.1],  # Moderate confidence class 1
            [0.1, 0.1, 2.5],  # High confidence class 2
            [0.8, 0.9, 0.8],  # Low confidence (similar scores)
            [3.0, 0.1, 0.1],  # Very high confidence class 0
            [0.1, 2.0, 0.5],  # Good confidence class 1
            [0.3, 0.3, 1.8],  # Moderate confidence class 2
            [1.2, 1.1, 0.2],  # Ambiguous between 0 and 1
            [0.1, 0.1, 3.5],  # Very high confidence class 2
            [0.5, 2.5, 0.1],  # High confidence class 1
        ])
        return logits
    
    @pytest.fixture
    def sample_targets(self):
        """Create sample targets"""
        return torch.tensor([0, 1, 2, 1, 0, 1, 2, 0, 2, 1])
    
    def test_uncertainty_low_entropy(self):
        """Test uncertainty classification for low entropy"""
        result = Evaluator.get_uncertainty_statement(0.3)
        assert "Low" in result or "low" in result.lower()
    
    def test_uncertainty_moderate_entropy(self):
        """Test uncertainty classification for moderate entropy"""
        result = Evaluator.get_uncertainty_statement(0.8)
        assert "Moderate" in result or "moderate" in result.lower()
    
    def test_uncertainty_high_entropy(self):
        """Test uncertainty classification for high entropy"""
        result = Evaluator.get_uncertainty_statement(1.5)
        assert "High" in result or "high" in result.lower()
    
    def test_calculate_ece_perfect_calibration(self):
        """Test ECE calculation for perfectly calibrated predictions"""
        # Create perfectly calibrated predictions
        probs = torch.tensor([
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9]
        ])
        targets = torch.tensor([0, 1, 2])
        
        ece = Evaluator.calculate_ece(probs, targets, n_bins=10)
        assert 0 <= ece <= 1, f"ECE should be between 0 and 1, got {ece}"
    
    def test_calculate_ece_miscalibrated(self):
        """Test ECE for miscalibrated predictions"""
        # Overconfident but wrong
        probs = torch.tensor([
            [0.99, 0.005, 0.005],
            [0.99, 0.005, 0.005],
            [0.005, 0.99, 0.005]
        ])
        targets = torch.tensor([0, 1, 2])
        
        ece = Evaluator.calculate_ece(probs, targets, n_bins=10)
        assert ece > 0, "Miscalibrated predictions should have ECE > 0"
    
    def test_compute_metrics(self, sample_outputs, sample_targets):
        """Test main compute function returns correct metrics"""
        metrics = Evaluator.compute(sample_outputs, sample_targets, sample_targets)
        
        # Check required keys
        required_keys = ['accuracy', 'f1_macro', 'entropy', 'ece', 'conf_matrix']
        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"
        
        # Check value ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1_macro'] <= 1
        assert metrics['entropy'] >= 0
        assert 0 <= metrics['ece'] <= 1
    
    def test_compute_accuracy_perfect(self):
        """Test compute function with perfect predictions"""
        outputs = torch.tensor([
            [10.0, 0.1, 0.1],
            [0.1, 10.0, 0.1],
            [0.1, 0.1, 10.0]
        ])
        targets = torch.tensor([0, 1, 2])
        
        metrics = Evaluator.compute(outputs, targets, targets)
        assert metrics['accuracy'] == 1.0
    
    def test_confusion_matrix_shape(self, sample_outputs, sample_targets):
        """Test confusion matrix has correct shape"""
        metrics = Evaluator.compute(sample_outputs, sample_targets, sample_targets)
        
        n_classes = 3
        cm = np.array(metrics['conf_matrix'])
        assert cm.shape == (n_classes, n_classes)
    
    def test_plot_confusion_matrix(self, sample_outputs, sample_targets):
        """Test confusion matrix plotting function"""
        cm = [[10, 1, 0], [2, 8, 1], [0, 1, 9]]
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        try:
            Evaluator.plot_confusion_matrix(
                np.array(cm),
                ['Class 0', 'Class 1', 'Class 2'],
                temp_path
            )
            
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_plot_training_history(self):
        """Test training history plotting function"""
        history = {
            'train_loss': [0.8, 0.5, 0.3, 0.2],
            'val_loss': [0.9, 0.6, 0.45, 0.4],
            'train_acc': [0.6, 0.75, 0.85, 0.9],
            'val_acc': [0.55, 0.7, 0.8, 0.85]
        }
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        try:
            Evaluator.plot_training_history(history, temp_path)
            
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_calculate_ece_extreme_cases(self):
        """Test ECE with extreme cases"""
        # Completely random predictions
        probs = torch.softmax(torch.randn(100, 5), dim=1)
        targets = torch.randint(0, 5, (100,))
        
        ece = Evaluator.calculate_ece(probs, targets)
        assert 0 <= ece <= 1
    
    def test_get_uncertainty_statement_boundaries(self):
        """Test uncertainty statement at boundary values"""
        # Test exactly at boundaries
        low = Evaluator.get_uncertainty_statement(0.59)
        moderate = Evaluator.get_uncertainty_statement(0.61)
        high = Evaluator.get_uncertainty_statement(1.11)
        
        assert "Low" in low
        assert "Moderate" in moderate
        assert "High" in high