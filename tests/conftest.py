import os
import sys
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def sample_data_path():
    """Create a temporary CSV file with sample medical text data"""
    data = {
        'text': [
            "Patient presents with severe headache and blurred vision for 3 days",
            "Mild chest pain and shortness of breath during exercise",
            "Routine checkup, no significant findings, blood pressure normal",
            "Acute abdominal pain with nausea and vomiting since yesterday",
            "Skin rash on arms and legs, no fever, likely allergic reaction",
            "Follow-up visit for diabetes management, blood sugar levels stable",
            "Severe lower back pain radiating to left leg, MRI recommended",
            "Annual physical examination, all vitals within normal range",
            "Persistent cough and fever for 5 days, chest X-ray ordered",
            "Joint pain and stiffness in both knees, suspected osteoarthritis"
        ],
        'label': [3, 2, 1, 3, 1, 1, 3, 1, 2, 2]  # 1:Mild, 2:Moderate, 3:Severe
    }
    
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f, index=False)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_dataframe():
    """Return sample dataframe for testing"""
    return pd.DataFrame({
        'text': [
            "Patient with mild symptoms",
            "Moderate case requiring attention",
            "Severe condition needing immediate care"
        ],
        'label': [1, 2, 3]
    })


@pytest.fixture
def mock_tokenizer_config():
    """Mock tokenizer configuration for testing"""
    return {
        'model_name': 'bert-base-uncased',
        'max_length': 512
    }


@pytest.fixture
def sample_training_config():
    """Sample training configuration for testing"""
    return {
        'model_name': 'bert',
        'strategy': 'head_only',
        'batch_size': 8,
        'lr': 2e-5,
        'epochs': 2,
        'num_classes': 3
    }


@pytest.fixture
def sample_metrics():
    """Sample metrics for testing evaluation"""
    return {
        'accuracy': 0.85,
        'f1_macro': 0.83,
        'val_loss': 0.45,
        'entropy': 0.65,
        'ece': 0.12,
        'conf_matrix': [[10, 1, 0], [2, 8, 1], [0, 1, 9]]
    }


@pytest.fixture
def mock_history():
    """Sample training history for testing memory"""
    return [
        {
            'trial_id': 1,
            'config': {
                'model_name': 'bert',
                'strategy': 'head_only',
                'batch_size': 16,
                'lr': 1e-3,
                'epochs': 3
            },
            'metrics': {
                'accuracy': 0.78,
                'f1_macro': 0.76,
                'val_loss': 0.55
            }
        },
        {
            'trial_id': 2,
            'config': {
                'model_name': 'biobert',
                'strategy': 'lora',
                'batch_size': 16,
                'lr': 2e-4,
                'epochs': 3
            },
            'metrics': {
                'accuracy': 0.85,
                'f1_macro': 0.83,
                'val_loss': 0.42
            }
        }
    ]


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Automatically set up test environment for each test"""
    # Set random seeds for reproducibility
    import random
    import torch
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Disable MLflow tracking during tests
    os.environ['MLFLOW_TRACKING_URI'] = ''
    
    yield
    
    # Cleanup after tests
    torch.cuda.empty_cache() if torch.cuda.is_available() else None