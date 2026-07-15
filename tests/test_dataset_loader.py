import os
import sys
import pytest
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_layer.dataset_loader import MedicalTextDataset
from transformers import AutoTokenizer


class TestMedicalTextDataset:
    """Test suite for MedicalTextDataset class"""
    
    @pytest.fixture(autouse=True)
    def setup(self, sample_data_path, mock_tokenizer_config):
        """Setup test fixtures"""
        self.data_path = sample_data_path
        self.tokenizer = AutoTokenizer.from_pretrained(mock_tokenizer_config['model_name'])
        self.dataset = MedicalTextDataset(
            csv_file=self.data_path,
            tokenizer=self.tokenizer,
            max_length=mock_tokenizer_config['max_length']
        )
    
    def test_dataset_initialization(self):
        """Test if dataset initializes correctly"""
        assert self.dataset is not None
        assert hasattr(self.dataset, 'data_frame')
        assert hasattr(self.dataset, 'tokenizer')
        assert len(self.dataset) > 0
    
    def test_dataset_length(self):
        """Test dataset length matches CSV rows"""
        df = pd.read_csv(self.data_path)
        assert len(self.dataset) == len(df)
    
    def test_get_item_structure(self):
        """Test if __getitem__ returns correct structure"""
        sample = self.dataset[0]
        
        # Check required keys
        assert 'input_ids' in sample
        assert 'attention_mask' in sample
        assert 'grade_label' in sample
        
        # Check types
        assert isinstance(sample['input_ids'], torch.Tensor)
        assert isinstance(sample['attention_mask'], torch.Tensor)
        assert isinstance(sample['grade_label'], torch.Tensor)
        
        # Check dimensions
        assert sample['input_ids'].dim() == 1  # 1D tensor
        assert sample['attention_mask'].dim() == 1
        assert sample['grade_label'].dim() == 0  # scalar
    
    def test_get_item_shapes(self):
        """Test if tensors have correct shapes"""
        sample = self.dataset[0]
        max_length = 512
        
        assert sample['input_ids'].shape[0] == max_length
        assert sample['attention_mask'].shape[0] == max_length
    
    def test_label_range(self):
        """Test if labels are within valid range"""
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            label = sample['grade_label'].item()
            assert label >= 0, f"Label {label} at index {i} is negative"
            assert isinstance(label, int), f"Label at index {i} is not integer"
    
    def test_tokenizer_special_tokens(self):
        """Test if special tokens are present"""
        sample = self.dataset[0]
        input_ids = sample['input_ids'].tolist()
        
        # Check for CLS token (should be 101 for BERT)
        cls_token_id = self.tokenizer.cls_token_id
        assert input_ids[0] == cls_token_id
        
        # Check for SEP token
        sep_token_id = self.tokenizer.sep_token_id
        assert sep_token_id in input_ids
    
    def test_attention_mask_values(self):
        """Test attention mask has only 0 and 1 values"""
        sample = self.dataset[0]
        mask = sample['attention_mask']
        
        unique_values = mask.unique().tolist()
        assert all(v in [0, 1] for v in unique_values)
    
    def test_batch_loading(self):
        """Test if dataset works with DataLoader"""
        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=False)
        
        batch = next(iter(dataloader))
        
        assert batch['input_ids'].shape[0] == 4  # batch size
        assert batch['input_ids'].shape[1] == 512  # max length
        assert batch['grade_label'].shape[0] == 4
    
    def test_dataset_consistency(self):
        """Test if same index returns same data"""
        sample1 = self.dataset[3]
        sample2 = self.dataset[3]
        
        assert torch.equal(sample1['input_ids'], sample2['input_ids'])
        assert torch.equal(sample1['attention_mask'], sample2['attention_mask'])
        assert torch.equal(sample1['grade_label'], sample2['grade_label'])
    
    def test_string_input_handling(self):
        """Test if dataset handles string conversion"""
        sample = self.dataset[0]
        # Should not raise any error even if input was numeric in CSV
        assert sample['input_ids'] is not None
    

def test_dataset_validation(sample_dataframe):
    """Test basic data validation"""
    # Test that required columns exist
    assert 'text' in sample_dataframe.columns
    assert 'label' in sample_dataframe.columns
    
    # Test no null values
    assert not sample_dataframe['text'].isnull().any()
    assert not sample_dataframe['label'].isnull().any()
    
    # Test label is numeric
    assert pd.api.types.is_numeric_dtype(sample_dataframe['label'])