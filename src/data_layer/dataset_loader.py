import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Optional


class MedicalTextDataset(Dataset):
    # PyTorch Dataset for medical text classification tasks.
    
    def __init__(self, csv_file: str, tokenizer, max_length: int = 512):
        try:
            self.data_frame = pd.read_csv(csv_file)
        except Exception as e:
            raise ValueError(f"Could not read CSV file {csv_file}: {e}")
        
        if 'text' not in self.data_frame.columns or 'label' not in self.data_frame.columns:
            raise ValueError("CSV must contain 'text' and 'label' columns")
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self._validate_and_prepare_data()
    
    def _validate_and_prepare_data(self):
        # Clean and validate the loaded dataframe.
        self.data_frame['text'] = self.data_frame['text'].astype(str)
        self.data_frame['text'] = self.data_frame['text'].str.strip()
        self.data_frame['label'] = pd.to_numeric(self.data_frame['label'], errors='coerce')
        
        if self.data_frame['text'].isnull().any():
            print("[Dataset] Found null texts, filling with empty string")
            self.data_frame['text'] = self.data_frame['text'].fillna("")
        
        if self.data_frame['label'].isnull().any():
            print("[Dataset] Found null labels, removing those rows")
            self.data_frame = self.data_frame.dropna(subset=['label'])
        
        self.data_frame['label'] = self.data_frame['label'].astype(int)

    def __len__(self) -> int:
        return len(self.data_frame)

    def __getitem__(self, idx) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        text = str(self.data_frame.iloc[idx, 0])
        label = int(self.data_frame.iloc[idx, 1])
        
        if not text or text.isspace():
            text = " "

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        sample = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'grade_label': torch.tensor(label, dtype=torch.long)
        }
        
        return sample
    
    def get_class_distribution(self) -> dict:
        # Return class distribution as a dictionary.
        return self.data_frame['label'].value_counts().to_dict()
    
    def get_text_length_stats(self) -> dict:
        # Return statistics about text lengths.
        lengths = self.data_frame['text'].str.len()
        return {
            'min': int(lengths.min()),
            'max': int(lengths.max()),
            'mean': float(lengths.mean()),
            'median': float(lengths.median())
        }


def get_tokenizer(model_name: str):
    # Load a tokenizer for the given model name.
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        raise ValueError(f"Could not load tokenizer for {model_name}: {e}")

def validate_dataset(df: pd.DataFrame) -> List[str]:
    """Validate dataset quality before training.
    Returns a list of issues found (empty list means valid dataset)."""
    
    issues = []
    
    # Check required columns
    if 'text' not in df.columns:
        issues.append("CRITICAL: Missing 'text' column")
    if 'label' not in df.columns:
        issues.append("CRITICAL: Missing 'label' column")
    
    if issues:
        return issues
    
    # Check null values
    null_texts = df['text'].isnull().sum()
    if null_texts > 0:
        issues.append(f"WARNING: Found {null_texts} null texts")
    
    null_labels = df['label'].isnull().sum()
    if null_labels > 0:
        issues.append(f"WARNING: Found {null_labels} null labels")
    
    # Check text lengths
    text_lengths = df['text'].astype(str).str.len()
    very_short = (text_lengths < 5).sum()
    if very_short > 0:
        issues.append(f"WARNING: Found {very_short} very short texts (length < 5)")
    
    # Check class distribution
    unique_labels = df['label'].dropna().unique()
    n_classes = len(unique_labels)
    
    print(f"[Validation] Dataset: {len(df)} samples, {n_classes} classes")
    
    if n_classes < 2:
        issues.append("CRITICAL: Dataset must have at least 2 classes")
        return issues
    elif n_classes > 20:
        issues.append(f"WARNING: Large number of classes: {n_classes}")
    
    label_counts = df['label'].value_counts()
    print(f"[Validation] Class distribution: {dict(label_counts)}")
    
    min_samples = label_counts.min()
    max_samples = label_counts.max()
    
    if min_samples < 3:
        issues.append(f"CRITICAL: Some classes have fewer than 3 samples (min: {min_samples})")
    
    imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
    if imbalance_ratio > 10:
        issues.append(f"WARNING: High class imbalance (ratio: {imbalance_ratio:.2f})")
    
    # Check label format
    if not pd.api.types.is_numeric_dtype(df['label'].dropna()):
        issues.append("WARNING: Label column is not numeric, attempting conversion")
    
    # Check duplicates
    duplicate_texts = df['text'].duplicated().sum()
    if duplicate_texts > 0:
        issues.append(f"WARNING: Found {duplicate_texts} duplicate texts")
    
    return issues