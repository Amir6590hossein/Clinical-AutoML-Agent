import torch
from transformers import AutoModelForSequenceClassification, AutoConfig
from src.config import get_hf_model_name
def get_base_model(model_name, num_classes=5, pretrained=True):
    
    
    
    hf_model_name = get_hf_model_name(model_name)
    
    if pretrained:
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_model_name,
            num_labels=num_classes
        )
    else:
        config = AutoConfig.from_pretrained(hf_model_name, num_labels=num_classes)
        model = AutoModelForSequenceClassification.from_config(config)
        
    return model