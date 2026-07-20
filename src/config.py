"""
Central Configuration for Medical NLP Project
-----------------------------------------------
All dataset-specific mappings, model mappings, and descriptions 
are defined here to ensure consistency across the entire project.
"""

# =============================================================================
# 1. CLASS NAME MAPPING (Label → Disease Name)
# =============================================================================
# Change this dictionary when switching datasets.
# Format: {class_index: "disease_name"}

CLASS_NAMES_MAP = {
    0: "Psoriasis",
    1: "Varicose Veins",
    2: "Typhoid",
    3: "Chicken pox",
    4: "Impetigo",
    5: "Dengue",
    6: "Fungal infection",
    7: "Common Cold",
    8: "Pneumonia",
    9: "Dimorphic Hemorrhoids",
    10: "Arthritis",
    11: "Acne",
    12: "Bronchial Asthma",
    13: "Hypertension",
    14: "Migraine",
    15: "Cervical spondylosis",
    16: "Jaundice",
    17: "Malaria",
    18: "urinary tract infection",
    19: "allergy",
    20: "gastroesophageal reflux disease",
    21: "drug reaction",
    22: "peptic ulcer disease",
    23: "diabetes"
}

# =============================================================================
# 2. CLASS DESCRIPTIONS (Used in XAI & Inference explanations)
# =============================================================================
# Optional: Change or remove this dictionary when switching datasets.
# If a class is not found here, a generic message will be shown.

CLASS_DESCRIPTIONS = {
    "Psoriasis": "Chronic autoimmune skin condition with red, scaly patches",
    "Varicose Veins": "Swollen, twisted veins visible under the skin",
    "Typhoid": "Bacterial infection causing high fever and gastrointestinal issues",
    "Chicken pox": "Viral infection with itchy blisters all over the body",
    "Impetigo": "Contagious bacterial skin infection with honey-colored crusts",
    "Dengue": "Mosquito-borne viral disease with severe flu-like symptoms",
    "Fungal infection": "Infection caused by fungi affecting skin or other body parts",
    "Common Cold": "Mild viral upper respiratory tract infection",
    "Pneumonia": "Lung infection causing inflammation of air sacs",
    "Dimorphic Hemorrhoids": "Swollen blood vessels in the lower rectum or anus",
    "Arthritis": "Inflammation of joints with pain and stiffness",
    "Acne": "Skin condition with pimples, blackheads, and inflamed nodules",
    "Bronchial Asthma": "Chronic lung disease with airway inflammation and narrowing",
    "Hypertension": "Persistently elevated blood pressure in the arteries",
    "Migraine": "Recurrent severe headache with neurological symptoms",
    "Cervical spondylosis": "Age-related wear and tear of neck vertebrae",
    "Jaundice": "Yellowing of skin and eyes due to liver issues",
    "Malaria": "Parasitic disease transmitted by mosquitoes",
    "urinary tract infection": "Bacterial infection affecting the urinary system",
    "allergy": "Immune system overreaction to harmless substances",
    "gastroesophageal reflux disease": "Chronic acid reflux from stomach into esophagus",
    "drug reaction": "Adverse or allergic response to medication",
    "peptic ulcer disease": "Open sores in the stomach or upper intestine lining",
    "diabetes": "Chronic metabolic disorder with elevated blood sugar levels"
}

# =============================================================================
# 3. MODEL MAPPING (Short Name → HuggingFace Model ID)
# =============================================================================
MODEL_MAPPING = {
    "bert": "bert-base-uncased",
    "biobert": "dmis-lab/biobert-v1.1",
    "roberta": "roberta-base"
}

# =============================================================================
# 4. LLM PROVIDERS & AVAILABLE MODELS
# =============================================================================
LLM_MODEL_OPTIONS = {
    "Perplexity": ["sonar-pro", "sonar", "sonar-reasoning-pro", "sonar-reasoning"],
    "Google Gemini": [
        "models/gemini-2.0-flash", "models/gemini-2.0-flash-lite",
        "models/gemini-1.5-pro", "models/gemini-1.5-flash"
    ],
    "Groq": [
        "llama-3.3-70b-versatile", "llama-3.1-8b-instant",
        "mixtral-8x7b-32768", "qwen-2.5-32b"
    ]
}

# =============================================================================
# 5. HELPER FUNCTIONS
# =============================================================================
def get_class_name(label: int) -> str:
    """Convert numeric label to disease name."""
    return CLASS_NAMES_MAP.get(label, f"Unknown-{label}")

def get_class_description(class_name: str) -> str:
    """Get description for a disease class."""
    return CLASS_DESCRIPTIONS.get(class_name, "")

def get_hf_model_name(model_name: str) -> str:
    """Convert short model name to HuggingFace model ID."""
    return MODEL_MAPPING.get(model_name, model_name)

def get_num_classes() -> int:
    """Return the number of unique classes in the dataset."""
    return len(CLASS_NAMES_MAP)

def get_all_class_names() -> list:
    """Return a list of all disease names."""
    return list(CLASS_NAMES_MAP.values())

def get_llm_models(provider: str) -> list:
    """Get available models for a specific LLM provider."""
    return LLM_MODEL_OPTIONS.get(provider, [])