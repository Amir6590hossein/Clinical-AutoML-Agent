
# Clinical AutoML Agent - Medical NLP MLOps

[Python 3.8+](https://www.python.org/downloads/)
[PyTorch 2.0+](https://pytorch.org/)
[Transformers 4.38+](https://huggingface.co/transformers/)
[MLflow](https://mlflow.org/)
[Streamlit 1.32+](https://streamlit.io/)

An Agentic AutoML Pipeline for medical text classification that leverages Large Language Models (LLMs) as intelligent orchestrators to guide hyperparameter search, combined with Explainable AI (XAI) and MLflow for experiment tracking.

**Dataset:** Symptom2Disease (24 disease classes from clinical symptom descriptions)  
**Goal:** Maximize classification accuracy through LLM-guided architecture search and tuning strategies

---

## System Architecture

```
Streamlit UI (app.py)
    |
Agentic AutoML Pipeline
    |
    |-- Orchestrator (LLM Agent)
    |-- Memory (Trial Log)
    |-- Budget Manager (Constraints)
    |
Execution Layer
    |
    |-- Trainer
    |-- Evaluator
    |-- Pipeline Services
    |
Model Layer
    |
    |-- BERT / BioBERT / RoBERTa
    |-- Head Only / LoRA / Adapter / Full FT
    |
XAI + LLM Clinical Reasoning
    |
    |-- Captum Integrated Gradients
    |-- LLM Clinical Explanation (Perplexity/Gemini/Groq)
    |
MLflow Tracking (DagsHub)
```

---

## Key Features

### Agentic AutoML with LLM Orchestration
- LLM (Perplexity/Gemini/Groq) intelligently plans hyperparameter trials
- Adapts strategy based on previous trial results
- Falls back to heuristic strategies when no API key is available

### Multi-Strategy Fine-Tuning

| Strategy    | Description              | Trainable Params |
|-------------|--------------------------|------------------|
| head_only   | Only classifier head     | ~1-5%            |
| lora        | Low-Rank Adaptation      | ~2-8%            |
| adapter     | Bottleneck Adapters      | ~3-10%           |
| full_ft     | Full Fine-Tuning         | 100%             |

### Explainable AI (XAI)
- Captum Integrated Gradients for word-level attribution
- Identifies key clinical indicators influencing predictions
- Visual confidence calibration analysis

### LLM-Powered Clinical Reasoning
- Generates medical explanations for predictions
- Discusses differential diagnoses
- Evidence-based clinical narrative

### MLflow Experiment Tracking
- Full experiment lineage (DagsHub integration)
- Model registry with versioning
- Metrics, artifacts, and parameter logging

### Interactive Streamlit Dashboard
- Real-time training progress monitoring
- Model registry browser
- Interactive inference with XAI visualization
- Clinical explanation generation

---

## Project Structure

```
clinical-automl-agent/
    app.py                          Streamlit dashboard
    inference.py                    CLI inference tool
    requirements.txt                Core dependencies
    requirements-dev.txt            Dev dependencies

    src/
        config.py                   Central configuration (class names, models, LLM providers)

        agent_core/
            llm_orchestrator.py     LLM-powered trial planner
            experience_memory.py    Trial history and best model tracking

        execution/
            pipeline_services.py    Main AutoML pipeline
            trainer.py              Model training loop
            evaluator.py            Metrics, calibration, visualization
            budget_manager.py       Resource constraints

        model_layer/
            model_factory.py        BERT/BioBERT/RoBERTa factory
            tuners.py               LoRA, Adapter, Head-only strategies

        data_layer/
            dataset_loader.py       Dataset, tokenization, validation

        utils/
            reproducibility.py      Seed setting, determinism

    tests/
        conftest.py                 Shared fixtures
        test_data_validation.py     Data validation tests
        test_model_building.py      Model and tuner tests
        test_evaluation.py          Evaluator tests
        test_pipeline.py            Integration tests

    experiments/                    Output directory (auto-created)
        model_trial_*/              Saved models per trial
        test_samples.csv
        cm_trial_*.png
        history_trial_*.png
```

---

## Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, CPU works for inference)

### Installation

```bash
git clone https://github.com/Amir6590hossein/Clinical-AutoML-Agent.git
cd Clinical-AutoML-Agent
pip install -r requirements.txt
```

For development:

```bash
pip install -r requirements-dev.txt
```

### Data Preparation
Prepare a CSV file with `text` and `label` columns:

```csv
text,label
"I have red scaly patches on elbows and knees that itch",0
"Painful swollen twisted veins in legs",1
"High fever with severe headache and body pain",2
```

### Run Training Pipeline

```bash
streamlit run app.py
```

Or use CLI inference after training:

```bash
python inference.py --text "Patient has itchy red patches on skin" --model_path experiments/model_trial_1
```

### Configure LLM Provider
Set environment variables or use the Streamlit sidebar:

**For Perplexity:**
```bash
export LLM_PROVIDER="Perplexity"
export LLM_API_KEY="pplx-xxxxxxxxxxxxxxxx"
export LLM_MODEL="sonar-pro"
```

**For Google Gemini:**
```bash
export LLM_PROVIDER="Google Gemini"
export LLM_API_KEY="AIza-xxxxxxxxxxxxxxxx"
export LLM_MODEL="models/gemini-2.0-flash"
```

**For Groq:**
```bash
export LLM_PROVIDER="Groq"
export LLM_API_KEY="gsk_xxxxxxxxxxxxxxxx"
export LLM_MODEL="llama-3.3-70b-versatile"
```

---

## Running Tests

Run all tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_evaluator.py -v
```

---

## MLflow Integration

The system automatically tracks experiments to DagsHub MLflow:

- **Experiment:** Clinical-LLMOps
- **Tracking URI:** https://dagshub.com/Amir6590Hossein/Clinical-LLMOps.mlflow

**Tracked Metrics:**
- final_val_accuracy, final_val_f1_macro
- final_val_entropy, final_val_ece
- train_loss, train_accuracy (per epoch)
- Model parameters, LLM reasoning, and artifacts

---

## Switching Datasets

To use a different dataset, only one file needs to be modified: `src/config.py`

Update the following sections:
```python
# 1. Class name mapping
CLASS_NAMES_MAP = {
    0: "Disease A",
    1: "Disease B",
    ...
}

# 2. Class descriptions (optional)
CLASS_DESCRIPTIONS = {
    "Disease A": "Description of disease A",
    "Disease B": "Description of disease B",
    ...
}

# 3. Model mapping (if adding new models)
MODEL_MAPPING = {
    "bert": "bert-base-uncased",
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    ...
}
```

The rest of the project (pipeline, app, inference) automatically reads from this central configuration.

---

## Supported LLM Providers

| Provider      | Available Models                         | Key Prefix |
|---------------|------------------------------------------|------------|
| Perplexity    | sonar-pro, sonar, sonar-reasoning-pro    | pplx-      |
| Google Gemini | gemini-2.0-flash, gemini-1.5-pro         | AIza-      |
| Groq          | llama-3.3-70b, mixtral-8x7b              | gsk_       |

No API key? The system falls back to heuristic strategies automatically.

---

## Performance Benchmarks

| Model    | Strategy | Accuracy | F1 Macro |
|----------|----------|----------|----------|
| BioBERT  | LoRA     | 0.92+    | 0.91+    |
| RoBERTa  | Adapter  | 0.91+    | 0.90+    |
| BERT     | Full FT  | 0.90+    | 0.89+    |

Results vary based on data and hyperparameters. LLM-guided search typically finds optimal configurations in 3-5 trials.

---

## Advanced Usage

### Programmatic Pipeline

```python
from src.execution.pipeline_services import run_automl_pipeline

result = run_automl_pipeline(
    data_path="data/symptoms.csv",
    train_pct=0.70,
    val_pct=0.15,
    num_trials=5,
    max_epochs=5,
    llm_provider="Perplexity",
    api_key="pplx-xxx",
    model_name="sonar-pro"
)

print(f"Best Accuracy: {result['best_accuracy']:.4f}")
print(f"Best Run ID: {result['best_run_id']}")
```

### Inference with XAI

```python
from src.execution.pipeline_services import predict_pipeline

result = predict_pipeline(
    text="Patient has fever, headache, and joint pain for 3 days",
    model_path="experiments/model_trial_1",
    llm_provider="Perplexity",
    llm_api_key="pplx-xxx"
)

print(f"Diagnosis: {result['predicted_class_name']}")
print(f"Confidence: {result['confidence_score']}")
print(f"Key Indicators: {result['top_influential_words']}")
```

### CLI Inference

```bash
# Predict with best model
python inference.py --text "Patient symptoms here"

# List all available models
python inference.py --list_models

# Use specific model run
python inference.py --text "..." --run_id abc123

# Use local model path
python inference.py --text "..." --model_path experiments/model_trial_1

# With LLM clinical explanation
python inference.py --text "..." --llm-api-key pplx-xxx --llm-provider Perplexity
```

---

## Configuration Reference

All configuration is centralized in `src/config.py`:

| Variable            | Description                              |
|---------------------|------------------------------------------|
| CLASS_NAMES_MAP     | Label index to disease name mapping      |
| CLASS_DESCRIPTIONS  | Disease descriptions for XAI display     |
| MODEL_MAPPING       | Short name to HuggingFace model ID       |
| LLM_MODEL_OPTIONS   | Available models per LLM provider        |

Helper functions:
- `get_class_name(label)` - Convert label to disease name
- `get_hf_model_name(name)` - Convert short name to HF model ID
- `get_llm_models(provider)` - Get models for a provider
- `get_num_classes()` - Return number of classes
- `get_all_class_names()` - Return all disease names

---

## Contributing

Contributions are welcome. See `requirements-dev.txt` for development dependencies.

```bash
pip install -r requirements-dev.txt
pre-commit install
pytest tests/ -v
```

---

## License

MIT License - see LICENSE file for details.

---

## Acknowledgments

- **Symptom2Disease Dataset** - Clinical symptom classification
- **Hugging Face Transformers** - Pre-trained models
- **Captum** - Model interpretability
- **MLflow & DagsHub** - Experiment tracking
- **Streamlit** - Interactive dashboard

---
