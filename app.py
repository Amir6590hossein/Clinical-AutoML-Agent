import streamlit as st
import os
import sys
import pandas as pd
import json
import time
import threading
from pathlib import Path
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from src.execution.pipeline_services import (
    run_automl_pipeline,
    get_model_registry,
    predict_pipeline
)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
CLASS_NAMES = {
    0: "Psoriasis", 1: "Varicose Veins", 2: "Typhoid",
    3: "Chicken pox", 4: "Impetigo", 5: "Dengue",
    6: "Fungal infection", 7: "Common Cold", 8: "Pneumonia",
    9: "Dimorphic Hemorrhoids", 10: "Arthritis", 11: "Acne",
    12: "Bronchial Asthma", 13: "Hypertension", 14: "Migraine",
    15: "Cervical spondylosis", 16: "Jaundice", 17: "Malaria",
    18: "urinary tract infection", 19: "allergy",
    20: "gastroesophageal reflux disease", 21: "drug reaction",
    22: "peptic ulcer disease", 23: "diabetes"
}

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

MODEL_OPTIONS = {
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

# ---------------------------------------------------------------------
# Thread-safe log storage
# ---------------------------------------------------------------------
if "shared_logs" not in st.session_state:
    st.session_state.shared_logs = []
    st.session_state.log_lock = threading.Lock()

def add_log(log_entry):
    with st.session_state.log_lock:
        st.session_state.shared_logs.append(log_entry)

def get_logs():
    with st.session_state.log_lock:
        return list(st.session_state.shared_logs)

def initialize_session_state():
    defaults = {
        "training_in_progress": False,
        "training_complete": False,
        "best_accuracy": 0.0,
        "best_run_id": None,
        "training_thread": None,
        "pipeline_result": None,
        "pipeline_error": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ---------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------
st.set_page_config(page_title="Medical NLP MLOps - Symptom2Disease", layout="wide")
st.title("Medical NLP Agentic System - Symptom2Disease")
st.markdown("### AutoML Pipeline with LLM-Guided Hyperparameter Search & XAI")

initialize_session_state()

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Global Configuration")
    st.subheader("LLM Settings")
    llm_provider = st.selectbox(
        "Provider", list(MODEL_OPTIONS.keys()),
        disabled=st.session_state.training_in_progress
    )
    api_key = st.text_input(
        "API Key", type="password",
        disabled=st.session_state.training_in_progress
    )
    available_models = MODEL_OPTIONS[llm_provider]
    model_name = st.selectbox(
        "LLM Model", available_models,
        disabled=st.session_state.training_in_progress
    )
    if not api_key:
        st.warning("No API Key provided. Training will use Heuristics.")
    
    st.markdown("---")
    st.markdown("### Dataset Info")
    st.info("**Symptom2Disease Dataset**\n- 24 Disease Classes\n- Clinical text symptoms\n- Multi-class classification")
    
    st.markdown("---")
    st.markdown("### Session Info")
    if st.session_state.training_complete:
        st.success(f"Best Accuracy: {st.session_state.best_accuracy:.4f}")
    elif st.session_state.training_in_progress:
        st.warning("Training in progress...")
    else:
        st.info("Ready to train")

# ---------------------------------------------------------------------
# Main Tabs
# ---------------------------------------------------------------------
tab1, tab2 = st.tabs(["Agentic AutoML", "Model Registry & XAI Inference"])

# =====================================================================
# TAB 1: AutoML Training Pipeline
# =====================================================================
with tab1:
    st.header("Automated Training Pipeline")
    
    # ---- Data Configuration ----
    st.subheader("1. Data Configuration")
    data_path = st.text_input(
        "CSV Data Path", "/kaggle/working/data/medical_ready.csv",
        disabled=st.session_state.training_in_progress
    )
    s1, s2, s3 = st.columns(3)
    with s1:
        train_pct = st.number_input("Train %", 10, 90, 70, 5, disabled=st.session_state.training_in_progress)
    with s2:
        val_pct = st.number_input("Validation %", 5, 50, 15, 5, disabled=st.session_state.training_in_progress)
    with s3:
        test_pct = 100 - (train_pct + val_pct)
        st.metric("Test %", f"{test_pct}%")
    if test_pct < 0:
        st.error("Sum of Train and Validation exceeds 100%.")
    
    st.markdown("---")
    
    # ---- Budget Configuration ----
    st.subheader("2. Budget & Constraints")
    c1, c2, c3 = st.columns(3)
    with c1:
        num_trials = st.slider("Max Trials", 1, 10, 3, disabled=st.session_state.training_in_progress)
    with c2:
        random_seed = st.number_input("Random Seed", 1, 10000, 42, disabled=st.session_state.training_in_progress)
    with c3:
        max_epochs = st.slider("Max Epochs per Trial", 1, 10, 5, disabled=st.session_state.training_in_progress)
    
    st.markdown("---")
    
    # ---- Start Button ----
    if st.session_state.training_in_progress:
        st.button("Training in Progress...", disabled=True, use_container_width=True)
        start_btn = False
    else:
        start_btn = st.button("Start Training Pipeline", type="primary", use_container_width=True)
    
    # ---- Log Display ----
    log_container = st.container()
    with log_container:
        logs = get_logs()
        if logs:
            for log_entry in logs:
                if log_entry["type"] == "reasoning":
                    with st.expander(f"Trial {log_entry['trial_id']} - LLM Strategy Reasoning", expanded=False):
                        c1, c2 = st.columns([1, 1])
                        with c1:
                            st.markdown("**LLM Reasoning:**")
                            st.info(log_entry["reasoning"])
                        with c2:
                            st.markdown("**Configuration:**")
                            st.markdown(
                                f"- Model: {log_entry.get('model_name', 'N/A').upper()}\n"
                                f"- Strategy: {log_entry.get('strategy', 'N/A')}\n"
                                f"- Batch: {log_entry.get('batch_size', 'N/A')}\n"
                                f"- LR: {log_entry.get('lr', 0):.2e}\n"
                                f"- Epochs: {log_entry.get('epochs', 'N/A')}"
                            )
                elif log_entry["type"] == "results":
                    with st.expander(f"Trial {log_entry['trial_id']} - Results", expanded=True):
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Accuracy", f"{log_entry['accuracy']:.4f}")
                        m2.metric("F1 Macro", f"{log_entry['f1_macro']:.4f}")
                        m3.metric("Val Loss", f"{log_entry['val_loss']:.4f}")
                        m4.metric("Entropy", f"{log_entry.get('entropy', 0):.4f}")
                elif log_entry["type"] == "error":
                    st.error(log_entry["msg"])
                elif log_entry["type"] == "info":
                    st.info(log_entry["msg"])
                elif log_entry["type"] == "success":
                    st.success(log_entry["msg"])
    
    # ---- Auto-refresh during training ----
    if st.session_state.training_in_progress:
        if st.session_state.training_thread is not None and not st.session_state.training_thread.is_alive():
            if st.session_state.pipeline_error:
                add_log({"type": "error", "msg": f"Pipeline Error: {st.session_state.pipeline_error}"})
            elif st.session_state.pipeline_result:
                add_log({
                    "type": "success",
                    "msg": f"Completed! Best Accuracy: {st.session_state.pipeline_result.get('best_accuracy', 0):.4f}"
                })
                st.session_state.best_accuracy = st.session_state.pipeline_result.get("best_accuracy", 0)
                st.session_state.training_complete = True
            st.session_state.training_in_progress = False
            st.session_state.training_thread = None
            st.rerun()
        else:
            time.sleep(3)
            st.rerun()
    
    # ---- Start Training ----
    if start_btn:
        if test_pct < 0:
            st.stop()
        if not os.path.exists(data_path):
            st.error(f"Dataset not found: {data_path}")
            st.stop()
        
        st.session_state.training_in_progress = True
        st.session_state.training_complete = False
        st.session_state.shared_logs = []
        st.session_state.pipeline_result = None
        st.session_state.pipeline_error = None
        
        add_log({"type": "info", "msg": f"Starting {num_trials} trials with {llm_provider}..."})
        
        def trial_callback_accumulator(event_type, data):
            if event_type == "reasoning":
                add_log({
                    "type": "reasoning", "trial_id": data["trial_id"],
                    "model_name": data["model_name"], "strategy": data["strategy"],
                    "batch_size": data["batch_size"], "lr": data["lr"],
                    "epochs": data["epochs"], "reasoning": data["reasoning"]
                })
            elif event_type == "trial_complete":
                add_log({
                    "type": "results", "trial_id": data["trial_id"],
                    "accuracy": data["metrics"]["accuracy"],
                    "f1_macro": data["metrics"]["f1_macro"],
                    "val_loss": data["metrics"]["val_loss"],
                    "entropy": data["metrics"].get("entropy", 0)
                })
            elif event_type == "trial_error":
                add_log({"type": "error", "msg": f"Trial {data['trial_id']} Failed: {data['error']}"})
        
        def run_pipeline_background():
            try:
                result = run_automl_pipeline(
                    data_path=data_path,
                    train_pct=train_pct/100,
                    val_pct=val_pct/100,
                    num_trials=num_trials,
                    random_seed=random_seed,
                    max_epochs=max_epochs,
                    llm_provider=llm_provider,
                    api_key=api_key if api_key else None,
                    model_name=model_name,
                    trial_callback=trial_callback_accumulator
                )
                st.session_state.pipeline_result = result
            except Exception as e:
                import traceback
                st.session_state.pipeline_error = f"{str(e)}\n{traceback.format_exc()}"
        
        thread = threading.Thread(target=run_pipeline_background, daemon=True)
        thread.start()
        st.session_state.training_thread = thread
        st.rerun()


# =====================================================================
# TAB 2: Model Registry & XAI Inference
# =====================================================================
with tab2:
    st.header("Model Registry & XAI Inference")
    
    model_folders = list(Path("experiments").glob("model_trial_*"))
    
    if not model_folders:
        st.warning("No models found. Run training first.")
    else:
        registry_data = []
        for folder in sorted(model_folders):
            config_file = folder / "config.json"
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)
                
                # Extract MLflow info
                run_id = config.get("run_id", "N/A")
                if run_id and run_id != "N/A":
                    mlflow_url = f"https://dagshub.com/Amir6590Hossein/Clinical-LLMOps.mlflow/#/experiments/1/runs/{run_id}"
                else:
                    mlflow_url = "N/A"
                
                registry_data.append({
                    "Trial ID": folder.name.replace("model_trial_", ""),
                    "Model": config.get("model_name", "N/A").upper(),
                    "Strategy": config.get("strategy", "N/A"),
                    "Epochs": config.get("epochs", "N/A"),
                    "Accuracy": config.get("metrics", {}).get("accuracy", 0),
                    "F1 Macro": config.get("metrics", {}).get("f1_macro", 0),
                    "Val Loss": config.get("metrics", {}).get("val_loss", 0),
                    "Entropy": config.get("metrics", {}).get("entropy", 0),
                    "Run ID": run_id,
                    "MLflow URL": mlflow_url,
                    "Path": str(folder),
                    "Reasoning": config.get("reasoning", "")
                })
        
        if registry_data:
            registry_df = pd.DataFrame(registry_data)
            
            # ---- Registry Table ----
            st.subheader("Local Model Registry")
            st.dataframe(
                registry_df[["Trial ID", "Model", "Strategy", "Epochs", "Accuracy", "F1 Macro", "Val Loss", "Run ID"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Run ID": st.column_config.TextColumn("Run ID", width="small"),
                    "Accuracy": st.column_config.NumberColumn("Accuracy", format="%.4f"),
                    "F1 Macro": st.column_config.NumberColumn("F1 Macro", format="%.4f"),
                }
            )
            
            st.markdown("---")
            
            # ---- Model Selection ----
            selected_idx = st.selectbox(
                "Select Model for Inference",
                range(len(registry_df)),
                format_func=lambda x: (
                    f"Trial {registry_df.iloc[x]['Trial ID']} - "
                    f"{registry_df.iloc[x]['Model']} ({registry_df.iloc[x]['Strategy']}) - "
                    f"Acc: {registry_df.iloc[x]['Accuracy']:.4f}"
                )
            )
            
            # ---- Model Details & MLflow Info ----
            if selected_idx is not None:
                selected = registry_df.iloc[selected_idx]
                
                with st.expander("Model Details & MLflow Information", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Model Configuration")
                        st.markdown(f"**Trial ID:** {selected['Trial ID']}")
                        st.markdown(f"**Model:** {selected['Model']}")
                        st.markdown(f"**Strategy:** {selected['Strategy']}")
                        st.markdown(f"**Epochs:** {selected['Epochs']}")
                        st.markdown(f"**Local Path:** `{selected['Path']}`")
                    
                    with col2:
                        st.markdown("### Performance Metrics")
                        st.markdown(f"**Accuracy:** {selected['Accuracy']:.4f}")
                        st.markdown(f"**F1 Macro:** {selected['F1 Macro']:.4f}")
                        st.markdown(f"**Val Loss:** {selected['Val Loss']:.4f}")
                        st.markdown(f"**Entropy:** {selected['Entropy']:.4f}")
                        
                        st.markdown("### MLflow Tracking")
                        run_id = selected['Run ID']
                        
                        if run_id and run_id != "N/A":
                            st.code(f"Run ID: {run_id}")
                            mlflow_url = selected['MLflow URL']
                            st.markdown(f"[Open in DagsHub MLflow]({mlflow_url})")
                            st.text_input("Copy Run ID", value=run_id, key=f"run_id_{selected['Trial ID']}")
                        else:
                            st.warning("No MLflow Run ID available (local training only)")
                    
                    # LLM Reasoning
                    if selected.get("Reasoning") and selected["Reasoning"].strip() and selected["Reasoning"] != "N/A":
                        st.markdown("---")
                        st.markdown("### LLM Strategy Reasoning")
                        st.info(selected["Reasoning"])
            
            st.markdown("---")
            
            # ---- Inference Section ----
            st.header("Clinical Text Inference with XAI")
            text_input = st.text_area(
                "Enter Patient Symptoms / Clinical Notes",
                height=150,
                placeholder="Example: I have red, scaly patches on my elbows and knees that itch constantly. The skin is flaking and sometimes bleeds when I scratch too hard."
            )
            
            if st.button("Analyze with XAI", type="primary", use_container_width=True):
                if not text_input:
                    st.warning("Please enter patient symptoms.")
                elif selected_idx is None:
                    st.warning("Please select a model.")
                else:
                    selected = registry_df.iloc[selected_idx]
                    model_path = selected["Path"]
                    
                    with st.spinner("Running XAI Analysis..."):
                        try:
                            result = predict_pipeline(
                                text=text_input,
                                model_path=model_path,
                                llm_provider=llm_provider if api_key else None,
                                llm_api_key=api_key if api_key else None,
                                llm_model_name=model_name if api_key else None
                            )
                            
                            if "error" in result:
                                st.error(f"Prediction failed: {result['error']}")
                            else:
                                predicted_class_name = result["predicted_class_name"]
                                
                                # ---- Prediction Metrics ----
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Predicted Disease", predicted_class_name)
                                col2.metric("Confidence", result["confidence_score"])
                                col3.metric("Entropy", f"{float(result['entropy']):.4f}")
                                col4.metric("Status", result["uncertainty_status"].split("(")[0].strip())
                                
                                # ---- Disease Description ----
                                if predicted_class_name in CLASS_DESCRIPTIONS:
                                    st.info(f"**Description:** {CLASS_DESCRIPTIONS[predicted_class_name]}")
                                
                                # ---- Model Info Bar ----
                                model_info_parts = [
                                    f"**Model:** {selected['Model']}",
                                    f"**Strategy:** {selected['Strategy']}",
                                    f"**Trial ID:** {selected['Trial ID']}"
                                ]
                                
                                if result['model_info'].get('mlflow_metadata'):
                                    meta = result['model_info']['mlflow_metadata']
                                    run_id_short = meta.get('run_id', 'N/A')
                                    if run_id_short and run_id_short != 'N/A':
                                        run_id_short = run_id_short[:8] + "..."
                                    mlflow_url = meta.get('url', '#')
                                    model_info_parts.append(f"[MLflow: {run_id_short}]({mlflow_url})")
                                
                                st.markdown(" | ".join(model_info_parts))
                                
                                # ---- LLM Clinical Explanation ----
                                if result.get("llm_clinical_explanation") and result["llm_clinical_explanation"].strip():
                                    st.markdown("---")
                                    st.markdown("## Clinical AI Analysis (LLM + XAI)")
                                    st.markdown(result["llm_clinical_explanation"])
                                else:
                                    st.markdown("---")
                                    if not api_key:
                                        st.info("Provide an LLM API key in the sidebar to enable AI-generated clinical explanations based on XAI findings.")
                                    else:
                                        st.warning("Clinical explanation could not be generated. Check your LLM API key and connection.")
                                
                                # ---- XAI Word Attributions ----
                                if result.get("word_attributions"):
                                    st.markdown("---")
                                    st.markdown("## Key Clinical Indicators (XAI)")
                                    st.caption("Words with highest influence on the model's prediction. Intensity indicates importance (positive = supports diagnosis).")
                                    
                                    highlighted_html = (
                                        '<div style="line-height: 2.8; padding: 15px; '
                                        'background-color: #1a1a2e; border-radius: 10px;">'
                                    )
                                    for word, score in result["word_attributions"][:15]:
                                        intensity = min(abs(score) * 2.5, 1.0)
                                        bg_color = f"rgba(255, 215, 0, {intensity})"
                                        text_color = "black" if intensity > 0.5 else "white"
                                        highlighted_html += (
                                            f'<span style="background-color: {bg_color}; '
                                            f'padding: 5px 10px; margin: 3px; border-radius: 5px; '
                                            f'font-weight: bold; color: {text_color}; '
                                            f'display: inline-block;">'
                                            f'{word} ({score:.3f})</span> '
                                        )
                                    highlighted_html += "</div>"
                                    st.markdown(highlighted_html, unsafe_allow_html=True)
                                
                                # ---- Probability Distribution ----
                                st.markdown("---")
                                st.markdown("## Top 10 Disease Probabilities")
                                prob_df = pd.DataFrame({
                                    "Disease": [
                                        CLASS_NAMES.get(i, f"Class {i}")
                                        for i in range(len(result["class_probabilities"]))
                                    ],
                                    "Probability": result["class_probabilities"]
                                }).sort_values("Probability", ascending=False).head(10)
                                
                                fig, ax = plt.subplots(figsize=(10, 4))
                                ax.barh(prob_df["Disease"], prob_df["Probability"], color="skyblue")
                                ax.set_xlabel("Probability")
                                ax.invert_yaxis()
                                for i, (_, row) in enumerate(prob_df.iterrows()):
                                    ax.text(row["Probability"] + 0.01, i, f"{row['Probability']:.2%}", va="center")
                                st.pyplot(fig)
                                
                                # ---- Footer ----
                                st.caption(
                                    f"Model: {result['model_info']['model_name'].upper()} | "
                                    f"Strategy: {result['model_info']['strategy']} | "
                                    f"Run ID: {result['model_info'].get('run_id', 'N/A')}"
                                )
                                
                        except Exception as e:
                            st.error(f"Inference error: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())