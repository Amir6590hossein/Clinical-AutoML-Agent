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
from src.config import (
    CLASS_NAMES_MAP, 
    CLASS_DESCRIPTIONS, 
    LLM_MODEL_OPTIONS,
    get_class_name,
    get_class_description,
    get_llm_models
)
# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------





# ---------------------------------------------------------------------
# Initialize Session State
# ---------------------------------------------------------------------
def initialize_session_state():
    defaults = {
        "training_in_progress": False,
        "training_complete": False,
        "best_accuracy": 0.0,
        "best_run_id": None,
        "pipeline_result": None,
        "pipeline_error": None,
        "training_triggered": False,  # Flag to start training
        "training_config": {},  # Store training parameters
        "stop_requested": False,  # Flag for user stop
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
        "Provider", 
        list(LLM_MODEL_OPTIONS.keys()),
        disabled=st.session_state.training_in_progress
    )
    
    api_key = st.text_input(
        "API Key", 
        type="password",
        disabled=st.session_state.training_in_progress,
        help="Your LLM API key"
    )
    
    available_models = get_llm_models(llm_provider)
    model_name = st.selectbox(
        "LLM Model", 
        available_models,
        disabled=st.session_state.training_in_progress
    )
    
    if not api_key:
        st.warning("No API Key provided. Training will use Heuristics.")
    else:
        st.success(f"Using {llm_provider} - {model_name}")
    
    st.markdown("---")
    st.markdown("### Dataset Info")
    st.info(
        "**Symptom2Disease Dataset**\n\n"
        "- 24 Disease Classes\n"
        "- Clinical text symptoms\n"
        "- Multi-class classification"
    )
    
    st.markdown("---")
    st.markdown("### Session Status")
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
# =====================================================================
# TAB 1: AutoML Training Pipeline - LIVE PROGRESS VERSION
# =====================================================================
# =====================================================================
# TAB 1: AutoML Training Pipeline - LIVE PROGRESS VERSION (FIXED)
# =====================================================================
with tab1:
    st.header("Automated Training Pipeline")
    
    # ---- Data Configuration ----
    st.subheader("1. Data Configuration")
    data_path = st.text_input(
        "CSV Data Path", 
        "/kaggle/working/data/medical_ready.csv",
        disabled=st.session_state.training_in_progress
    )
    
    s1, s2, s3 = st.columns(3)
    with s1:
        train_pct = st.number_input(
            "Train %", 10, 90, 70, 5, 
            disabled=st.session_state.training_in_progress
        )
    with s2:
        val_pct = st.number_input(
            "Validation %", 5, 50, 15, 5, 
            disabled=st.session_state.training_in_progress
        )
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
        num_trials = st.slider(
            "Max Trials", 1, 10, 3, 
            disabled=st.session_state.training_in_progress
        )
    with c2:
        random_seed = st.number_input(
            "Random Seed", 1, 10000, 42, 
            disabled=st.session_state.training_in_progress
        )
    with c3:
        max_epochs = st.slider(
            "Max Epochs per Trial", 1, 10, 5, 
            disabled=st.session_state.training_in_progress
        )
    
    st.markdown("---")
    
    # ---- Start/Stop Button ----
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if not st.session_state.training_in_progress:
            if st.button("Start Training Pipeline", type="primary", use_container_width=True):
                if test_pct < 0:
                    st.error("Cannot start: Train + Validation exceeds 100%")
                elif not os.path.exists(data_path):
                    st.error(f"Dataset not found: {data_path}")
                else:
                    st.session_state.training_config = {
                        "data_path": data_path,
                        "train_pct": train_pct/100,
                        "val_pct": val_pct/100,
                        "num_trials": num_trials,
                        "random_seed": random_seed,
                        "max_epochs": max_epochs,
                        "llm_provider": llm_provider,
                        "api_key": api_key,
                        "model_name": model_name,
                    }
                    st.session_state.training_triggered = True
                    st.session_state.training_in_progress = True
                    st.session_state.training_complete = False
                    st.rerun()
        else:
            st.button("Training in Progress...", disabled=True, use_container_width=True, type="primary")
    
    with col2:
        if st.session_state.training_in_progress:
            if st.button("Stop Training", use_container_width=True, type="secondary"):
                st.session_state.stop_requested = True
                st.rerun()
    
    # ---- Training Execution with LIVE Progress ----
    if st.session_state.training_triggered and not st.session_state.training_complete:
        st.session_state.training_triggered = False
        cfg = st.session_state.training_config
        
        st.markdown("---")
        st.subheader("Training Progress")
        
        # Data collectors - use session_state for persistence across reruns
        if "trial_reasonings" not in st.session_state:
            st.session_state.trial_reasonings = []
        if "trial_results_list" not in st.session_state:
            st.session_state.trial_results_list = []
        if "progress_messages" not in st.session_state:
            st.session_state.progress_messages = []
        
        # Clear previous data
        st.session_state.trial_reasonings = []
        st.session_state.trial_results_list = []
        st.session_state.progress_messages = []
        
        # Create containers for live updates
        status_box = st.empty()
        progress_bar = st.progress(0)
        trial_status = st.empty()
        
        # Placeholder for dynamic content
        dynamic_area = st.empty()
        
        def render_all_logs():
            """Render all collected logs in the dynamic area."""
            with dynamic_area.container():
                # Show progress messages
                for msg in st.session_state.progress_messages:
                    st.info(msg)
                
                # Show reasonings
                for data in st.session_state.trial_reasonings:
                    with st.expander(f"Trial {data['trial_id']} - LLM Strategy Reasoning", expanded=True):
                        c1, c2 = st.columns([1, 1])
                        with c1:
                            st.markdown("**LLM Reasoning:**")
                            st.info(data["reasoning"])
                        with c2:
                            st.markdown("**Configuration:**")
                            st.code(
                                f"Model: {data.get('model_name', 'N/A').upper()}\n"
                                f"Strategy: {data.get('strategy', 'N/A')}\n"
                                f"Batch Size: {data.get('batch_size', 'N/A')}\n"
                                f"Learning Rate: {data.get('lr', 0):.2e}\n"
                                f"Epochs: {data.get('epochs', 'N/A')}",
                                language="yaml"
                            )
                
                # Show results
                for data in st.session_state.trial_results_list:
                    with st.expander(f"Trial {data['trial_id']} - Results", expanded=True):
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Accuracy", f"{data['metrics']['accuracy']:.4f}")
                        m2.metric("F1 Macro", f"{data['metrics']['f1_macro']:.4f}")
                        m3.metric("Val Loss", f"{data['metrics']['val_loss']:.4f}")
                        m4.metric("Entropy", f"{data['metrics'].get('entropy', 0):.4f}")
        
        def progress_callback(message, progress_val):
            """Update progress bar and store message."""
            progress_bar.progress(min(progress_val, 1.0))
            st.session_state.progress_messages.append(message)
        
        def trial_callback(event_type, data):
            """Collect trial events."""
            if event_type == "reasoning":
                st.session_state.trial_reasonings.append(data)
                trial_status.info(f"Trial {data['trial_id']}/{cfg['num_trials']}: Planning with {data.get('model_name', 'N/A').upper()} + {data.get('strategy', 'N/A')}")
            elif event_type == "trial_complete":
                st.session_state.trial_results_list.append(data)
                trial_status.success(f"Trial {data['trial_id']}/{cfg['num_trials']}: Completed! Accuracy: {data['metrics']['accuracy']:.4f}")
            elif event_type == "trial_error":
                trial_status.error(f"Trial {data['trial_id']} Failed: {data['error']}")
                st.session_state.progress_messages.append(f"ERROR: Trial {data['trial_id']} Failed: {data['error']}")
        
        try:
            status_box.info(f"Starting {cfg['num_trials']} trials with {cfg['llm_provider']} ({cfg['model_name']})...")
            trial_status.info("Initializing...")
            
            result = run_automl_pipeline(
                data_path=cfg["data_path"],
                train_pct=cfg["train_pct"],
                val_pct=cfg["val_pct"],
                num_trials=cfg["num_trials"],
                random_seed=cfg["random_seed"],
                max_epochs=cfg["max_epochs"],
                llm_provider=cfg["llm_provider"],
                api_key=cfg["api_key"] if cfg["api_key"] else None,
                model_name=cfg["model_name"],
                progress_callback=progress_callback,
                trial_callback=trial_callback
            )
            
            st.session_state.pipeline_result = result
            st.session_state.best_accuracy = result.get("best_accuracy", 0)
            st.session_state.best_run_id = result.get("best_run_id", None)
            
            progress_bar.progress(1.0)
            status_box.success(f"Pipeline Complete! Best Accuracy: {st.session_state.best_accuracy:.4f}")
            trial_status.success(f"All {cfg['num_trials']} trials completed successfully!")
            
            # Render final logs
            render_all_logs()
            
        except Exception as e:
            st.session_state.pipeline_error = str(e)
            status_box.error(f"Pipeline Failed: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
        
        finally:
            st.session_state.training_in_progress = False
            st.session_state.training_complete = True
            st.rerun()

    # Show completion status if training finished previously
    elif st.session_state.training_complete and not st.session_state.training_in_progress:
        if st.session_state.pipeline_error:
            st.error(f"Pipeline Failed: {st.session_state.pipeline_error}")
        elif st.session_state.pipeline_result:
            st.success(f"Training Complete! Best Accuracy: {st.session_state.best_accuracy:.4f}")
            
            # Show stored results
            if "trial_reasonings" in st.session_state:
                for data in st.session_state.trial_reasonings:
                    with st.expander(f"Trial {data['trial_id']} - LLM Strategy Reasoning", expanded=False):
                        c1, c2 = st.columns([1, 1])
                        with c1:
                            st.markdown("**LLM Reasoning:**")
                            st.info(data["reasoning"])
                        with c2:
                            st.markdown("**Configuration:**")
                            st.code(
                                f"Model: {data.get('model_name', 'N/A').upper()}\n"
                                f"Strategy: {data.get('strategy', 'N/A')}\n"
                                f"Batch Size: {data.get('batch_size', 'N/A')}\n"
                                f"Learning Rate: {data.get('lr', 0):.2e}\n"
                                f"Epochs: {data.get('epochs', 'N/A')}",
                                language="yaml"
                            )
            
            if "trial_results_list" in st.session_state:
                for data in st.session_state.trial_results_list:
                    with st.expander(f"Trial {data['trial_id']} - Results", expanded=True):
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Accuracy", f"{data['metrics']['accuracy']:.4f}")
                        m2.metric("F1 Macro", f"{data['metrics']['f1_macro']:.4f}")
                        m3.metric("Val Loss", f"{data['metrics']['val_loss']:.4f}")
                        m4.metric("Entropy", f"{data['metrics'].get('entropy', 0):.4f}")
# =====================================================================
# TAB 2: Model Registry & XAI Inference
# =====================================================================
with tab2:
    st.header("Model Registry & XAI Inference")
    
    model_folders = list(Path("experiments").glob("model_trial_*"))
    
    if not model_folders:
        st.warning("No models found. Run training in Tab 1 first!")
    else:
        # Build registry
        registry_data = []
        for folder in sorted(model_folders):
            config_file = folder / "config.json"
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)
                
                run_id = config.get("run_id", "N/A")
                
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
                    "Path": str(folder),
                    "Reasoning": config.get("reasoning", "")
                })
        
        if registry_data:
            registry_df = pd.DataFrame(registry_data)
            
            st.subheader("Local Model Registry")
            st.dataframe(
                registry_df[["Trial ID", "Model", "Strategy", "Epochs", "Accuracy", "F1 Macro", "Val Loss"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Accuracy": st.column_config.NumberColumn("Accuracy", format="%.4f"),
                    "F1 Macro": st.column_config.NumberColumn("F1 Macro", format="%.4f"),
                    "Val Loss": st.column_config.NumberColumn("Val Loss", format="%.4f"),
                }
            )
            
            st.markdown("---")
            
            # Model Selection
            st.subheader("Select Model for Inference")
            
            selected_idx = st.selectbox(
                "Choose Model",
                range(len(registry_df)),
                format_func=lambda x: (
                    f"Trial {registry_df.iloc[x]['Trial ID']} - "
                    f"{registry_df.iloc[x]['Model']} ({registry_df.iloc[x]['Strategy']}) - "
                    f"Acc: {registry_df.iloc[x]['Accuracy']:.4f}"
                )
            )
            
            if selected_idx is not None:
                selected = registry_df.iloc[selected_idx]
                
                with st.expander("Model Details", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", f"{selected['Accuracy']:.4f}")
                    with col2:
                        st.metric("F1 Macro", f"{selected['F1 Macro']:.4f}")
                    with col3:
                        st.metric("Strategy", selected['Strategy'])
                
                st.markdown("---")
                
                # Inference
                st.subheader("Clinical Text Inference with XAI")
                
                text_input = st.text_area(
                    "Enter Patient Symptoms / Clinical Notes",
                    height=150,
                    placeholder="Example: I have red, scaly patches on my elbows and knees that itch constantly."
                )
                
                if st.button("Analyze with XAI", type="primary", use_container_width=True):
                    if not text_input:
                        st.warning("Please enter patient symptoms.")
                    else:
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
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    col1.metric("Predicted Disease", predicted_class_name)
                                    col2.metric("Confidence", result["confidence_score"])
                                    col3.metric("Entropy", f"{float(result['entropy']):.4f}")
                                    col4.metric("Status", result["uncertainty_status"].split("(")[0].strip())
                                    
                                    if predicted_class_name in CLASS_DESCRIPTIONS:
                                        st.info(f"**Description:** {CLASS_DESCRIPTIONS[predicted_class_name]}")
                                    
                                    if result.get("llm_clinical_explanation") and result["llm_clinical_explanation"].strip():
                                        st.markdown("---")
                                        st.markdown("## Clinical AI Analysis (LLM + XAI)")
                                        st.markdown(result["llm_clinical_explanation"])
                                    
                                    if result.get("word_attributions"):
                                        st.markdown("---")
                                        st.markdown("## Key Clinical Indicators (XAI)")
                                        st.caption("Words with highest influence on prediction")
                                        
                                        for word, score in result["word_attributions"][:10]:
                                            intensity = min(abs(score) * 2.5, 1.0)
                                            bg_color = f"rgba(66, 133, 244, {intensity})"
                                            st.markdown(
                                                f'<span style="background-color: {bg_color}; '
                                                f'padding: 5px 12px; margin: 3px; border-radius: 5px; '
                                                f'font-weight: bold; color: white; '
                                                f'display: inline-block;">'
                                                f'{word} ({score:.3f})</span>',
                                                unsafe_allow_html=True
                                            )
                                        st.markdown("<br>", unsafe_allow_html=True)
                                    
                                    st.markdown("---")
                                    st.markdown("## Top 10 Disease Probabilities")
                                    
                                    prob_df = pd.DataFrame({
                                        "Disease": [
                                            get_class_name(i)
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
                                    
                                    st.caption(
                                        f"Model: {result['model_info']['model_name'].upper()} | "
                                        f"Strategy: {result['model_info']['strategy']}"
                                    )
                            
                            except Exception as e:
                                st.error(f"Inference error: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())