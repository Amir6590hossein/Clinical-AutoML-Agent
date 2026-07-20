import os
import sys
import torch
import mlflow
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.config import CLASS_NAMES_MAP, get_class_name
from src.agent_core.llm_orchestrator import Orchestrator
from src.agent_core.experience_memory import ExperienceMemory
from src.execution.budget_manager import BudgetManager
from src.execution.trainer import ModelTrainer
from src.execution.evaluator import Evaluator
from src.model_layer.model_factory import get_base_model
from src.model_layer.tuners import apply_tuning_strategy
from src.data_layer.dataset_loader import MedicalTextDataset, validate_dataset
from src.utils.reproducibility import set_seed
from src.config import get_hf_model_name
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer
import torch.nn.functional as F


# Disease class names mapping for Symptom2Disease dataset


def setup_mlflow(experiment_name: str = "Clinical-LLMOps"):
    dagshub_uri = "https://dagshub.com/Amir6590Hossein/Clinical-LLMOps.mlflow"
    
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        os.environ["MLFLOW_TRACKING_USERNAME"] = user_secrets.get_secret("DAGSHUB_USERNAME")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = user_secrets.get_secret("DAGSHUB_TOKEN")
        print("[MLflow] Using Kaggle Secrets for authentication")
    except ImportError:
        if os.environ.get("MLFLOW_TRACKING_USERNAME") and os.environ.get("MLFLOW_TRACKING_PASSWORD"):
            print("[MLflow] Using Environment Variables for authentication")
        else:
            print("[MLflow] No authentication found. Using local SQLite tracking.")
            local_uri = "sqlite:///mlflow.db"
            mlflow.set_tracking_uri(local_uri)
            try:
                experiment_id = mlflow.create_experiment(experiment_name)
            except:
                experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
            print(f"[MLflow] Local Tracking URI: {local_uri}")
            print(f"[MLflow] Experiment: {experiment_name} (ID: {experiment_id})")
            return experiment_id
    
    mlflow.set_tracking_uri(dagshub_uri)
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    mlflow.set_experiment(experiment_name)
    print(f"[MLflow] Tracking URI: {dagshub_uri}")
    print(f"[MLflow] Experiment: {experiment_name} (ID: {experiment_id})")
    return experiment_id


def run_automl_pipeline(
    data_path: str,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
    num_trials: int = 3,
    random_seed: int = 42,
    max_epochs: int = 5,
    llm_provider: str = "Perplexity",
    api_key: Optional[str] = None,
    model_name: Optional[str] = "sonar-pro",
    progress_callback=None,
    trial_callback=None
) -> Dict[str, Any]:
    """Run AutoML pipeline with LLM-guided hyperparameter search."""
    set_seed(random_seed)
    
    try:
        setup_mlflow()
    except Exception as e:
        print(f"[Pipeline] MLflow setup warning: {e}")
        print("[Pipeline] Continuing without MLflow tracking...")
    
    os.makedirs("experiments", exist_ok=True)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    print(f"[Pipeline] Loading dataset from: {data_path}")
    df_full = pd.read_csv(data_path)
    
    validation_issues = validate_dataset(df_full)
    if validation_issues:
        print("[Pipeline] Dataset Validation Issues Found:")
        for issue in validation_issues:
            print(f"   {issue}")
        critical_issues = [i for i in validation_issues if i.startswith("CRITICAL:")]
        if critical_issues:
            raise ValueError(f"Cannot continue due to critical data issues: {critical_issues}")
    
    print(f"[Pipeline] Dataset validated successfully")
    print(f"[Pipeline] Dataset shape: {df_full.shape}")
    
    if "text" not in df_full.columns or "label" not in df_full.columns:
        raise ValueError("CSV must have text and label columns")
    
    targets = df_full["label"].values
    indices = np.arange(len(df_full))
    unique_classes = np.unique(targets)
    n_classes = len(unique_classes)
    
    print(f"[Pipeline] Classes: {n_classes} | Distribution: {dict(zip(*np.unique(targets, return_counts=True)))}")
    
    try:
        train_idx, temp_idx = train_test_split(
            indices, train_size=train_pct, stratify=targets, random_state=random_seed
        )
    except ValueError as e:
        print(f"[Pipeline] Stratified split failed: {e}")
        print("[Pipeline] Falling back to random split...")
        train_idx, temp_idx = train_test_split(
            indices, train_size=train_pct, random_state=random_seed
        )
    
    test_pct = 1 - train_pct - val_pct
    
    if test_pct <= 0:
        val_idx = temp_idx
        test_idx = np.array([], dtype=int)
        print("[Pipeline] No test set (train + val = 100%)")
    else:
        val_relative_ratio = val_pct / (val_pct + test_pct)
        temp_targets = targets[temp_idx]
        try:
            val_idx, test_idx = train_test_split(
                temp_idx, train_size=val_relative_ratio, stratify=temp_targets, random_state=random_seed
            )
        except ValueError:
            val_idx, test_idx = train_test_split(
                temp_idx, train_size=val_relative_ratio, random_state=random_seed
            )
    
    if len(test_idx) > 0:
        test_df_save = df_full.iloc[test_idx][["text", "label"]]
        test_df_save.to_csv("experiments/test_samples.csv", index=False)
        print(f"[Pipeline] Test set saved: {len(test_idx)} samples")
    
    y_train = targets[train_idx]
    class_weights = compute_class_weight(
        class_weight="balanced", classes=unique_classes, y=y_train
    )
    print(f"[Pipeline] Class weights: {dict(zip(unique_classes, class_weights))}")
    
    memory = ExperienceMemory()
    orchestrator = Orchestrator(
        max_trials=num_trials,
        provider=llm_provider,
        api_key=api_key if api_key else None,
        model_name=model_name
    )
    budget = BudgetManager(max_trials=num_trials, max_time_hours=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    separator = "=" * 60
    print("")
    print(separator)
    print("[Pipeline] Starting AutoML Pipeline")
    print(f"[Pipeline] Device: {device}")
    print(f"[Pipeline] Trials: {num_trials} | Max Epochs: {max_epochs}")
    print(f"[Pipeline] Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    print(f"[Pipeline] LLM Provider: {llm_provider} | Model: {model_name}")
    print(separator)
    print("")
    
    trial_results = []
    
    for i in range(num_trials):
        budget.start_trial()
        trial_id = budget.current_trial_count
        
        if progress_callback:
            progress_callback(f"Trial {trial_id}/{num_trials}: Planning...", (i) / num_trials)
        
        try:
            plan, reasoning = orchestrator.plan_next_trial(memory)
        except Exception as e:
            print(f"[Pipeline] Orchestrator error: {e}")
            print("[Pipeline] Using default fallback plan...")
            plan = {
                "model_name": "bert",
                "strategy": "head_only",
                "batch_size": 16,
                "lr": 2e-5,
                "epochs": min(3, max_epochs)
            }
            reasoning = "Fallback plan due to orchestrator error"
        
        if "epochs" in plan:
            plan["epochs"] = min(plan["epochs"], max_epochs)
        
        if not plan:
            print(f"[Pipeline] No plan returned. {reasoning}")
            break
        
        if trial_callback:
            trial_callback("reasoning", {
                "trial_id": trial_id,
                "model_name": plan["model_name"],
                "strategy": plan["strategy"],
                "batch_size": plan["batch_size"],
                "lr": plan["lr"],
                "epochs": plan["epochs"],
                "reasoning": reasoning
            })
        
        dash_separator = "-" * 50
        print("")
        print(dash_separator)
        print(f"[Pipeline] Trial {trial_id}: {plan['model_name']} | {plan['strategy']}")
        print(f"[Pipeline] Reasoning: {reasoning}")
        print(dash_separator)
        
        try:
            active_run = mlflow.start_run(run_name=f"trial_{trial_id}_{plan['model_name']}_{plan['strategy']}")
        except Exception as e:
            print(f"[Pipeline] MLflow run creation failed: {e}")
            active_run = None
        
        try:
            if active_run:
                mlflow.log_params({
                    "trial_id": trial_id,
                    "model_name": plan["model_name"],
                    "strategy": plan["strategy"],
                    "batch_size": plan["batch_size"],
                    "learning_rate": plan["lr"],
                    "epochs": plan["epochs"],
                    "random_seed": random_seed,
                    "train_size": len(train_idx),
                    "val_size": len(val_idx),
                    "n_classes": n_classes,
                    "llm_provider": llm_provider,
                    "reasoning": reasoning
                })
            
            
            hf_name = get_hf_model_name(plan["model_name"])
            
            print(f"[Pipeline] Loading tokenizer: {hf_name}")
            tokenizer = AutoTokenizer.from_pretrained(hf_name)
            
            full_ds = MedicalTextDataset(data_path, tokenizer)
            train_subset = Subset(full_ds, train_idx)
            val_subset = Subset(full_ds, val_idx)
            
            train_loader = DataLoader(train_subset, batch_size=plan["batch_size"], shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=plan["batch_size"], shuffle=False)
            
            if active_run:
                mlflow.log_param("tokenizer", hf_name)
            
            print(f"[Pipeline] Building model: {plan['model_name']}")
            model = get_base_model(plan["model_name"], num_classes=n_classes)
            model = apply_tuning_strategy(model, plan["strategy"])
            model = model.to(device)
            
            trainer = ModelTrainer(device=device, class_weights=class_weights.tolist())
            
            if progress_callback:
                progress_callback(f"Trial {trial_id}: Training...", (i + 0.5) / num_trials)
            
            metrics = trainer.run_training(model, train_loader, val_loader, plan)
            
            local_model_path = f"experiments/model_trial_{trial_id}"
            os.makedirs(local_model_path, exist_ok=True)
            
            run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
            
            try:
                torch.save(model.state_dict(), f"{local_model_path}/model_weights.pt")
                
                config_dict = {
                    'model_name': plan['model_name'],
                    'strategy': plan['strategy'],
                    'num_classes': n_classes,
                    'batch_size': plan['batch_size'],
                    'lr': plan['lr'],
                    'epochs': plan['epochs'],
                    'run_id': run_id,
                    'reasoning': reasoning,
                    'metrics': {
                        'accuracy': metrics['accuracy'],
                        'f1_macro': metrics['f1_macro'],
                        'val_loss': metrics['val_loss'],
                        'entropy': metrics.get('entropy', 0),
                        'ece': metrics.get('ece', 0)
                    }
                }
                with open(f"{local_model_path}/config.json", 'w') as f:
                    json.dump(config_dict, f, indent=2)
                
                tokenizer.save_pretrained(f"{local_model_path}/tokenizer")
                
                print(f"[Pipeline] Model saved locally: {local_model_path}")
                
                if active_run:
                    mlflow.log_param("local_model_path", local_model_path)
                    mlflow.log_artifact(f"{local_model_path}/config.json", artifact_path="model_metadata")
                    
            except Exception as e:
                print(f"[Pipeline] Local model save failed: {e}")
            
            if active_run:
                mlflow.log_metrics({
                    "final_val_accuracy": metrics["accuracy"],
                    "final_val_f1_macro": metrics["f1_macro"],
                    "final_val_entropy": metrics["entropy"],
                    "final_val_ece": metrics["ece"],
                    "final_val_loss": metrics["val_loss"]
                })
            
            cm_path = f"experiments/cm_trial_{trial_id}.png"
            Evaluator.plot_confusion_matrix(
                np.array(metrics["conf_matrix"]),
                [f"Class {c}" for c in unique_classes],
                cm_path
            )
            
            hist_path = f"experiments/history_trial_{trial_id}.png"
            Evaluator.plot_training_history(metrics["history"], hist_path)
            
            memory.add_experience(
                trial_id=trial_id,
                config=plan,
                metrics=metrics,
                run_id=run_id
            )
            
            trial_result = {
                "trial_id": trial_id,
                "run_id": run_id,
                "config": plan,
                "metrics": {
                    "accuracy": metrics["accuracy"],
                    "f1_macro": metrics["f1_macro"],
                    "val_loss": metrics["val_loss"],
                    "entropy": metrics.get("entropy", 0),
                    "ece": metrics.get("ece", 0),
                    "conf_matrix": metrics["conf_matrix"],
                    "history": metrics["history"]
                },
                "reasoning": reasoning
            }
            
            trial_results.append(trial_result)
            
            if trial_callback:
                trial_callback("trial_complete", trial_result)
            
            print(f"[Pipeline] Trial {trial_id} Completed!")
            print(f"[Pipeline] Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f}")
            
        except Exception as e:
            print(f"[Pipeline] Trial {trial_id} Failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
            if trial_callback:
                trial_callback("trial_error", {"trial_id": trial_id, "error": str(e)})
        
        finally:
            if "model" in locals():
                del model
            torch.cuda.empty_cache()
            if active_run:
                try:
                    mlflow.end_run()
                except:
                    pass
    
    if progress_callback:
        progress_callback("AutoML Complete!", 1.0)
    
    best_trial = memory.get_best_trial() if memory.history else None
    
    print("")
    print(separator)
    print("[Pipeline] AutoML Pipeline Completed!")
    print(f"[Pipeline] Total Trials: {len(trial_results)}")
    if best_trial:
        print(f"[Pipeline] Best Accuracy: {best_trial['metrics']['accuracy']:.4f}")
        print(f"[Pipeline] Best Run ID: {best_trial['run_id']}")
    print(separator)
    print("")
    
    return {
        "status": "completed",
        "num_trials_completed": len(trial_results),
        "best_accuracy": best_trial["metrics"]["accuracy"] if best_trial else 0.0,
        "best_run_id": best_trial["run_id"] if best_trial else None,
        "trial_results": trial_results
    }


def get_model_registry(experiment_name: str = "medical_nlp_agent") -> pd.DataFrame:
    """Get model registry from MLflow."""
    try:
        setup_mlflow(experiment_name)
    except Exception as e:
        print(f"[Registry] MLflow setup warning: {e}")
    
    try:
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            order_by=["metrics.final_val_accuracy DESC"]
        )
    except Exception as e:
        print(f"[Registry] Could not fetch runs: {e}")
        runs = pd.DataFrame()
    
    if runs.empty:
        return pd.DataFrame(columns=[
            "run_id", "trial_id", "model_name", "strategy",
            "accuracy", "f1_macro", "val_loss", "start_time"
        ])
    
    result_df = pd.DataFrame({
        "run_id": runs["run_id"],
        "trial_id": runs.get("params.trial_id", "N/A"),
        "model_name": runs.get("params.model_name", "N/A"),
        "strategy": runs.get("params.strategy", "N/A"),
        "accuracy": runs.get("metrics.final_val_accuracy", 0.0).round(4),
        "f1_macro": runs.get("metrics.final_val_f1_macro", 0.0).round(4),
        "val_loss": runs.get("metrics.final_val_loss", 0.0).round(4),
        "start_time": runs.get("start_time", "N/A"),
        "llm_reasoning": runs.get("params.reasoning", "")
    })
    
    return result_df


def predict_pipeline(
    text: str,
    model_path: str,
    experiment_name: str = "medical_nlp_agent",
    llm_provider: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model_name: Optional[str] = None
) -> Dict[str, Any]:
    """Run inference with XAI explanations and LLM clinical reasoning."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(model_path):
        return {"error": f"Model path not found: {model_path}"}
    
    try:
        with open(f"{model_path}/config.json", 'r') as f:
            config = json.load(f)
        
        model_name = config.get("model_name", "roberta")
        strategy = config.get("strategy", "full_ft")
        n_classes = config.get("num_classes", 24)
        run_id = config.get("run_id", None)
        
        mlflow_metadata = None
        if run_id:
            try:
                setup_mlflow(experiment_name)
                run = mlflow.get_run(run_id)
                mlflow_metadata = {
                    "run_id": run_id,
                    "accuracy": run.data.metrics.get("final_val_accuracy", 0),
                    "f1": run.data.metrics.get("final_val_f1_macro", 0),
                    "reasoning": run.data.params.get("reasoning", "N/A"),
                    "url": f"https://dagshub.com/Amir6590Hossein/Clinical-LLMOps.mlflow/#/experiments/1/runs/{run_id}"
                }
            except Exception as e:
                print(f"[Predict] MLflow metadata unavailable: {e}")
        
        model = get_base_model(model_name, num_classes=n_classes)
        model = apply_tuning_strategy(model, strategy)
        model.load_state_dict(torch.load(f"{model_path}/model_weights.pt", map_location=device))
        model = model.to(device)
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/tokenizer")
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            conf, pred_idx = torch.max(probs, 1)
        
        predicted_class = pred_idx.item()
        confidence = conf.item()
        predicted_class_name = get_class_name(predicted_class)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
        uncertainty_msg = Evaluator.get_uncertainty_statement(entropy)
        
        # XAI - Word Attributions
        word_attributions = []
        top_keywords = []
        
        try:
            from captum.attr import LayerIntegratedGradients
            
            embedding_layer = None
            if hasattr(model, "roberta"):
                embedding_layer = model.roberta.embeddings
            elif hasattr(model, "bert"):
                embedding_layer = model.bert.embeddings
            
            if embedding_layer is not None:
                lig = LayerIntegratedGradients(model, embedding_layer)
                attributions, delta = lig.attribute(
                    inputs=inputs["input_ids"],
                    baselines=torch.zeros_like(inputs["input_ids"]),
                    additional_forward_args=(inputs["attention_mask"]),
                    target=predicted_class,
                    return_convergence_delta=True,
                    internal_batch_size=1
                )
                
                attributions_sum = attributions.sum(dim=2).squeeze(0)
                attributions_norm = attributions_sum / (torch.norm(attributions_sum) + 1e-9)
                tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                
                current_word = ""
                current_score = 0.0
                token_count = 0
                
                for token, score in zip(tokens, attributions_norm):
                    if token in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"]:
                        if current_word:
                            avg_score = current_score / max(token_count, 1)
                            word_attributions.append((current_word.strip(), round(avg_score, 4)))
                            current_word = ""
                            current_score = 0.0
                            token_count = 0
                        continue
                    
                    if token.startswith("##"):
                        clean_token = token[2:]
                        current_word += clean_token
                        current_score += score.item()
                        token_count += 1
                    elif token.startswith("G") and len(token) > 1 and current_word:
                        clean_token = token[1:]
                        current_word += clean_token
                        current_score += score.item()
                        token_count += 1
                    else:
                        if current_word:
                            avg_score = current_score / max(token_count, 1)
                            word_attributions.append((current_word.strip(), round(avg_score, 4)))
                        clean_token = token.replace("G", "")
                        current_word = clean_token
                        current_score = score.item()
                        token_count = 1
                
                if current_word and current_word.strip():
                    avg_score = current_score / max(token_count, 1)
                    word_attributions.append((current_word.strip(), round(avg_score, 4)))
                
                if word_attributions:
                    word_attributions = [(w, s) for w, s in word_attributions if len(w) > 1 and w.strip()]
                    
                    if word_attributions:
                        max_abs = max(abs(s) for _, s in word_attributions)
                        if max_abs > 0:
                            word_attributions = [
                                (w, round(s/max_abs, 4))
                                for w, s in word_attributions
                                if abs(s/max_abs) > 0.05
                            ]
                        word_attributions.sort(key=lambda x: abs(x[1]), reverse=True)
                        word_attributions = word_attributions[:25]
                    
        except Exception as e:
            print(f"[Predict] XAI failed: {e}")
            word_attributions = []
        
        top_keywords = [w[0] for w in word_attributions[:15]] if word_attributions else []
        
        if not top_keywords:
            import re
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            stopwords = {
                'the', 'and', 'for', 'with', 'that', 'this', 'was', 'are',
                'have', 'been', 'has', 'had', 'not', 'but', 'from', 'they',
                'will', 'would', 'could', 'should', 'what', 'when', 'where',
                'which', 'who', 'whom', 'how', 'very', 'also', 'some'
            }
            top_keywords = [w for w in words if w not in stopwords][:15]
        
        print(f"[Predict] XAI Keywords: {top_keywords[:10]}")
        
        # Generate LLM clinical explanation
        llm_clinical_explanation = ""
        try:
            provider = llm_provider or os.environ.get("LLM_PROVIDER", "Perplexity")
            api_key_value = llm_api_key or os.environ.get("LLM_API_KEY", "")
            model_llm = llm_model_name or os.environ.get("LLM_MODEL", "sonar-pro")
            
            if api_key_value and api_key_value.strip():
                clinical_orchestrator = Orchestrator(
                    max_trials=1,
                    provider=provider,
                    api_key=api_key_value,
                    model_name=model_llm
                )
                llm_clinical_explanation = clinical_orchestrator.generate_explanation(
                    text_snippet=text,
                    predicted_class=predicted_class,
                    predicted_class_name=predicted_class_name,
                    keywords=top_keywords
                )
                # Clean up any escaped newlines
                llm_clinical_explanation = llm_clinical_explanation.replace('\\n', '\n')
                print(f"[Predict] LLM clinical explanation generated")
            else:
                print("[Predict] No LLM API key, using fallback explanation")
                keyword_list = '\n'.join([f'- {kw}' for kw in top_keywords[:8]]) if top_keywords else '- No keywords extracted'
                llm_clinical_explanation = (
                    f"## Clinical Analysis: {predicted_class_name}\n\n"
                    f"**Key Clinical Indicators Identified by XAI:**\n"
                    f"{keyword_list}\n\n"
                    f"These findings are consistent with **{predicted_class_name}**. "
                    f"The pattern of symptoms and clinical features supports this diagnosis.\n\n"
                    f"**Confidence:** {confidence:.2%}\n\n"
                    f"*Note: This is an automated analysis. Clinical correlation required for final diagnosis.*"
                )
        except Exception as e:
            print(f"[Predict] LLM explanation failed: {e}")
            llm_clinical_explanation = (
                f"## Clinical Analysis: {predicted_class_name}\n\n"
                f"The AI model has classified this as **{predicted_class_name}** "
                f"with **{confidence:.2%}** confidence.\n\n"
                f"*Clinical correlation recommended.*"
            )
        
        return {
            "text_snippet": text[:100] + ("..." if len(text) > 100 else ""),
            "predicted_class": predicted_class,
            "predicted_class_name": predicted_class_name,
            "confidence_score": f"{confidence:.2%}",
            "entropy": f"{entropy:.4f}",
            "uncertainty_status": uncertainty_msg,
            "class_probabilities": probs.cpu().numpy().tolist()[0],
            "word_attributions": word_attributions,
            "top_influential_words": top_keywords,
            "llm_clinical_explanation": llm_clinical_explanation,
            "model_info": {
                "model_name": model_name,
                "strategy": strategy,
                "run_id": run_id,
                "mlflow_metadata": mlflow_metadata
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Prediction failed: {str(e)}"}


def delete_model_from_registry(run_id: str) -> bool:
    """Delete a model run from MLflow registry."""
    try:
        mlflow.delete_run(run_id)
        print(f"[MLflow] Run {run_id} deleted successfully")
        return True
    except Exception as e:
        print(f"[MLflow] Error deleting run: {e}")
        return False