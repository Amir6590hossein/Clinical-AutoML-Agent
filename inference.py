import os
import sys
import argparse

sys.path.append(os.getcwd())

from src.execution.pipeline_services import predict_pipeline, get_model_registry


def list_available_models():
    """Display all registered models in the MLflow registry."""
    print("Available Models in MLflow Registry:")
    print("-" * 80)
    try:
        registry_df = get_model_registry()
        if registry_df.empty:
            print("No models found. Run training first.")
        else:
            for _, row in registry_df.iterrows():
                print(f"Run ID: {row['run_id']}")
                print(f"  Model: {row['model_name']} | Strategy: {row['strategy']}")
                print(f"  Accuracy: {row['accuracy']:.4f} | F1: {row['f1_macro']:.4f}")
                print(f"  Date: {row['start_time']}")
                print("-" * 80)
    except Exception as e:
        print(f"Error loading registry: {e}")


def get_best_run_id():
    """Find the best model run ID from the registry."""
    try:
        registry_df = get_model_registry()
        if registry_df.empty:
            print("No models found. Run training first.")
            return None
        
        best_run_id = registry_df.iloc[0]['run_id']
        best_model = registry_df.iloc[0]['model_name']
        best_acc = registry_df.iloc[0]['accuracy']
        
        print(f"Using best model: {best_run_id}")
        print(f"  Model: {best_model} | Accuracy: {best_acc:.4f}")
        
        return best_run_id
    except Exception as e:
        print(f"Error finding best model: {e}")
        return None


def find_local_model_path():
    """Find the latest local model trial folder."""
    from pathlib import Path
    import json
    
    model_folders = sorted(Path("experiments").glob("model_trial_*"))
    if not model_folders:
        return None
    
    # Return the most recent model folder
    latest_folder = model_folders[-1]
    
    # Verify config exists
    config_file = latest_folder / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        print(f"Found local model: {latest_folder}")
        print(f"  Model: {config.get('model_name', 'N/A').upper()} | Strategy: {config.get('strategy', 'N/A')}")
        print(f"  Accuracy: {config.get('metrics', {}).get('accuracy', 0):.4f}")
        return str(latest_folder)
    
    return None


def display_prediction_result(result):
    """Pretty-print prediction results with XAI and LLM clinical explanation."""
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    print("
" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    print(f"Input Snippet: {result['text_snippet']}")
    print(f"Predicted Disease: {result['predicted_class_name']} (Class {result['predicted_class']})")
    print(f"Confidence: {result['confidence_score']}")
    print(f"Uncertainty: {result['uncertainty_status']} (Entropy: {result['entropy']})")
    
    # XAI Word Attributions
    if result.get('word_attributions'):
        print("-" * 70)
        print("XAI KEY CLINICAL INDICATORS:")
        print("-" * 70)
        for word, score in result['word_attributions'][:15]:
            bar_length = int(abs(score) * 40)
            bar = "#" * bar_length + "-" * (40 - bar_length)
            direction = "+" if score > 0 else "-"
            print(f"  {word.ljust(20)} [{direction}] {bar} ({score:.4f})")
    
    # Class Probabilities (Top 5)
    if result.get('class_probabilities'):
        print("-" * 70)
        print("TOP 5 DISEASE PROBABILITIES:")
        print("-" * 70)
        
        # Map class indices to names
        from src.config import get_class_name
        
        class_probs = result['class_probabilities']
        indexed_probs = [(i, prob) for i, prob in enumerate(class_probs)]
        sorted_probs = sorted(indexed_probs, key=lambda x: x[1], reverse=True)[:5]
        
        for class_idx, prob in sorted_probs:
            class_name = get_class_name(class_idx)
            bar_len = int(prob * 50)
            bar = "#" * bar_len + "-" * (50 - bar_len)
            print(f"  {class_name.ljust(35)} {bar} ({prob:.2%})")
    
    # LLM Clinical Explanation
    if result.get('llm_clinical_explanation') and result['llm_clinical_explanation'].strip():
        print("
" + "=" * 70)
        print("AI CLINICAL ANALYSIS (LLM + XAI)")
        print("=" * 70)
        print(result['llm_clinical_explanation'])
    else:
        print("
" + "-" * 70)
        print("Clinical explanation not available. To enable, provide an LLM API key via:")
        print("  --llm-api-key parameter or LLM_API_KEY environment variable")
    
    # Model Info
    print("
" + "-" * 70)
    print(f"Model Info: {result['model_info']['model_name'].upper()} | {result['model_info']['strategy']}")
    if result['model_info'].get('mlflow_metadata'):
        meta = result['model_info']['mlflow_metadata']
        print(f"MLflow Run ID: {meta.get('run_id', 'N/A')}")
        print(f"MLflow URL: {meta.get('url', 'N/A')}")
    print("-" * 70 + "
")


def main():
    parser = argparse.ArgumentParser(
        description="Medical Text Inference Tool with XAI and LLM Clinical Reasoning"
    )
    parser.add_argument(
        "--text", type=str, required=True,
        help="Input medical text to classify"
    )
    parser.add_argument(
        "--run_id", type=str, default=None,
        help="MLflow Run ID (uses best model if not specified)"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Local model path (e.g., experiments/model_trial_1)"
    )
    parser.add_argument(
        "--list_models", action="store_true",
        help="List all available models"
    )
    parser.add_argument(
        "--llm-provider", type=str, default=None,
        choices=["Perplexity", "Google Gemini", "Groq"],
        help="LLM provider for clinical explanation (default: from env or Perplexity)"
    )
    parser.add_argument(
        "--llm-api-key", type=str, default=None,
        help="LLM API key for clinical explanation (default: from LLM_API_KEY env var)"
    )
    parser.add_argument(
        "--llm-model", type=str, default=None,
        help="LLM model name (default: from env or sonar-pro)"
    )
    
    args = parser.parse_args()
    
    # Handle --list_models flag
    if args.list_models:
        list_available_models()
        return
    
    # Determine which model to use
    model_path = None
    
    if args.model_path:
        # Use explicitly provided local model path
        if os.path.exists(args.model_path):
            model_path = args.model_path
            print(f"Using provided model path: {model_path}")
        else:
            print(f"Error: Model path not found: {args.model_path}")
            return
    elif args.run_id:
        # Use MLflow run ID
        print(f"Using MLflow Run ID: {args.run_id}")
        # For now, MLflow runs are identified by run_id in local config
        local_models = sorted(Path("experiments").glob("model_trial_*"))
        found = False
        import json
        for folder in local_models:
            config_file = folder / "config.json"
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)
                if config.get("run_id") == args.run_id:
                    model_path = str(folder)
                    found = True
                    break
        if not found:
            print(f"Error: No local model found with run_id: {args.run_id}")
            print("Try using --model_path instead, or omit --run_id to use best model")
            return
    else:
        # Find best available model
        print("No model specified. Searching for best available model...")
        
        # Try MLflow first
        run_id = get_best_run_id()
        if run_id:
            # Find corresponding local model
            import json
            local_models = sorted(Path("experiments").glob("model_trial_*"))
            for folder in local_models:
                config_file = folder / "config.json"
                if config_file.exists():
                    with open(config_file) as f:
                        config = json.load(f)
                    if config.get("run_id") == run_id:
                        model_path = str(folder)
                        print(f"Found local model: {model_path}")
                        break
        
        # Fallback to local model if no MLflow match
        if not model_path:
            model_path = find_local_model_path()
        
        if not model_path:
            print("Error: No models found. Run training first.")
            return
    
    # Set LLM credentials from args or environment
    llm_provider = args.llm_provider or os.environ.get("LLM_PROVIDER", "Perplexity")
    llm_api_key = args.llm_api_key or os.environ.get("LLM_API_KEY", "")
    llm_model = args.llm_model or os.environ.get("LLM_MODEL", "sonar-pro")
    
    if llm_api_key:
        print(f"LLM Clinical Explanation: Enabled ({llm_provider}/{llm_model})")
    else:
        print("LLM Clinical Explanation: Disabled (no API key provided)")
        print("  Set --llm-api-key or LLM_API_KEY environment variable to enable")
    
    print("-" * 70)
    
    try:
        result = predict_pipeline(
            text=args.text,
            model_path=model_path,
            llm_provider=llm_provider,
            llm_api_key=llm_api_key if llm_api_key else None,
            llm_model_name=llm_model if llm_api_key else None
        )
        display_prediction_result(result)
    except Exception as e:
        print(f"Critical Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    from pathlib import Path
    main()