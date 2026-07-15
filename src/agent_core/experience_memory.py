import mlflow
import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime


class ExperienceMemory:
    """
    Memory module that tracks all AutoML trials.
    Stores trial configurations, metrics, and MLflow run IDs.
    Provides methods to query best trials and format history for the orchestrator.
    """
    
    def __init__(self, experiment_name: str = "medical_nlp_agent"):
        self.experiment_name = experiment_name
        self.history = []
        
    def add_experience(self, trial_id: int, config: Dict[str, Any], 
                       metrics: Dict[str, Any], run_id: Optional[str] = None):
        """Record a completed trial."""
        entry = {
            'trial_id': trial_id,
            'config': config,
            'metrics': metrics,
            'run_id': run_id or (mlflow.active_run().info.run_id if mlflow.active_run() else None),
            'timestamp': datetime.now().isoformat()
        }
        self.history.append(entry)
        print(f"[Memory] Trial {trial_id} recorded | "
              f"Acc: {metrics.get('accuracy', 0):.4f} | "
              f"Run ID: {entry['run_id']}")

    def get_best_trial(self) -> Optional[Dict[str, Any]]:
        """Return the trial with the highest accuracy."""
        if not self.history:
            return self._get_best_from_mlflow()
        
        sorted_history = sorted(
            self.history, 
            key=lambda x: x['metrics'].get('accuracy', 0), 
            reverse=True
        )
        return sorted_history[0]

    def get_top_k_trials(self, k: int = 3) -> List[Dict[str, Any]]:
        """Return the top K trials by accuracy."""
        if not self.history:
            return self._get_top_k_from_mlflow(k)
        
        sorted_history = sorted(
            self.history, 
            key=lambda x: x['metrics'].get('accuracy', 0), 
            reverse=True
        )
        return sorted_history[:min(k, len(sorted_history))]

    def get_all_trials_from_mlflow(self) -> pd.DataFrame:
        """Query all trials from MLflow tracking server."""
        try:
            runs_df = mlflow.search_runs(
                experiment_names=[self.experiment_name],
                order_by=["metrics.final_val_accuracy DESC"]
            )
            return runs_df
        except Exception as e:
            print(f"[Memory] Error querying MLflow: {e}")
            return pd.DataFrame()

    def get_history_for_orchestrator(self) -> List[Dict[str, Any]]:
        """
        Format trial history for the LLM orchestrator.
        Returns a simplified list of trial configs and metrics.
        """
        # Use in-memory history if available
        if self.history:
            return [
                {
                    'trial_id': h['trial_id'],
                    'config': h['config'],
                    'metrics': {
                        'accuracy': h['metrics'].get('accuracy', 0),
                        'f1_macro': h['metrics'].get('f1_macro', 0),
                        'val_loss': h['metrics'].get('val_loss', 0)
                    }
                }
                for h in self.history
            ]
        
        # Fallback: fetch from MLflow
        try:
            runs_df = mlflow.search_runs(
                experiment_names=[self.experiment_name],
                order_by=["start_time DESC"]
            )
            
            if runs_df.empty:
                return []
            
            history = []
            for _, run in runs_df.iterrows():
                trial_id = run.get('params.trial_id', 'N/A')
                config = {
                    'model_name': run.get('params.model_name', 'unknown'),
                    'strategy': run.get('params.strategy', 'unknown'),
                    'batch_size': run.get('params.batch_size', 16),
                    'lr': run.get('params.learning_rate', 2e-5),
                    'epochs': run.get('params.epochs_actual', 3)
                }
                metrics = {
                    'accuracy': run.get('metrics.final_val_accuracy', 0),
                    'f1_macro': run.get('metrics.final_val_f1_macro', 0),
                    'val_loss': run.get('metrics.final_val_loss', 0)
                }
                
                history.append({
                    'trial_id': trial_id,
                    'config': config,
                    'metrics': metrics
                })
            
            return history
            
        except Exception as e:
            print(f"[Memory] Error building history for orchestrator: {e}")
            return []

    def to_dataframe(self) -> pd.DataFrame:
        """Convert memory to a pandas DataFrame."""
        if not self.history:
            return self.get_all_trials_from_mlflow()
        
        data = []
        for h in self.history:
            row = {
                'trial_id': h['trial_id'],
                'run_id': h.get('run_id', 'N/A'),
                'timestamp': h.get('timestamp', '')
            }
            row.update({f"config_{k}": v for k, v in h['config'].items()})
            row.update({f"metric_{k}": v for k, v in h['metrics'].items() if k != 'history'})
            data.append(row)
        
        return pd.DataFrame(data)

    def _get_best_from_mlflow(self) -> Optional[Dict[str, Any]]:
        """Query MLflow for the best performing run."""
        try:
            runs_df = mlflow.search_runs(
                experiment_names=[self.experiment_name],
                order_by=["metrics.final_val_accuracy DESC"],
                max_results=1
            )
            
            if runs_df.empty:
                return None
            
            best_run = runs_df.iloc[0]
            return {
                'trial_id': best_run.get('params.trial_id', 'N/A'),
                'run_id': best_run['run_id'],
                'config': {
                    'model_name': best_run.get('params.model_name', 'unknown'),
                    'strategy': best_run.get('params.strategy', 'unknown'),
                },
                'metrics': {
                    'accuracy': best_run.get('metrics.final_val_accuracy', 0),
                    'f1_macro': best_run.get('metrics.final_val_f1_macro', 0)
                }
            }
        except Exception as e:
            print(f"[Memory] Error getting best from MLflow: {e}")
            return None

    def _get_top_k_from_mlflow(self, k: int = 3) -> List[Dict[str, Any]]:
        """Query MLflow for the top K runs."""
        try:
            runs_df = mlflow.search_runs(
                experiment_names=[self.experiment_name],
                order_by=["metrics.final_val_accuracy DESC"],
                max_results=k
            )
            
            if runs_df.empty:
                return []
            
            top_trials = []
            for _, run in runs_df.iterrows():
                top_trials.append({
                    'trial_id': run.get('params.trial_id', 'N/A'),
                    'run_id': run['run_id'],
                    'config': {
                        'model_name': run.get('params.model_name', 'unknown'),
                        'strategy': run.get('params.strategy', 'unknown'),
                    },
                    'metrics': {
                        'accuracy': run.get('metrics.final_val_accuracy', 0)
                    }
                })
            
            return top_trials
            
        except Exception as e:
            print(f"[Memory] Error getting top-k from MLflow: {e}")
            return []

    def save_memory(self, path: str = 'experiments/results.json'):
        """Deprecated: MLflow handles persistence automatically."""
        print("[Memory] save_memory() is deprecated. MLflow handles persistence automatically.")
        pass