import mlflow
import pandas as pd
from typing import Dict, Any, Optional, List
from pathlib import Path


class ModelRegistry:
    """
    MLflow Model Registry wrapper for managing trained models.
    Provides methods to query, compare, and delete model runs.
    """
    
    def __init__(self, experiment_name: str = "medical_nlp_agent"):
        self.experiment_name = experiment_name
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Configure MLflow tracking URI and experiment."""
        mlflow_tracking_dir = Path("mlruns").absolute()
        mlflow.set_tracking_uri(f"file://{mlflow_tracking_dir}")
        
        try:
            mlflow.create_experiment(self.experiment_name)
        except mlflow.exceptions.MlflowException:
            pass
        
        mlflow.set_experiment(self.experiment_name)
    
    def get_all_models(self) -> pd.DataFrame:
        """Retrieve all registered models sorted by accuracy."""
        try:
            runs = mlflow.search_runs(
                experiment_names=[self.experiment_name],
                order_by=["metrics.final_val_accuracy DESC"]
            )
        except Exception as e:
            print(f"[Registry] Error fetching runs: {e}")
            runs = pd.DataFrame()
        
        if runs.empty:
            return pd.DataFrame(columns=[
                'run_id', 'trial_id', 'model_name', 'strategy',
                'accuracy', 'f1_macro', 'val_loss', 'start_time'
            ])
        
        result_df = pd.DataFrame({
            'run_id': runs['run_id'],
            'trial_id': runs.get('params.trial_id', 'N/A'),
            'model_name': runs.get('params.model_name', 'N/A'),
            'strategy': runs.get('params.strategy', 'N/A'),
            'accuracy': runs.get('metrics.final_val_accuracy', 0.0).round(4),
            'f1_macro': runs.get('metrics.final_val_f1_macro', 0.0).round(4),
            'val_loss': runs.get('metrics.final_val_loss', 0.0).round(4),
            'start_time': runs.get('start_time', 'N/A'),
            'llm_reasoning': runs.get('params.reasoning', '')
        })
        
        return result_df
    
    def get_best_model(self) -> Optional[Dict[str, Any]]:
        """Return the best model based on validation accuracy."""
        try:
            runs = mlflow.search_runs(
                experiment_names=[self.experiment_name],
                order_by=["metrics.final_val_accuracy DESC"],
                max_results=1
            )
        except Exception as e:
            print(f"[Registry] Error fetching best model: {e}")
            return None
        
        if runs.empty:
            return None
        
        best_run = runs.iloc[0]
        return {
            'run_id': best_run['run_id'],
            'model_name': best_run.get('params.model_name', 'unknown'),
            'strategy': best_run.get('params.strategy', 'unknown'),
            'accuracy': best_run.get('metrics.final_val_accuracy', 0.0),
            'f1_macro': best_run.get('metrics.final_val_f1_macro', 0.0)
        }
    
    def get_run_details(self, run_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific run."""
        try:
            run = mlflow.get_run(run_id)
            params = run.data.params
            metrics = run.data.metrics
            
            return {
                'run_id': run_id,
                'params': dict(params),
                'metrics': dict(metrics),
                'artifacts': [a.path for a in mlflow.list_artifacts(run_id)]
            }
        except Exception as e:
            print(f"[Registry] Error getting run details for {run_id}: {e}")
            return {'run_id': run_id, 'error': str(e)}
    
    def delete_run(self, run_id: str) -> bool:
        """Delete a model run from the registry."""
        try:
            mlflow.delete_run(run_id)
            print(f"[Registry] Run {run_id} deleted successfully")
            return True
        except Exception as e:
            print(f"[Registry] Error deleting run {run_id}: {e}")
            return False
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """Compare multiple runs side by side."""
        all_runs = self.get_all_models()
        if all_runs.empty:
            return all_runs
        return all_runs[all_runs['run_id'].isin(run_ids)]