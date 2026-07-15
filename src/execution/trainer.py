import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from tqdm import tqdm
from src.execution.evaluator import Evaluator
import torch.nn.functional as F
import numpy as np


class ModelTrainer:
    def __init__(self, device, class_weights=None):
        self.device = device
        
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
            print(f"[Trainer] Using Weighted Cross Entropy: {class_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss()

    def train_one_epoch(self, model, loader, optimizer):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc="Training", leave=False)
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['grade_label'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            loss = self.criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': correct/total})
            
        return running_loss / len(loader), correct / total

    def evaluate(self, model, loader, return_raw=False):
        model.eval()
        all_outputs = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['grade_label'].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                all_outputs.append(logits)
                all_targets.append(labels)
        
        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)
        
        metrics = Evaluator.compute(all_outputs, all_targets, all_targets)
        metrics['val_loss'] = total_loss / len(loader)
        
        if return_raw:
            return metrics, F.softmax(all_outputs, dim=1), all_targets
            
        return metrics

    def run_training(self, model, train_loader, val_loader, config):
        print(f"[Trainer] Starting training on {self.device}...")
        lr = config.get('lr', 2e-5)
        epochs = config.get('epochs', 3)
        
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        
        mlflow.log_params({
            "learning_rate": lr,
            "epochs_actual": epochs,
            "batch_size": config.get('batch_size', 16),
            "optimizer": "AdamW"
        })
        
        best_acc = 0.0
        best_metrics = {}
        
        history = {
            'train_loss': [], 
            'train_acc': [], 
            'val_loss': [], 
            'val_acc': []
        }
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(model, train_loader, optimizer)
            val_metrics = self.evaluate(model, val_loader)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_metrics['val_loss'],
                "val_accuracy": val_metrics['accuracy'],
                "val_f1_macro": val_metrics['f1_macro'],
                "val_entropy": val_metrics['entropy'],
                "val_ece": val_metrics.get('ece', 0.0)
            }, step=epoch)
            
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_metrics['val_loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val F1: {val_metrics['f1_macro']:.4f}")
            
            if val_metrics['accuracy'] > best_acc:
                best_acc = val_metrics['accuracy']
                best_metrics = val_metrics.copy()
                
                mlflow.log_metrics({
                    "best_epoch": epoch,
                    "best_val_accuracy": best_acc
                })
        
        if not best_metrics:
            best_metrics = val_metrics
        
        best_metrics['history'] = history
        
        mlflow.log_metrics({
            "final_train_loss": history['train_loss'][-1],
            "final_val_accuracy": best_metrics['accuracy'],
            "final_val_f1": best_metrics['f1_macro']
        })
        
        print(f"[Trainer] Training Complete. Best Val Acc: {best_metrics['accuracy']:.4f}")
        
        return best_metrics