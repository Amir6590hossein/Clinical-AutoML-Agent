import json
import requests
import os
import re
from typing import Tuple, Optional, Dict, Any, List

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("[Orchestrator] Google Generative AI library not installed")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("[Orchestrator] Groq library not installed")


class Orchestrator:
    """
    LLM-powered orchestrator for AutoML hyperparameter search.
    Falls back to heuristic strategies when no API key is provided.
    """
    
    def __init__(
        self,
        max_trials: int = 10,
        provider: str = "Perplexity",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        max_epochs: int = 5
    ):
        self.max_trials = max_trials
        self.current_trial = 0
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name
        self.max_epochs = max_epochs
        self.client = None
        
        if self.api_key and self.api_key.strip():
            self._initialize_client()
        else:
            print("[Orchestrator] No API key provided. Will use heuristic strategies.")
    
    def _initialize_client(self):
        """Initialize the LLM client based on the selected provider."""
        try:
            if self.provider == "Google Gemini":
                if GEMINI_AVAILABLE:
                    genai.configure(api_key=self.api_key)
                    self.client = genai.GenerativeModel(self.model_name)
                    print(f"[Orchestrator] Gemini client initialized: {self.model_name}")
                else:
                    print("[Orchestrator] Gemini library not installed. Switching to heuristic.")
                    self.api_key = None
                    
            elif self.provider == "Groq":
                if GROQ_AVAILABLE:
                    self.client = Groq(api_key=self.api_key)
                    print(f"[Orchestrator] Groq client initialized: {self.model_name}")
                else:
                    print("[Orchestrator] Groq library not installed. Switching to heuristic.")
                    self.api_key = None
                    
            elif self.provider == "Perplexity":
                print(f"[Orchestrator] Perplexity ready: {self.model_name}")
                
            else:
                print(f"[Orchestrator] Unknown provider: {self.provider}. Using heuristic.")
                self.api_key = None
                
        except Exception as e:
            print(f"[Orchestrator] Client initialization failed: {e}")
            print("[Orchestrator] Switching to heuristic mode...")
            self.api_key = None
            self.client = None

    def plan_next_trial(self, memory) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Plan the next trial configuration.
        Returns (config_dict, reasoning_string).
        """
        self.current_trial += 1
        
        if self.current_trial > self.max_trials:
            return None, "Budget exhausted"
        
        history = memory.get_history_for_orchestrator()
        
        if not self.api_key or self.api_key.isspace():
            return self._get_heuristic_plan(history)
        
        try:
            config, reasoning = self._query_llm(history)
            return config, reasoning
        except Exception as e:
            print(f"[Orchestrator] LLM query failed: {e}")
            print(f"[Orchestrator] Falling back to heuristic strategy...")
            return self._get_heuristic_plan(history)

    def _query_llm(self, history: List[Dict]) -> Tuple[Dict[str, Any], str]:
        """Send a request to the LLM for hyperparameter suggestions."""
        
        system_prompt = (
            "You are an expert AutoML Orchestrator for Medical Text Classification (NLP). "
            "Your goal is to maximize validation accuracy with a limited budget. "
            "\n\nAvailable models: ['bert', 'biobert', 'roberta']"
            "\nTuning strategies: ['head_only', 'lora', 'adapter', 'full_ft']"
            f"\n\n*** CRITICAL: Maximum epochs allowed = {self.max_epochs}."
            "\n  DO NOT always use the maximum! Be smart about epoch allocation:"
            "\n    - 'head_only': 1-3 epochs is enough (only classifier head trains)"
            "\n    - 'lora': 2-4 epochs sufficient (parameter-efficient)"
            "\n    - 'adapter': 2-4 epochs sufficient (similar to LoRA)"
            f"\n    - 'full_ft': Can use up to {self.max_epochs} epochs (full model training)"
            "\n  SAVE epochs for better strategies. Don't waste on simple methods!"
            "\n\nReturn ONLY a valid JSON object with these exact keys: "
            "'model_name', 'strategy', 'batch_size', 'lr', 'epochs', 'reasoning'"
            "\n\nGOOD examples:"
            '\n  {"model_name": "bert", "strategy": "head_only", "batch_size": 32, "lr": 1e-3, "epochs": 2, "reasoning": "Quick baseline, head_only only needs 2 epochs"}'
            '\n  {"model_name": "biobert", "strategy": "lora", "batch_size": 16, "lr": 2e-4, "epochs": 3, "reasoning": "LoRA is efficient, 3 epochs is plenty"}'
            f'\n  {{"model_name": "roberta", "strategy": "full_ft", "batch_size": 8, "lr": 2e-5, "epochs": {self.max_epochs}, "reasoning": "Full fine-tuning benefits from all available epochs"}}'
            "\n\nBAD example (wasting budget):"
            f'\n  {{"model_name": "bert", "strategy": "head_only", "batch_size": 16, "lr": 1e-3, "epochs": {self.max_epochs}, "reasoning": "Using max epochs"}}  <-- TOO MANY for head_only!'
            "\n\nNo markdown, no extra text. Only JSON."
        )
        
        history_summary = self._summarize_history(history)
        user_message = (
            f"*** Trial {self.current_trial}/{self.max_trials}\n"
            f"*** Remaining budget: {self.max_trials - self.current_trial + 1} trials\n\n"
            f"Previous results:\n{json.dumps(history_summary, indent=2)}\n\n"
            f"*** HARD CONSTRAINTS:"
            f"\n  - MAX EPOCHS: {self.max_epochs} (use less for simple strategies!)"
            f"\n  - GPU VRAM: ~16GB"
            f"\n  - Goal: Maximize accuracy, NOT epoch count"
            f"\n\n*** Suggest a smart config. Don't waste epochs on head_only/lora/adapter!"
        )
        
        result_text = ""
        
        try:
            if self.provider == "Google Gemini":
                if not self.client:
                    raise ValueError("Gemini client not initialized")
                full_prompt = system_prompt + "\n\n" + user_message
                response = self.client.generate_content(full_prompt)
                result_text = response.text
                
            elif self.provider == "Groq":
                if not self.client:
                    raise ValueError("Groq client not initialized")
                chat = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    model=self.model_name,
                    temperature=0.2,
                )
                result_text = chat.choices[0].message.content
                
            elif self.provider == "Perplexity":
                url = "https://api.perplexity.ai/chat/completions"
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ]
                }
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                response = requests.post(url, json=payload, headers=headers, timeout=30)
                response.raise_for_status()
                result_text = response.json()['choices'][0]['message']['content']
            
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        
        except requests.exceptions.Timeout:
            raise Exception("LLM API timeout")
        except requests.exceptions.RequestException as e:
            raise Exception(f"LLM API request failed: {e}")
        except Exception as e:
            raise Exception(f"LLM query error: {e}")
        
        if not result_text:
            raise ValueError("Empty response from LLM")
        
        # Parse JSON from response
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        try:
            config = json.loads(result_text)
        except json.JSONDecodeError:
            # Fixed: use raw string for regex
            json_match = re.compile(r'\{.*\}', re.DOTALL).search(result_text)
            if json_match:
                try:
                    config = json.loads(json_match.group())
                except:
                    raise ValueError(f"Could not parse JSON from LLM response: {result_text[:200]}")
            else:
                raise ValueError(f"No JSON found in LLM response: {result_text[:200]}")
        
        reasoning = config.pop('reasoning', 'AI generated strategy based on history analysis.')
        
        config['batch_size'] = int(config.get('batch_size', 16))
        config['epochs'] = int(config.get('epochs', 3))
        config['lr'] = float(config.get('lr', 2e-5))
        
        # SMART EPOCH ADJUSTMENT
        if config['strategy'] == 'head_only' and config['epochs'] > 3:
            print(f"[Orchestrator] Reducing epochs from {config['epochs']} to 3 (head_only)")
            config['epochs'] = 3
        elif config['strategy'] in ['lora', 'adapter'] and config['epochs'] > 4:
            print(f"[Orchestrator] Reducing epochs from {config['epochs']} to 4 ({config['strategy']})")
            config['epochs'] = 4
        elif config['epochs'] > self.max_epochs:
            print(f"[Orchestrator] Capping epochs from {config['epochs']} to {self.max_epochs}")
            config['epochs'] = self.max_epochs
        
        valid_models = ['bert', 'biobert', 'roberta']
        if config['model_name'] not in valid_models:
            print(f"[Orchestrator] Invalid model '{config['model_name']}', using 'bert'")
            config['model_name'] = 'bert'
        
        valid_strategies = ['head_only', 'lora', 'adapter', 'full_ft']
        if config['strategy'] not in valid_strategies:
            print(f"[Orchestrator] Invalid strategy '{config['strategy']}', using 'head_only'")
            config['strategy'] = 'head_only'
        
        return config, reasoning

    def _summarize_history(self, history: List[Dict]) -> List[Dict]:
        """Summarize trial history for LLM context (last 5 trials only)."""
        if not history:
            return []
        
        summary = []
        for h in history[-5:]:
            summary.append({
                'trial': h.get('trial_id', '?'),
                'model': h.get('config', {}).get('model_name', '?'),
                'strategy': h.get('config', {}).get('strategy', '?'),
                'epochs_used': h.get('config', {}).get('epochs', '?'),
                'accuracy': round(h.get('metrics', {}).get('accuracy', 0), 4),
                'f1': round(h.get('metrics', {}).get('f1_macro', 0), 4)
            })
        return summary

    def _get_heuristic_plan(self, history: List[Dict]) -> Tuple[Dict[str, Any], str]:
        """
        Heuristic strategy for hyperparameter selection.
        Used when LLM is unavailable.
        """
        if self.current_trial == 1:
            return {
                'model_name': 'bert',
                'strategy': 'head_only',
                'batch_size': 32,
                'lr': 1e-3,
                'epochs': min(2, self.max_epochs)
            }, "Baseline: BERT head-only (2 epochs - Heuristic)"
        
        if self.current_trial == 2:
            return {
                'model_name': 'biobert',
                'strategy': 'lora',
                'batch_size': 16,
                'lr': 2e-4,
                'epochs': min(3, self.max_epochs)
            }, "Exploration: BioBERT LoRA (3 epochs - Heuristic)"
        
        if self.current_trial == 3:
            return {
                'model_name': 'roberta',
                'strategy': 'adapter',
                'batch_size': 16,
                'lr': 2e-4,
                'epochs': min(3, self.max_epochs)
            }, "Exploration: RoBERTa Adapter (3 epochs - Heuristic)"
        
        if history:
            try:
                sorted_history = sorted(
                    history,
                    key=lambda x: x.get('metrics', {}).get('accuracy', 0),
                    reverse=True
                )
                best = sorted_history[0]
                config = best.get('config', {}).copy()
                
                if config.get('strategy') == 'full_ft':
                    config['epochs'] = min(config.get('epochs', 3) + 1, self.max_epochs)
                else:
                    config['epochs'] = min(config.get('epochs', 3), self.max_epochs)
                
                config['lr'] = config.get('lr', 2e-5) * 0.5
                
                reasoning = f"Refining best trial {best.get('trial_id', '?')} (Heuristic)"
                return config, reasoning
                
            except Exception as e:
                print(f"[Orchestrator] Heuristic refinement failed: {e}")
        
        return {
            'model_name': 'bert',
            'strategy': 'full_ft',
            'batch_size': 8,
            'lr': 2e-5,
            'epochs': self.max_epochs
        }, "Fallback: BERT full fine-tuning (Heuristic)"

    def generate_explanation(
        self,
        text_snippet: str,
        predicted_class: int,
        predicted_class_name: str = "",
        keywords: List[str] = None
    ) -> str:
        """
        Generate a medical explanation for the model's prediction.
        
        Args:
            text_snippet: Patient symptoms text
            predicted_class: Numeric class ID
            predicted_class_name: Human-readable disease name
            keywords: Important clinical terms from XAI analysis
        
        Returns:
            Formatted medical explanation string
        """
        if keywords is None:
            keywords = []
        
        if not self.api_key or not self.api_key.strip():
            disease_name = predicted_class_name or f"Class {predicted_class}"
            return (
                f"AI Diagnosis: {disease_name}\\n\\n"
                f"Key clinical indicators identified: {', '.join(keywords[:5]) if keywords else 'N/A'}\\n\\n"
                f"(Connect an LLM API key for detailed medical explanations.)"
            )
        
        # Enhanced system prompt with dermatology context
        system_prompt = (
            "You are an expert Dermatologist and Clinical AI Consultant specializing in skin diseases. "
            "Your task is to explain a Deep Learning model's diagnosis for the Symptom2Disease dataset.\\n\\n"
            "**Guidelines:**\\n"
            "1. Be concise, professional, and evidence-based\\n"
            "2. Explain WHY specific symptoms support the diagnosis\\n"
            "3. Mention typical clinical presentation of the predicted disease\\n"
            "4. If applicable, mention common differential diagnoses\\n"
            "5. Use 3-5 sentences maximum\\n"
            "6. Focus on clinical reasoning, not technical ML details"
        )
        
        # Enhanced user message with disease name
        disease_display = predicted_class_name or f"Class {predicted_class}"
        
        user_message = (
            f'**Patient Case:** "{text_snippet}"\n\n'
            f"**AI Predicted Diagnosis:** {disease_display} (Class {predicted_class})\\n\\n"
            f"**Key Clinical Indicators Identified by XAI:**\\n"
            f"{', '.join(keywords[:8]) if keywords else 'No specific keywords extracted'}\\n\\n"
            f"**Please explain:**\\n"
            f"1. Why do these specific symptoms/terms support the diagnosis of {disease_display}?\\n"
            f"2. What is the typical clinical presentation of {disease_display}?\\n"
            f"3. What other conditions might present similarly (differential diagnoses)?\\n\\n"
            f"*Provide a clear, medical explanation suitable for healthcare professionals.*"
        )
        
        explanation = "Could not generate clinical explanation."
        
        try:
            if self.provider == "Google Gemini" and self.client:
                full_prompt = system_prompt + "\\n\\n" + user_message
                response = self.client.generate_content(full_prompt)
                explanation = response.text
                
            elif self.provider == "Groq" and self.client:
                chat = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    model=self.model_name,
                    temperature=0.3,
                    max_tokens=300
                )
                explanation = chat.choices[0].message.content
                
            elif self.provider == "Perplexity":
                url = "https://api.perplexity.ai/chat/completions"
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    "max_tokens": 300,
                    "temperature": 0.3
                }
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                response = requests.post(url, json=payload, headers=headers, timeout=30)
                response.raise_for_status()
                explanation = response.json()['choices'][0]['message']['content']
                
        except Exception as e:
            print(f"[Orchestrator] Explanation generation failed: {e}")
            disease_name = predicted_class_name or f"Class {predicted_class}"
            return (
                f"AI Predicted Diagnosis: {disease_name}\\n\\n"
                f"Key Clinical Indicators: {', '.join(keywords[:5]) if keywords else 'N/A'}\\n\\n"
                f"(Automated explanation unavailable. Error: {str(e)})"
            )
        
        # Format explanation with disease name header
        formatted_explanation = (
            f"## Clinical Analysis: {disease_display}\\n\\n"
            f"{explanation}\\n\\n"
            f"---\\n"
            f"*This AI-generated explanation is based on XAI-identified clinical terms: "
            f"{', '.join(keywords[:5]) if keywords else 'N/A'}*"
        )
        
        return formatted_explanation