import pandas as pd
import numpy as np
import torch
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

class AdvancedEvaluator:
    def __init__(self, test_data_path=None):
        """Initialize Advanced Evaluator"""
        self.test_data_path = test_data_path
        self.test_data = None
        self.text_column = None
        self.label_column = None
        self.models = {}
        self.tokenizers = {}
        self.label_encoder = LabelEncoder()
        self.label_mapping = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Auto-detect and load test data
        if test_data_path is None:
            self.test_data_path = self._find_test_data()
        
        if self.test_data_path:
            self.load_test_data()
    
    def _find_test_data(self):
        """Auto-detect test data file location"""
        possible_paths = [
            "sentiment_ablation/data/dataset_tweet_sentimen_tayangan_tv.csv",
            "sentiment_ablation/data/dataset_tweet_sentiment_opini_film.csv",
            "sentiment_ablation/data/Dataset Sentimen kurikulum 2013.xlsx",
            "sentiment_ablation/data/id-tourism-sentimentanalysis.xlsx",
            "data/test.csv",
            "test.csv",
            "dataset/test.csv",
            "data/dataset.csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found sentiment data at: {path}")
                return path
        
        raise FileNotFoundError("❌ No test data file found")
    
    def _detect_columns(self):
        """Detect text and label columns automatically"""
        columns = self.test_data.columns.tolist()
        
        # Text column detection
        text_candidates = ['text', 'tweet', 'Text Tweet', 'content', 'message']
        for candidate in text_candidates:
            if candidate in columns:
                self.text_column = candidate
                break
        
        if not self.text_column:
            # Use first string column
            for col in columns:
                if self.test_data[col].dtype == 'object':
                    self.text_column = col
                    break
        
        # Label column detection
        label_candidates = ['label', 'sentiment', 'Sentiment', 'target', 'class']
        for candidate in label_candidates:
            if candidate in columns:
                self.label_column = candidate
                break
        
        if not self.label_column:
            self.label_column = columns[-1]  # Use last column as fallback
    
    def _preprocess_labels(self, labels):
        """Convert string labels to numeric and handle mixed types"""
        # Convert all to string first to handle mixed types
        labels_str = [str(label).strip().lower() for label in labels]
        
        # Define standard mapping for Indonesian sentiment
        sentiment_mapping = {
            'positif': 2,
            'positive': 2,
            'pos': 2,
            '2': 2,
            '2.0': 2,
            
            'negatif': 0,
            'negative': 0,
            'neg': 0,
            '0': 0,
            '0.0': 0,
            
            'netral': 1,
            'neutral': 1,
            'net': 1,
            '1': 1,
            '1.0': 1
        }
        
        # Convert using mapping
        numeric_labels = []
        for label in labels_str:
            if label in sentiment_mapping:
                numeric_labels.append(sentiment_mapping[label])
            else:
                # Try to convert directly to int
                try:
                    num_label = int(float(label))
                    if num_label in [0, 1, 2]:
                        numeric_labels.append(num_label)
                    else:
                        print(f"⚠️  Unknown label '{label}', defaulting to neutral (1)")
                        numeric_labels.append(1)
                except:
                    print(f"⚠️  Cannot parse label '{label}', defaulting to neutral (1)")
                    numeric_labels.append(1)
        
        return numeric_labels
    
    def _analyze_labels(self, labels):
        """Analyze label distribution for debugging"""
        unique_labels = pd.Series(labels).value_counts()
        print(f"📊 Label distribution: {dict(unique_labels)}")
        return unique_labels
    
    def load_test_data(self):
        """Load test data with label preprocessing"""
        try:
            if self.test_data_path.endswith('.xlsx'):
                self.test_data = pd.read_excel(self.test_data_path)
            else:
                self.test_data = pd.read_csv(self.test_data_path)
            
            print(f"✅ Loaded data: {len(self.test_data)} samples")
            print(f"📊 Columns: {list(self.test_data.columns)}")
            
            # Detect text and label columns
            self._detect_columns()
            
            # Clean and preprocess data
            self.test_data = self.test_data.dropna(subset=[self.text_column, self.label_column])
            
            # Analyze original labels
            print(f"🔍 Original labels analysis:")
            original_labels = self.test_data[self.label_column].tolist()
            self._analyze_labels(original_labels)
            
            # Preprocess labels to numeric
            print(f"🔄 Converting labels to numeric format...")
            numeric_labels = self._preprocess_labels(original_labels)
            self.test_data['numeric_label'] = numeric_labels
            
            # Analyze converted labels
            print(f"✅ Converted labels analysis:")
            self._analyze_labels(numeric_labels)
            
            # Update label column to use numeric version
            self.label_column = 'numeric_label'
            
            print(f"📝 Text column: {self.text_column}")
            print(f"🏷️  Label column: {self.label_column}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading test data: {str(e)}")
            return False
    
    def load_model(self, model_name, model_path):
        """Load a single model and tokenizer"""
        try:
            print(f"✅ Loading {model_name} from {model_path}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.tokenizers[model_name] = tokenizer
            
            # Load model
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, 
                local_files_only=True
            )
            model.to(self.device)
            model.eval()
            self.models[model_name] = model
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading {model_name}: {str(e)}")
            return False
    
    def predict_single_model(self, model_name, texts):
        """Make predictions using a single model"""
        try:
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
            
            predictions = []
            batch_size = 8
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Predict
                with torch.no_grad():
                    outputs = model(**inputs)
                    preds = torch.argmax(outputs.logits, dim=-1)
                    predictions.extend(preds.cpu().numpy().tolist())
            
            return predictions
            
        except Exception as e:
            print(f"❌ Error in prediction for {model_name}: {str(e)}")
            return []
    
    def evaluate_model(self, model_name, model_path):
        """Evaluate single model"""
        try:
            # Load model if not already loaded
            if model_name not in self.models:
                success = self.load_model(model_name, model_path)
                if not success:
                    return None
            
            # Get text and true labels
            texts = self.test_data[self.text_column].fillna('').astype(str).tolist()
            true_labels = self.test_data[self.label_column].tolist()
            
            # Ensure all labels are integers
            true_labels = [int(label) for label in true_labels]
            
            print(f"🔍 Evaluating {model_name} with {len(texts)} samples")
            
            # Make predictions
            predictions = self.predict_single_model(model_name, texts)
            
            if not predictions:
                return None
            
            # Ensure predictions are also integers
            predictions = [int(pred) for pred in predictions]
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, predictions)
            report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)
            
            result = {
                'accuracy': accuracy,
                'f1_score': report['weighted avg']['f1-score'],
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall']
            }
            
            print(f"✅ {model_name}: Accuracy={accuracy:.4f}, F1={result['f1_score']:.4f}")
            return result
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _majority_vote(self, predictions_list):
        """Combine predictions using majority voting"""
        try:
            predictions_array = np.array(predictions_list)
            ensemble_predictions = []
            
            for i in range(predictions_array.shape[1]):
                votes = predictions_array[:, i]
                # Get most frequent prediction
                unique, counts = np.unique(votes, return_counts=True)
                majority_pred = unique[np.argmax(counts)]
                ensemble_predictions.append(majority_pred)
            
            return ensemble_predictions
            
        except Exception as e:
            print(f"❌ Error in majority vote: {str(e)}")
            return []
    
    def evaluate_ensemble(self, model_names, model_paths, method='majority_vote'):
        """Evaluate ensemble of models"""
        try:
            # Load all models
            for name, path in zip(model_names, model_paths):
                if name not in self.models:
                    success = self.load_model(name, path)
                    if not success:
                        print(f"⚠️  Skipping {name} due to loading error")
                        continue
            
            # Filter to only loaded models
            loaded_models = [name for name in model_names if name in self.models]
            
            if len(loaded_models) < 2:
                print(f"❌ Need at least 2 models for ensemble, only {len(loaded_models)} loaded")
                return None
            
            # Get test data
            texts = self.test_data[self.text_column].fillna('').astype(str).tolist()
            true_labels = self.test_data[self.label_column].tolist()
            
            # Ensure all labels are integers
            true_labels = [int(label) for label in true_labels]
            
            print(f"🔍 Ensemble evaluation with {len(loaded_models)} models: {loaded_models}")
            
            # Get predictions from all loaded models
            all_predictions = []
            for model_name in loaded_models:
                preds = self.predict_single_model(model_name, texts)
                if preds:
                    preds = [int(pred) for pred in preds]
                    all_predictions.append(preds)
            
            if not all_predictions:
                raise ValueError("No valid predictions from ensemble models")
            
            # Combine predictions
            ensemble_predictions = self._majority_vote(all_predictions)
            
            if not ensemble_predictions:
                return None
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, ensemble_predictions)
            report = classification_report(true_labels, ensemble_predictions, output_dict=True, zero_division=0)
            
            result = {
                'accuracy': accuracy,
                'f1_score': report['weighted avg']['f1-score'],
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall']
            }
            
            print(f"✅ Ensemble ({method}): Accuracy={accuracy:.4f}, F1={result['f1_score']:.4f}")
            return result
            
        except Exception as e:
            print(f"❌ Error evaluating ensemble: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def validate_model_paths(self):
        """Validate and list available model paths"""
        base_dir = "models"
        
        print("🔍 Scanning for available models...")
        available_models = {}
        
        if os.path.exists(base_dir):
            model_mapping = {
                'BERT': 'bert/final',
                'DistilBERT': 'distilbert/final', 
                'IndoBERT-Lite': 'indobertlite/final',
                'IndoBERTweet': 'indobertweet/final'
            }
            
            for model_name, relative_path in model_mapping.items():
                model_path = os.path.join(base_dir, relative_path)
                
                if os.path.exists(model_path):
                    # Check for required files
                    has_config = os.path.exists(os.path.join(model_path, 'config.json'))
                    has_model = (
                        os.path.exists(os.path.join(model_path, 'model.safetensors')) or 
                        os.path.exists(os.path.join(model_path, 'pytorch_model.bin'))
                    )
                    has_tokenizer = (
                        os.path.exists(os.path.join(model_path, 'tokenizer.json')) or
                        os.path.exists(os.path.join(model_path, 'tokenizer_config.json'))
                    )
                    
                    if has_config and has_model and has_tokenizer:
                        available_models[model_name] = model_path
                        print(f"✅ Found model: {model_name} at {model_path}")
                    else:
                        print(f"⚠️  Incomplete model: {model_name} at {model_path}")
                else:
                    print(f"❌ Model path not found: {model_path}")
        
        return available_models
    
    def evaluate_individual_models(self):
        """Evaluate all individual models"""
        available_models = self.validate_model_paths()
        
        results = {}
        print("📊 Evaluating Individual Models...")
        
        for model_name, model_path in available_models.items():
            result = self.evaluate_model(model_name, model_path)
            if result:
                results[model_name] = result
            else:
                # Create dummy result to prevent crash
                results[model_name] = {
                    'accuracy': 0.0,
                    'f1_score': 0.0,
                    'precision': 0.0,
                    'recall': 0.0
                }
        
        return results
    
    def evaluate_ensemble_methods(self):
        """Evaluate different ensemble methods"""
        available_models = self.validate_model_paths()
        
        results = {}
        
        if len(available_models) >= 2:
            print(f"🔄 Evaluating Ensemble with {len(available_models)} models...")
            
            model_names = list(available_models.keys())
            model_paths = list(available_models.values())
            
            # Majority vote ensemble
            ensemble_result = self.evaluate_ensemble(model_names, model_paths, 'majority_vote')
            if ensemble_result:
                results['Ensemble_All'] = ensemble_result
            
        else:
            print(f"⚠️  Only {len(available_models)} models available for ensemble (need ≥2)")
        
        return results
    
    def compare_models(self):
        """Compare all models and return results"""
        try:
            # Get individual results
            individual_results = self.evaluate_individual_models()
            
            # Get ensemble results
            ensemble_results = self.evaluate_ensemble_methods()
            
            # Combine all results
            all_results = {**individual_results, **ensemble_results}
            
            # Create comparison DataFrame
            if all_results:
                df_results = pd.DataFrame(all_results).T
                df_results = df_results.round(4)
                
                print("\n=== MODEL COMPARISON RESULTS ===")
                print(df_results.to_string())
                
                # Save results
                df_results.to_csv("model_comparison_results.csv")
                print(f"\n✅ Results saved to: model_comparison_results.csv")
                
                # Find best performers
                if not df_results.empty:
                    best_accuracy = df_results['accuracy'].max()
                    best_model = df_results['accuracy'].idxmax()
                    best_f1 = df_results['f1_score'].max()
                    best_f1_model = df_results['f1_score'].idxmax()
                    
                    print(f"\n=== BEST PERFORMERS ===")
                    print(f"🏆 Best Accuracy: {best_accuracy:.4f} ({best_model})")
                    print(f"🏆 Best F1-Score: {best_f1:.4f} ({best_f1_model})")
                
                return df_results
            else:
                print("❌ No results to compare")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error in model comparison: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

if __name__ == "__main__":
    # Test the evaluator
    evaluator = AdvancedEvaluator()
    results = evaluator.compare_models()
    print("Evaluation completed!")