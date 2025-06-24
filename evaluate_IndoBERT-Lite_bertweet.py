import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizerFast
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils import load_and_clean_all_datasets, safe_f1_score
from datasets import Dataset
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import os

print("🔧 PyTorch version:", torch.__version__)
print("🚀 CUDA available:", torch.cuda.is_available())
print("🖥️  Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

# === PENGATURAN BOBOT ENSEMBLE ===
ALPHA = 0.2  # bobot IndoBERT Lite, sisanya IndoBERTweet

output_folder = 'resultevaluateindobert'
os.makedirs(output_folder, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_names = ['Negative', 'Neutral', 'Positive']

# === LOAD DATASET ===
print("📂 Loading and cleaning datasets...")
df = load_and_clean_all_datasets()
print("✅ Total data after cleaning:", len(df))

# Create dataset and split (use same split as other evaluations)
dataset = Dataset.from_pandas(df[['text', 'label']])
dataset = dataset.train_test_split(test_size=0.2, seed=42)
test_dataset = dataset['test']

print("📊 Test dataset size:", len(test_dataset))

# Extract texts and labels
texts = [example['text'] for example in test_dataset]
true_labels = [example['label'] for example in test_dataset]

# === FUNGSI PREDIKSI ENSEMBLE ===
def ensemble_prediction(texts, model_lite, tokenizer_lite, model_bertweet, tokenizer_bertweet, alpha=0.5):
    """Ensemble prediction with proper error handling"""
    predictions = []
    probabilities = []
    
    model_lite.eval()
    model_bertweet.eval()

    for text in tqdm(texts, desc="🔍 Ensemble Inference"):
        try:
            # Tokenize for both models
            inputs_lite = tokenizer_lite(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs_bertweet = tokenizer_bertweet(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            
            # Move to device
            inputs_lite = {k: v.to(device) for k, v in inputs_lite.items()}
            inputs_bertweet = {k: v.to(device) for k, v in inputs_bertweet.items()}

            with torch.no_grad():
                # Get logits from both models
                logits_lite = model_lite(**inputs_lite).logits
                logits_bertweet = model_bertweet(**inputs_bertweet).logits

            # Ensemble average
            avg_logits = alpha * logits_lite + (1 - alpha) * logits_bertweet
            
            # Get probabilities and prediction
            probs = torch.softmax(avg_logits, dim=1).cpu().numpy()[0]
            pred = torch.argmax(avg_logits, dim=1).cpu().item()
            
            predictions.append(pred)
            probabilities.append(probs)
            
        except Exception as e:
            print(f"Error processing text: {e}")
            # Default to neutral in case of error
            predictions.append(1)
            probabilities.append([0.33, 0.34, 0.33])

    return np.array(predictions), np.array(probabilities)

def load_model_safely(model_path, tokenizer_class, model_class, model_name):
    """Load model with error handling"""
    if not os.path.exists(model_path):
        print(f"❌ {model_name} not found at {model_path}")
        return None, None
    
    try:
        print(f"🔧 Loading {model_name} from {model_path}...")
        tokenizer = tokenizer_class.from_pretrained(model_path)
        model = model_class.from_pretrained(model_path)
        model.to(device)
        print(f"✅ {model_name} loaded successfully")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        return None, None

# === LOAD MODELS ===
print("\n" + "="*60)
print("🤖 LOADING MODELS")
print("="*60)

# Load IndoBERT Lite
model_lite, tokenizer_lite = load_model_safely(
    "models/indobertlite/final",
    BertTokenizerFast,
    AutoModelForSequenceClassification,
    "IndoBERT Lite"
)

# Load IndoBERTweet  
model_bertweet, tokenizer_bertweet = load_model_safely(
    "models/indobertweet/final",
    BertTokenizerFast,
    BertForSequenceClassification,
    "IndoBERTweet"
)

# Check if both models loaded successfully
if model_lite is None or model_bertweet is None:
    print("❌ Could not load both models. Exiting...")
    exit(1)

# === INDIVIDUAL MODEL PREDICTIONS ===
print("\n" + "="*60)
print("🔍 GETTING INDIVIDUAL PREDICTIONS")
print("="*60)

def get_single_model_predictions(model, tokenizer, texts, model_name):
    """Get predictions from a single model"""
    predictions = []
    probabilities = []
    
    model.eval()
    
    for text in tqdm(texts, desc=f"🔍 {model_name} Inference"):
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                pred = torch.argmax(outputs.logits, dim=1).cpu().item()
                
            predictions.append(pred)
            probabilities.append(probs)
            
        except Exception as e:
            print(f"Error in {model_name}: {e}")
            predictions.append(1)  # Default to neutral
            probabilities.append([0.33, 0.34, 0.33])
    
    return np.array(predictions), np.array(probabilities)

# Get individual predictions
lite_preds, lite_probs = get_single_model_predictions(model_lite, tokenizer_lite, texts, "IndoBERT Lite")
bertweet_preds, bertweet_probs = get_single_model_predictions(model_bertweet, tokenizer_bertweet, texts, "IndoBERTweet")

# === ENSEMBLE PREDIKSI ===
print("\n" + "="*60)
print("🔄 CREATING ENSEMBLE PREDICTIONS")
print("="*60)

ensemble_preds, ensemble_probs = ensemble_prediction(
    texts,
    model_lite=model_lite,
    tokenizer_lite=tokenizer_lite,
    model_bertweet=model_bertweet,
    tokenizer_bertweet=tokenizer_bertweet,
    alpha=ALPHA
)

# === EVALUASI SEMUA MODEL ===
models_results = {
    'IndoBERT Lite': lite_preds,
    'IndoBERTweet': bertweet_preds,
    f'Ensemble (α={ALPHA})': ensemble_preds
}

results_summary = []

for model_name, predictions in models_results.items():
    print(f"\n{'='*60}")
    print(f"📊 {model_name.upper()} EVALUATION")
    print(f"{'='*60}")
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1_weighted = safe_f1_score(true_labels, predictions, average='weighted')
    f1_macro = safe_f1_score(true_labels, predictions, average='macro')
    f1_micro = safe_f1_score(true_labels, predictions, average='micro')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Micro): {f1_micro:.4f}")
    
    # Check present classes
    unique_true = set(true_labels)
    unique_pred = set(predictions)
    present_labels = sorted(list(unique_true.union(unique_pred)))
    
    print(f"\nClasses in true labels: {sorted(unique_true)}")
    print(f"Classes in predictions: {sorted(unique_pred)}")
    
    # Classification report
    print(f"\n📋 {model_name} Classification Report:")
    try:
        present_target_names = [label_names[i] for i in present_labels]
        print(classification_report(
            true_labels, predictions,
            labels=present_labels,
            target_names=present_target_names,
            zero_division=0,
            digits=4
        ))
    except Exception as e:
        print(f"Error in classification report: {e}")
        print(classification_report(
            true_labels, predictions,
            labels=[0, 1, 2],
            target_names=label_names,
            zero_division=0,
            digits=4
        ))
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions, labels=[0, 1, 2])
    print(f"\n🔄 {model_name} Confusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    # Safe filename
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace(".", "")
    plt.savefig(os.path.join(output_folder, f'confusion_matrix_{safe_name}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    results_summary.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'F1_Weighted': f1_weighted,
        'F1_Macro': f1_macro,
        'F1_Micro': f1_micro
    })
    
    # Save individual results
    results_df = pd.DataFrame({
        'True_Label': true_labels,
        'Predicted_Label': predictions,
        'True_Label_Name': [label_names[i] for i in true_labels],
        'Predicted_Label_Name': [label_names[i] for i in predictions]
    })
    results_df.to_csv(os.path.join(output_folder, f'{safe_name}_evaluation_results.csv'), index=False)

# === COMPARISON SUMMARY ===
comparison_df = pd.DataFrame(results_summary)
print(f"\n{'='*70}")
print("🏆 INDOBERT MODELS COMPARISON SUMMARY")
print(f"{'='*70}")
print(comparison_df.to_string(index=False, float_format='%.4f'))

# Save comparison
comparison_df.to_csv(os.path.join(output_folder, 'indobert_models_comparison.csv'), index=False)

# Plot comparison
metrics_to_plot = ['Accuracy', 'F1_Weighted', 'F1_Macro', 'F1_Micro']
x = np.arange(len(comparison_df))

fig, ax = plt.subplots(figsize=(12, 8))
width = 0.2

colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']

for i, metric in enumerate(metrics_to_plot):
    values = comparison_df[metric].values
    bars = ax.bar(x + i * width, values, width, label=metric, alpha=0.8, color=colors[i])
    
    # Add value labels on bars
    for j, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Models')
ax.set_ylabel('Score')
ax.set_title('IndoBERT Models and Ensemble Performance Comparison')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'indobert_ensemble_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

# Find best method
best_method_idx = comparison_df['F1_Weighted'].idxmax()
best_method = comparison_df.iloc[best_method_idx]

print(f"\n🏆 Best Method: {best_method['Model']}")
print(f"   Accuracy: {best_method['Accuracy']:.4f}")
print(f"   F1 Weighted: {best_method['F1_Weighted']:.4f}")

# Show improvement
lite_f1 = comparison_df[comparison_df['Model'] == 'IndoBERT Lite']['F1_Weighted'].iloc[0]
bertweet_f1 = comparison_df[comparison_df['Model'] == 'IndoBERTweet']['F1_Weighted'].iloc[0]
ensemble_f1 = comparison_df[comparison_df['Model'] == f'Ensemble (α={ALPHA})']['F1_Weighted'].iloc[0]

print(f"\n📈 Ensemble Analysis:")
print(f"   IndoBERT Lite F1: {lite_f1:.4f}")
print(f"   IndoBERTweet F1: {bertweet_f1:.4f}")
print(f"   Ensemble F1: {ensemble_f1:.4f}")

if ensemble_f1 > max(lite_f1, bertweet_f1):
    improvement = ensemble_f1 - max(lite_f1, bertweet_f1)
    print(f"   ✅ Ensemble improved by: {improvement:.4f}")
else:
    print(f"   ❌ Ensemble did not improve over individual models")

print(f"\n💾 All results saved!")