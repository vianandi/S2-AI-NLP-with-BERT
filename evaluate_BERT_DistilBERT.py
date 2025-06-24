import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import BertForSequenceClassification, DistilBertForSequenceClassification, BertTokenizerFast, DistilBertTokenizerFast
from datasets import Dataset
from utils import load_and_clean_all_datasets, safe_f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

print("🔧 PyTorch version:", torch.__version__)
print("🚀 CUDA available:", torch.cuda.is_available())
print("🖥️  Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

# Folder untuk menyimpan hasil
output_folder = 'resultevaluatebertdistilbert'
os.makedirs(output_folder, exist_ok=True)

# Load dataset
print("📂 Loading and cleaning datasets...")
df = load_and_clean_all_datasets()
print("✅ Total data after cleaning:", len(df))

# Buat dataset dan bagi
dataset = Dataset.from_pandas(df[['text', 'label']])
dataset = dataset.train_test_split(test_size=0.2, seed=42)
test_dataset = dataset['test']

print("📊 Test dataset size:", len(test_dataset))

# Ekstrak teks dan label asli
texts = [example['text'] for example in test_dataset]
true_labels = np.array([example['label'] for example in test_dataset])

# Pengaturan device dan label
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_names = ['Negative', 'Neutral', 'Positive']

def get_model_predictions(model_path, tokenizer_class, model_class, texts, model_name):
    """Mendapatkan prediksi dari satu model dengan penanganan error."""
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return None, None
    
    print(f"🔧 Loading {model_name} from {model_path}...")
    tokenizer = tokenizer_class.from_pretrained(model_path)
    model = model_class.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print(f"✅ {model_name} loaded successfully.")
    
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for text in tqdm(texts, desc=f"🔍 {model_name} Inference"):
            try:
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
                pred = np.argmax(probs)
                
                predictions.append(pred)
                probabilities.append(probs)
            except Exception as e:
                print(f"Error processing text with {model_name}: {e}")
                predictions.append(1)  # Default ke netral jika error
                probabilities.append([0.33, 0.34, 0.33])

    return np.array(predictions), np.array(probabilities)

# === LOAD MODELS AND GET PREDICTIONS ===
print("\n" + "="*60)
print("🤖 GETTING INDIVIDUAL PREDICTIONS")
print("="*60)

bert_preds, bert_probs = get_model_predictions(
    "models/bert/final", 
    BertTokenizerFast, 
    BertForSequenceClassification, 
    texts,
    "BERT"
)

distilbert_preds, distilbert_probs = get_model_predictions(
    "models/distilbert/final",
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    texts,
    "DistilBERT"
)

if bert_preds is None or distilbert_preds is None:
    print("❌ Could not load both models. Exiting...")
    exit(1)

# === ENSEMBLE METHODS ===
print("\n" + "="*60)
print("🔄 CREATING ENSEMBLE PREDICTIONS")
print("="*60)

# Metode 1: Majority Voting
ensemble_majority = []
for i in range(len(bert_preds)):
    votes = [bert_preds[i], distilbert_preds[i]]
    vote_counts = np.bincount(votes, minlength=len(label_names))
    ensemble_majority.append(np.argmax(vote_counts))
ensemble_majority = np.array(ensemble_majority)
print("✅ Ensemble (Majority) created.")

# Metode 2: Average Probabilities
ensemble_avg_probs = (bert_probs + distilbert_probs) / 2
ensemble_avg_preds = np.argmax(ensemble_avg_probs, axis=1)
print("✅ Ensemble (Avg Prob) created.")

# Metode 3: Weighted Average (bobot bisa disesuaikan)
weight_bert = 0.8
weight_distilbert = 0.2
ensemble_weighted_probs = (weight_bert * bert_probs + weight_distilbert * distilbert_probs)
ensemble_weighted_preds = np.argmax(ensemble_weighted_probs, axis=1)
print(f"✅ Ensemble (Weighted {weight_bert}/{weight_distilbert}) created.")

# === EVALUASI SEMUA METODE ===
methods = {
    'BERT': bert_preds,
    'DistilBERT': distilbert_preds,
    'Ensemble (Majority)': ensemble_majority,
    'Ensemble (Avg Prob)': ensemble_avg_preds,
    'Ensemble (Weighted)': ensemble_weighted_preds
}

results_summary = []

for method_name, predictions in methods.items():
    print(f"\n{'='*60}")
    print(f"📊 {method_name.upper()} EVALUATION")
    print(f"{'='*60}")
    
    # Hitung metrik
    accuracy = accuracy_score(true_labels, predictions)
    f1_weighted = safe_f1_score(true_labels, predictions, average='weighted')
    f1_macro = safe_f1_score(true_labels, predictions, average='macro')
    f1_micro = safe_f1_score(true_labels, predictions, average='micro')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Micro): {f1_micro:.4f}")
    
    # Cek kelas yang ada
    unique_true = set(true_labels)
    unique_pred = set(predictions)
    present_labels = sorted(list(unique_true.union(unique_pred)))
    
    print(f"\nClasses in true labels: {sorted(unique_true)}")
    print(f"Classes in predictions: {sorted(unique_pred)}")
    
    # Laporan Klasifikasi
    print(f"\n📋 {method_name} Classification Report:")
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
    print(f"\n🔄 {method_name} Confusion Matrix:")
    print(cm)
    
    # Simpan hasil
    results_summary.append({
        'Method': method_name,
        'Accuracy': accuracy,
        'F1_Weighted': f1_weighted,
        'F1_Macro': f1_macro,
        'F1_Micro': f1_micro
    })
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.title(f'Confusion Matrix - {method_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    safe_name = method_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(os.path.join(output_folder, f'confusion_matrix_{safe_name}.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

# === RINGKASAN PERBANDINGAN ===
comparison_df = pd.DataFrame(results_summary)
print(f"\n{'='*70}")
print("🏆 MODEL AND ENSEMBLE COMPARISON SUMMARY")
print(f"{'='*70}")
print(comparison_df.to_string(index=False, float_format='%.4f'))

# Simpan perbandingan
comparison_df.to_csv(os.path.join(output_folder, 'ensemble_comparison_results.csv'), index=False)

# Plot perbandingan
metrics_to_plot = ['Accuracy', 'F1_Weighted', 'F1_Macro', 'F1_Micro']
x = np.arange(len(comparison_df))

fig, ax = plt.subplots(figsize=(15, 8))
width = 0.2
colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']

for i, metric in enumerate(metrics_to_plot):
    values = comparison_df[metric].values
    bars = ax.bar(x + i * width, values, width, label=metric, alpha=0.8, color=colors[i])
    
    for j, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Methods')
ax.set_ylabel('Score')
ax.set_title('Model and Ensemble Performance Comparison')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(comparison_df['Method'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'ensemble_performance_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

# Cari metode terbaik
best_method_idx = comparison_df['F1_Weighted'].idxmax()
best_method = comparison_df.iloc[best_method_idx]

print(f"\n🏆 Best Method: {best_method['Method']}")
print(f"   Accuracy: {best_method['Accuracy']:.4f}")
print(f"   F1 Weighted: {best_method['F1_Weighted']:.4f}")

# === ANALISIS PENINGKATAN ENSEMBLE ===
print(f"\n📈 Ensemble Analysis:")
try:
    bert_f1 = comparison_df[comparison_df['Method'] == 'BERT']['F1_Weighted'].iloc[0]
    distilbert_f1 = comparison_df[comparison_df['Method'] == 'DistilBERT']['F1_Weighted'].iloc[0]
    best_performer_f1 = best_method['F1_Weighted']
    best_performer_name = best_method['Method']

    print(f"   BERT F1: {bert_f1:.4f}")
    print(f"   DistilBERT F1: {distilbert_f1:.4f}")
    print(f"   Best Performer ('{best_performer_name}') F1: {best_performer_f1:.4f}")

    # Cek apakah metode terbaik adalah sebuah ensemble
    if "Ensemble" in best_performer_name:
        if best_performer_f1 > max(bert_f1, distilbert_f1):
            improvement = best_performer_f1 - max(bert_f1, distilbert_f1)
            print(f"   ✅ Ensemble improved by: {improvement:.4f}")
        else:
            print(f"   ❌ Ensemble did not improve over individual models")
    else:
        print("   ℹ️ The best performing method was an individual model, not an ensemble.")

except (IndexError, KeyError) as e:
    print(f"   Could not perform ensemble improvement analysis due to an error: {e}")

print(f"\n💾 All results saved to '{output_folder}' folder!")
print("✅ Ensemble evaluation completed!")