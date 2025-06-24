import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from transformers import (
    BertTokenizerFast,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from utils import load_and_clean_all_datasets, safe_f1_score

# === INFO GPU ===
print("🔧 PyTorch version:", torch.__version__)
print("🚀 CUDA available:", torch.cuda.is_available())
print("🖥️  Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

# === LOAD DATASET ===
df = load_and_clean_all_datasets()
print("✅ Total data after cleaning:", len(df))

dataset = Dataset.from_pandas(df[['text', 'label']])

# === MODEL & TOKENIZER ===
model_name = "indobenchmark/indobert-lite-base-p1"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

def tokenize(example):
    return tokenizer(example["text"], padding=True, truncation=True, max_length=128)

dataset = dataset.train_test_split(test_size=0.2)
dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

# === TRAINING SETUP ===
training_args = TrainingArguments(
    output_dir="models/indobertlite",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2, 
    save_strategy="epoch", 
    logging_dir="./logs",
    logging_steps=10,
)

# === METRICS ===
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average='weighted', labels=[0, 1, 2], zero_division=0),
        "recall": recall_score(labels, preds, average='weighted', labels=[0, 1, 2], zero_division=0),
        "f1": safe_f1_score(labels, preds, average='weighted', labels=[0, 1, 2])
    }

# === TRAINER ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# === TRAINING ===
print("🚀 Fine-tuning IndoBERT Lite dimulai...")
trainer.train()

# === EVALUASI ===
results = trainer.evaluate()
print("✅ Evaluasi akhir:", results)

# === SIMPAN MODEL FINAL ===
model.save_pretrained("models/indobertlite/final")
tokenizer.save_pretrained("models/indobertlite/final")

# === VISUALISASI METRIK ===
metrics = ['accuracy', 'precision', 'recall', 'f1']
values = [results['eval_accuracy'], results['eval_precision'], results['eval_recall'], results['eval_f1']]

plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color=['skyblue', 'orange', 'green', 'red'], alpha=0.8)
plt.ylim(0, 1)
plt.title("Evaluation Metrics for IndoBERT Lite Model")
plt.ylabel("Score")
plt.xlabel("Metrics")

# Add value labels on bars
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontsize=10)

# Save the plot as a PNG file
plt.tight_layout()
plt.savefig("resultmodels/indobertlite_evaluation_metrics.png", dpi=300, bbox_inches='tight')
plt.show()