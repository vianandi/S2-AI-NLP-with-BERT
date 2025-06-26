import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from utils import load_and_clean_all_datasets, safe_f1_score

print("🔧 PyTorch version:", torch.__version__)
print("🚀 CUDA available:", torch.cuda.is_available())
print("🖥️  Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

df = load_and_clean_all_datasets()
print("✅ Jumlah data:", len(df))

dataset = Dataset.from_pandas(df[['text', 'label']])
dataset = dataset.train_test_split(test_size=0.2)

model_name = "indolem/indobertweet-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

def tokenize(example):
    return tokenizer(example["text"], padding=True, truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

training_args = TrainingArguments(
    output_dir="models/indobertweet",
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,
    num_train_epochs=10, 
    save_strategy="epoch", 
    logging_dir="./logs",
    logging_steps=10,
)

def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average='weighted', labels=[0, 1, 2], zero_division=0),
        "recall": recall_score(labels, preds, average='weighted', labels=[0, 1, 2], zero_division=0),
        "f1": safe_f1_score(labels, preds, average='weighted', labels=[0, 1, 2]),
        "roc_auc": roc_auc_score(pd.get_dummies(labels), pd.get_dummies(preds), multi_class='ovr', average='weighted')
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("🚀 Fine-tuning IndoBERTweet dimulai...")
trainer.train()

results = trainer.evaluate()
print("✅ Evaluasi akhir:", results)

model.save_pretrained("models/indobertweet/final")
tokenizer.save_pretrained("models/indobertweet/final")

# === VISUALISASI METRIK ===
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
values = [results['eval_accuracy'], results['eval_precision'], results['eval_recall'], results['eval_f1'], results['eval_roc_auc']]

plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color=['skyblue', 'orange', 'green', 'red', 'pink'], alpha=0.8)
plt.ylim(0, 1)
plt.title("Evaluation Metrics for IndoBERTweet Model")
plt.ylabel("Score")
plt.xlabel("Metrics")

# Add value labels on bars
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontsize=10)

# Save the plot as a PNG file
plt.tight_layout()
plt.savefig("resultmodels/indobertweet_evaluation_metrics.png", dpi=300, bbox_inches='tight')
plt.show()