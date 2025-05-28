import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    BertTokenizerFast,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from utils import load_and_clean_data

# === INFO GPU ===
print("🔧 PyTorch version:", torch.__version__)
print("🚀 CUDA available:", torch.cuda.is_available())
print("🖥️  Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

# === LOAD DATASET ===
df = load_and_clean_data("sentiment_ablation/data/INA_TweetsPPKM_Labeled_Pure.csv")
print("✅ Jumlah data:", len(df))

# (Opsional) subset agar cepat
# df = df.sample(n=5000, random_state=42)

dataset = Dataset.from_pandas(df[['text', 'label']])
dataset = dataset.train_test_split(test_size=0.2)

# === MODEL & TOKENIZER ===
model_name = "indobenchmark/indobert-lite-base-p1"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def tokenize(example):
    return tokenizer(example["text"], padding=True, truncation=True, max_length=128)

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
        "f1": f1_score(labels, preds)
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