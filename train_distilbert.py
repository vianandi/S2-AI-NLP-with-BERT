import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments
)

from datasets import Dataset
from utils import load_and_clean_data

print("🔧 PyTorch version:", torch.__version__)
print("🚀 CUDA available:", torch.cuda.is_available())
print("🖥️  Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

df = load_and_clean_data("sentiment_ablation/data/INA_TweetsPPKM_Labeled_Pure.csv")
print("✅ Total data after cleaning:", len(df))

dataset = Dataset.from_pandas(df[['text', 'label']])

model_name = "indolem/indobertweet-base-uncased" 
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example['text'], padding=True, truncation=True, max_length=128)

dataset = dataset.train_test_split(test_size=0.2)
dataset = dataset.map(tokenize, batched=True, remove_columns=['text'])

print("📊 Train:", len(dataset['train']), "| Test:", len(dataset['test']))

model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(
    output_dir="models/distilbert",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2, 
    save_strategy="epoch", 
    logging_dir="./logs",
    logging_steps=10,
)

def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("🚀 Training dimulai...")
trainer.train()

results = trainer.evaluate()
print("✅ Evaluasi akhir:", results)

model.save_pretrained("models/distilbert/final")
tokenizer.save_pretrained("models/distilbert/final")
