import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from utils import load_and_clean_data

print("🔧 PyTorch version:", torch.__version__)
print("🚀 CUDA available:", torch.cuda.is_available())
print("🖥️  Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

df = load_and_clean_data("sentiment_ablation/data/INA_TweetsPPKM_Labeled_Pure.csv")
print("✅ Jumlah data:", len(df))

dataset = Dataset.from_pandas(df[['text', 'label']])
dataset = dataset.train_test_split(test_size=0.2)

model_name = "indolem/indobertweet-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

def tokenize(example):
    return tokenizer(example["text"], padding=True, truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

training_args = TrainingArguments(
    output_dir="models/indobertweet",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs/indobertweet",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2
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