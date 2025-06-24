import os
import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
    BertTokenizerFast
)
from utils import load_and_clean_all_datasets
from tqdm import tqdm

# === CONFIG ===
df = load_and_clean_all_datasets()
NUM_SAMPLES = len(df)
NUM_WARMUP = 5     # warmup run (tidak dihitung)
MODEL_PATHS = {
    "Lite": "models/indobertlite/final",
    "Tweet": "models/indobertweet/final"
}

# === LOAD SAMPLE DATA ===
df = load_and_clean_all_datasets()
df = df.sample(n=NUM_SAMPLES, random_state=42, replace=True)
texts = df["text"].tolist()

# === HELPER ===
def get_model_size(path):
    for fname in ["pytorch_model.bin", "model.safetensors"]:
        fpath = os.path.join(path, fname)
        if os.path.exists(fpath):
            return os.path.getsize(fpath) / 1e6  # in MB
    return 0.0

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark(model, tokenizer, texts):
    model.eval()
    for i in range(NUM_WARMUP):
        inputs = tokenizer(texts[i], return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            _ = model(**inputs)

    total_time = 0
    for text in tqdm(texts, desc=f"⏱️ Benchmarking {model.__class__.__name__}"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        start = time.time()
        with torch.no_grad():
            _ = model(**inputs)
        end = time.time()
        total_time += (end - start)
    return total_time / len(texts)

def benchmark_ensemble(model1, tokenizer1, model2, tokenizer2, texts):
    total_time = 0
    for i in range(NUM_WARMUP):
        t = texts[i]
        in1 = tokenizer1(t, return_tensors="pt", truncation=True, padding=True, max_length=128)
        in2 = tokenizer2(t, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            _ = model1(**in1).logits
            _ = model2(**in2).logits

    for text in tqdm(texts, desc="⏱️ Benchmarking Ensemble"):
        in1 = tokenizer1(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        in2 = tokenizer2(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        start = time.time()
        with torch.no_grad():
            _ = model1(**in1).logits
            _ = model2(**in2).logits
        end = time.time()
        total_time += (end - start)
    return total_time / len(texts)

# === BENCHMARKING ===
results = []

# IndoBERT Lite
print("\n📊 Benchmark: IndoBERT Lite")
model_lite = AutoModelForSequenceClassification.from_pretrained(MODEL_PATHS["Lite"])
tokenizer_lite = BertTokenizerFast.from_pretrained(MODEL_PATHS["Lite"])
size_lite = get_model_size(MODEL_PATHS["Lite"])
params_lite = count_parameters(model_lite)
time_lite = benchmark(model_lite, tokenizer_lite, texts)

results.append({
    "Model": "IndoBERT Lite",
    "Size_MB": round(size_lite, 2),
    "Params_M": round(params_lite / 1_000_000, 2),
    "Inference_s": round(time_lite, 4),
    "Throughput": round(1 / time_lite, 2),
    "Notes": "indobert-lite-base-p1"
})

# IndoBERTweet
print("\n📊 Benchmark: IndoBERTweet")
model_tweet = BertForSequenceClassification.from_pretrained(MODEL_PATHS["Tweet"])
tokenizer_tweet = BertTokenizerFast.from_pretrained(MODEL_PATHS["Tweet"])
size_tweet = get_model_size(MODEL_PATHS["Tweet"])
params_tweet = count_parameters(model_tweet)
time_tweet = benchmark(model_tweet, tokenizer_tweet, texts)

results.append({
    "Model": "IndoBERTweet",
    "Size_MB": round(size_tweet, 2),
    "Params_M": round(params_tweet / 1_000_000, 2),
    "Inference_s": round(time_tweet, 4),
    "Throughput": round(1 / time_tweet, 2),
    "Notes": "indobertweet-base-uncased"
})

# Ensemble
print("\n📊 Benchmark: Ensemble")
time_ens = benchmark_ensemble(model_lite, tokenizer_lite, model_tweet, tokenizer_tweet, texts)
results.append({
    "Model": "Ensemble (Lite + Tweet)",
    "Size_MB": round(size_lite + size_tweet, 2),
    "Params_M": round(params_lite + params_tweet / 1_000_000, 2),
    "Inference_s": round(time_ens, 4),
    "Throughput": round(1 / time_ens, 2),
    "Notes": "Average logits"
})

# === SIMPAN & PLOT ===

df = pd.DataFrame(results)
df.to_csv("benchmarkindobert/benchmark_lite_tweet.csv", index=False)
df.to_markdown("benchmarkindobert/benchmark_lite_tweet.md", index=False)
print("\n✅ Disimpan ke: benchmark_lite_tweet.csv dan benchmark_lite_tweet.md")
print(df)

# === VISUALISASI ===
plt.figure(figsize=(10, 5))
x = np.arange(len(df))
width = 0.35

plt.bar(x - width/2, df["Size_MB"], width, label="Size (MB)", color="deepskyblue")
plt.bar(x + width/2, df["Inference_s"] * 1000, width, label="Inference Time (ms)", color="darkorange")

plt.xticks(x, df["Model"], rotation=30)
plt.ylabel("Value")
plt.title("IndoBERT Lite vs IndoBERTweet Benchmark")
plt.legend()
plt.tight_layout()
plt.savefig("benchmarkindobert/benchmark_lite_tweet.png")
plt.show()

# Tambahkan grafik throughput
plt.figure(figsize=(8, 5))
plt.bar(df["Model"], df["Throughput"], color="mediumseagreen")
plt.ylabel("Samples per second")
plt.title("Model Throughput Comparison")
plt.tight_layout()
plt.savefig("benchmarkindobert/benchmark_throughput_plot.png")
plt.show()