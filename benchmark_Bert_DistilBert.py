import os
import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import (
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    BertTokenizerFast
)
from utils import load_and_clean_all_datasets
from tqdm import tqdm

df = load_and_clean_all_datasets()

NUM_SAMPLES = len(df)  
NUM_WARMUP = 5     
BATCH_SIZES = [1, 8] 

MODEL_PATHS = {
    "BERT": "models/bert/final",
    "DistilBERT": "models/distilbert/final"
}

df = load_and_clean_all_datasets()
df = df.sample(n=NUM_SAMPLES, random_state=42, replace=True)
texts = df["text"].tolist()

tokenizer = BertTokenizerFast.from_pretrained("indobenchmark/indobert-base-p1")

def get_model_size(path):
    for fname in ["pytorch_model.bin", "model.safetensors"]:
        fpath = os.path.join(path, fname)
        if os.path.exists(fpath):
            return os.path.getsize(fpath) / 1e6 
    raise FileNotFoundError("Tidak ditemukan model file (.bin atau .safetensors) di " + path)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark(model, tokenizer, texts, remove_token_type_ids=False):
    model.eval()
    
    for i in range(NUM_WARMUP):
        inputs = tokenizer(texts[i % len(texts)], return_tensors="pt", truncation=True, padding=True, max_length=128)
        if remove_token_type_ids:
            inputs.pop("token_type_ids", None)
        with torch.no_grad():
            _ = model(**inputs)
    
    total_time = 0
    for text in tqdm(texts, desc=f"Benchmarking {model.__class__.__name__}"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        if remove_token_type_ids:
            inputs.pop("token_type_ids", None)
        start = time.time()
        with torch.no_grad():
            _ = model(**inputs)
        end = time.time()
        total_time += (end - start)
    
    return total_time / len(texts)

def benchmark_batch(model, tokenizer, texts, batch_size=8, remove_token_type_ids=False):
    model.eval()
    total_time = 0
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
        if remove_token_type_ids:
            inputs.pop("token_type_ids", None)
        
        start = time.time()
        with torch.no_grad():
            _ = model(**inputs)
        end = time.time()
        total_time += (end - start)
    
    return total_time / len(texts)

def benchmark_ensemble(bert_model, distil_model, tokenizer, texts):
    # Warmup
    for i in range(NUM_WARMUP):
        inputs = tokenizer(texts[i % len(texts)], return_tensors="pt", truncation=True, padding=True, max_length=128)
        distil_inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
        with torch.no_grad():
            _ = bert_model(**inputs).logits
            _ = distil_model(**distil_inputs).logits
    
    total_time = 0
    for text in tqdm(texts, desc="Benchmarking Ensemble"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        distil_inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
        start = time.time()
        with torch.no_grad():
            _ = bert_model(**inputs).logits
            _ = distil_model(**distil_inputs).logits
        end = time.time()
        total_time += (end - start)
    
    return total_time / len(texts)

results = []

try:
    print("\n📊 Benchmarking BERT model...")
    bert_model = BertForSequenceClassification.from_pretrained(MODEL_PATHS["BERT"])
    bert_size = get_model_size(MODEL_PATHS["BERT"])
    bert_params = count_parameters(bert_model)
    bert_time = benchmark(bert_model, tokenizer, texts)
    results.append({
        "Model": "BERT",
        "Size_MB": round(bert_size, 2),
        "Parameters_M": round(bert_params/1_000_000, 2),
        "Avg_Inference_s": round(bert_time, 4),
        "Notes": "indobert-base-p1"
    })

    print("\n📊 Benchmarking DistilBERT model...")
    distil_model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATHS["DistilBERT"])
    distil_size = get_model_size(MODEL_PATHS["DistilBERT"])
    distil_params = count_parameters(distil_model)
    distil_time = benchmark(distil_model, tokenizer, texts, remove_token_type_ids=True)
    results.append({
        "Model": "DistilBERT",
        "Size_MB": round(distil_size, 2),
        "Parameters_M": round(distil_params/1_000_000, 2),
        "Avg_Inference_s": round(distil_time, 4),
        "Notes": "indobertweet-base-uncased"
    })

    print("\n📊 Benchmarking Ensemble model...")
    ensemble_size = bert_size + distil_size
    ensemble_params = bert_params + distil_params
    ensemble_time = benchmark_ensemble(bert_model, distil_model, tokenizer, texts)
    results.append({
        "Model": "Ensemble",
        "Size_MB": round(ensemble_size, 2),
        "Parameters_M": round(ensemble_params/1_000_000, 2),
        "Avg_Inference_s": round(ensemble_time, 4),
        "Notes": "Average logits (BERT + DistilBERT)"
    })

    for result in results:
        result["Throughput_samples_per_second"] = round(1 / result["Avg_Inference_s"], 2)

except Exception as e:
    print(f"❌ Error during benchmarking: {e}")

df_results = pd.DataFrame(results)
df_results.to_csv("benchmarkbertdistilbert/compression_results.csv", index=False)
df_results.to_markdown("benchmarkbertdistilbert/compression_results.md", index=False)
print("✅ Hasil disimpan ke: compression_results.csv dan compression_results.md")
print(df_results)

plt.figure(figsize=(10, 6))
x = np.arange(len(df_results))
width = 0.35

plt.bar(x - width/2, df_results["Size_MB"], width, label="Size (MB)", color="steelblue")
plt.bar(x + width/2, df_results["Avg_Inference_s"]*1000, width, label="Inference time (ms)", color="darkorange")

plt.xticks(x, df_results["Model"])
plt.ylabel("Value")
plt.title("Model Size & Inference Time Comparison")
plt.legend()

for i, v in enumerate(df_results["Size_MB"]):
    plt.text(i - width/2, v + 5, f"{v} MB", ha='center')
    
for i, v in enumerate(df_results["Avg_Inference_s"]*1000):
    plt.text(i + width/2, v + 5, f"{v:.1f} ms", ha='center')

plt.tight_layout()
plt.savefig("benchmarkbertdistilbert/compression_plot.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(df_results["Model"], df_results["Throughput_samples_per_second"], color="mediumseagreen")
plt.ylabel("Samples per second")
plt.title("Model Throughput Comparison")
plt.tight_layout()
plt.savefig("benchmarkbertdistilbert/throughput_plot.png")
plt.show()