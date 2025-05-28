import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === DATA REKAP MANUAL ===
models = [
    {
        "Model": "BERT",
        "Accuracy": 0.865,
        "F1": 0.862,
        "Size_MB": 497.80,
        "Params_M": 124.44,
        "Inference_s": 0.0401,
        "Throughput": 24.94,
        "Notes": "indobert-base-p1"
    },
    {
        "Model": "DistilBERT",
        "Accuracy": 0.850,
        "F1": 0.847,
        "Size_MB": 442.26,
        "Params_M": 110.56,
        "Inference_s": 0.0405,
        "Throughput": 24.69,
        "Notes": "indobertweet-uncased"
    },
    {
        "Model": "IndoBERT Lite",
        "Accuracy": 0.858,
        "F1": 0.855,
        "Size_MB": 250.32,
        "Params_M": 66.00,
        "Inference_s": 0.0287,
        "Throughput": 34.84,
        "Notes": "indobert-lite-base-p1"
    },
    {
        "Model": "IndoBERTweet",
        "Accuracy": 0.860,
        "F1": 0.857,
        "Size_MB": 410.22,
        "Params_M": 110.00,
        "Inference_s": 0.0350,
        "Throughput": 28.57,
        "Notes": "indobertweet-base-uncased"
    },
    {
        "Model": "Ensemble (BERT + DistilBERT)",
        "Accuracy": 0.870,
        "F1": 0.868,
        "Size_MB": 940.05,
        "Params_M": 235.00,
        "Inference_s": 0.0794,
        "Throughput": 12.59,
        "Notes": "Average logits"
    },
    {
        "Model": "Ensemble (Lite + IndoBERTweet)",
        "Accuracy": 0.872,
        "F1": 0.869,
        "Size_MB": 660.54,
        "Params_M": 176.00,
        "Inference_s": 0.0642,
        "Throughput": 15.58,
        "Notes": "Average logits (ringan)"
    }
]

# === SIMPAN KE FILE ===
df = pd.DataFrame(models)
df.to_csv("results_final.csv", index=False)
df.to_markdown("results_final.md", index=False)
print("✅ Rekap disimpan ke: results_final.csv & results_final.md")

# === PLOT 1: Accuracy vs F1 ===
plt.figure(figsize=(10, 5))
x = np.arange(len(df))
width = 0.35

plt.bar(x - width/2, df["Accuracy"], width, label="Accuracy", color="skyblue")
plt.bar(x + width/2, df["F1"], width, label="F1 Score", color="orange")

plt.xticks(x, df["Model"], rotation=30, ha="right")
plt.ylabel("Score")
plt.title("Model Comparison: Accuracy vs F1")
plt.legend()
plt.tight_layout()
plt.savefig("results_final_accuracy_f1.png")
plt.show()

# === PLOT 2: Size vs Accuracy (Scatter) ===
plt.figure(figsize=(8, 5))
plt.scatter(df["Size_MB"], df["Accuracy"], s=120, color="seagreen")

for i, row in df.iterrows():
    plt.text(row["Size_MB"] + 10, row["Accuracy"], row["Model"], fontsize=8)

plt.xlabel("Model Size (MB)")
plt.ylabel("Accuracy")
plt.title("Size vs Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig("results_final_size_vs_accuracy.png")
plt.show()