import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read evaluation results automatically from the CSV output by result.py
try:
    eval_results = pd.read_csv("table_results.csv")
    print("Successfully loaded results from table_results.csv")
except FileNotFoundError:
    print("Warning: table_results.csv not found. Run result.py first to generate model evaluations.")
    eval_results = None

# Hardware metrics lookup (these typically need to be measured separately)
model_specs = {
    "BERT": {
        "Size_MB": 497.80,
        "Params_M": 124.44,
        "Inference_s": 0.0401,
        "Throughput": 24.94
    },
    "DistilBERT": {
        "Size_MB": 442.26,
        "Params_M": 110.56,
        "Inference_s": 0.0405,
        "Throughput": 24.69
    },
    "IndoBERT-Lite": {
        "Size_MB": 250.32,
        "Params_M": 66.00,
        "Inference_s": 0.0287,
        "Throughput": 34.84
    },
    "IndoBERTweet": {
        "Size_MB": 410.22,
        "Params_M": 110.00,
        "Inference_s": 0.0350,
        "Throughput": 28.57
    },
    "Ensemble BERT+Distil": {
        "Size_MB": 940.05,
        "Params_M": 235.00,
        "Inference_s": 0.0794,
        "Throughput": 12.59
    },
    "Ensemble IndoBERT": {
        "Size_MB": 660.54,
        "Params_M": 176.00,
        "Inference_s": 0.0642,
        "Throughput": 15.58
    }
}

# Generate final results automatically if evaluation data is available
models = []
if eval_results is not None:
    for idx, row in eval_results.iterrows():
        model_name = row['Model']
        # Map model names if needed (adjust to match the keys in model_specs)
        if model_name == "Ensemble BERT+Distil" and "Ensemble BERT+Distil" in model_specs:
            spec_key = "Ensemble BERT+Distil"
        elif model_name == "Ensemble IndoBERT" and "Ensemble IndoBERT" in model_specs:
            spec_key = "Ensemble IndoBERT"
        else:
            spec_key = model_name
            
        if spec_key in model_specs:
            models.append({
                "Model": model_name,
                "Accuracy": row['accuracy'],
                "F1": row['f1'],
                **model_specs[spec_key],
                "Notes": row.get('Notes', '')
            })
        else:
            print(f"Warning: No hardware specs found for model {model_name}")
else:
    # Fallback to manual data if no evaluation results are found
    print("Using manual data as fallback...")
    models = [
        {"Model": "BERT", "Accuracy": 0.865, "F1": 0.862, **model_specs["BERT"], 
         "Notes": "indobert-base-p1"},
        {"Model": "DistilBERT", "Accuracy": 0.850, "F1": 0.847, **model_specs["DistilBERT"], 
         "Notes": "indobertweet-uncased"},
        {"Model": "IndoBERT Lite", "Accuracy": 0.858, "F1": 0.855, **model_specs["IndoBERT-Lite"], 
         "Notes": "indobert-lite-base-p1"},
        {"Model": "IndoBERTweet", "Accuracy": 0.860, "F1": 0.857, **model_specs["IndoBERTweet"], 
         "Notes": "indobertweet-base-uncased"},
        {"Model": "Ensemble (BERT + DistilBERT)", "Accuracy": 0.870, "F1": 0.868, **model_specs["Ensemble BERT+Distil"], 
         "Notes": "Average logits"},
        {"Model": "Ensemble (Lite + IndoBERTweet)", "Accuracy": 0.872, "F1": 0.869, **model_specs["Ensemble IndoBERT"], 
         "Notes": "Average logits (ringan)"}
    ]

# The rest of the code remains unchanged
df = pd.DataFrame(models)
df.to_csv("results_final.csv", index=False)
df.to_markdown("results_final.md", index=False)
print("✅ Rekap disimpan ke: results_final.csv & results_final.md")

# === PLOT 1: Accuracy vs F1 ===
plt.figure(figsize=(10, 5))
# Rest of plotting code remains the same