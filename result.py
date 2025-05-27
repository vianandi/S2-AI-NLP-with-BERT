import torch
import pandas as pd
import numpy as np
from transformers import (
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    BertTokenizerFast
)
from sklearn.metrics import accuracy_score, f1_score
from utils import load_and_clean_data
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_predictions(model, tokenizer, texts, remove_token_type_ids=False):
    model.eval()
    predictions = []

    for text in tqdm(texts, desc=f"Predicting with {model.name_or_path}"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        if remove_token_type_ids:
            inputs.pop("token_type_ids", None)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
        predictions.append(pred)

    return predictions

def ensemble_predictions(bert_model, distil_model, tokenizer, texts):
    predictions = []
    for text in tqdm(texts, desc="Predicting with Ensemble"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        distil_inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
        with torch.no_grad():
            bert_logits = bert_model(**inputs).logits
            distil_logits = distil_model(**distil_inputs).logits
        avg_logits = (bert_logits + distil_logits) / 2
        pred = torch.argmax(avg_logits, dim=1).item()
        predictions.append(pred)
    return predictions

def evaluate(true_labels, preds):
    return {
        "accuracy": accuracy_score(true_labels, preds),
        "f1": f1_score(true_labels, preds)
    }

df = load_and_clean_data("sentiment_ablation/data/INA_TweetsPPKM_Labeled_Pure.csv")
df = df.sample(n=3000, random_state=42) 
texts = df['text'].tolist()
true_labels = df['label'].tolist()

bert_model = BertForSequenceClassification.from_pretrained("models/bert/final")
distil_model = DistilBertForSequenceClassification.from_pretrained("models/distilbert/final")
tokenizer = BertTokenizerFast.from_pretrained("indobenchmark/indobert-base-p1")

bert_preds = get_predictions(bert_model, tokenizer, texts)
distil_preds = get_predictions(distil_model, tokenizer, texts, remove_token_type_ids=True)
ensemble_preds = ensemble_predictions(bert_model, distil_model, tokenizer, texts)

results = [
    {"Model": "BERT", **evaluate(true_labels, bert_preds), "Epoch": 2, "Notes": "indobert-base-p1"},
    {"Model": "DistilBERT", **evaluate(true_labels, distil_preds), "Epoch": 2, "Notes": "indobertweet-uncased"},
    {"Model": "Ensemble", **evaluate(true_labels, ensemble_preds), "Epoch": "-", "Notes": "Average logits"}
]

df_results = pd.DataFrame(results)
df_results.to_csv("table_results.csv", index=False)
df_results.to_markdown("table_results.md", index=False)
print("✅ Hasil disimpan sebagai table_results.csv dan table_results.md")

plt.figure(figsize=(8, 5))
bar_width = 0.35
x = np.arange(len(df_results))

plt.bar(x - bar_width/2, df_results['accuracy'], width=bar_width, label='Accuracy', color='skyblue')
plt.bar(x + bar_width/2, df_results['f1'], width=bar_width, label='F1 Score', color='orange')

plt.xticks(x, df_results['Model'])
plt.ylim(0.7, 1.0)
plt.ylabel("Score")
plt.title("Perbandingan Model: Accuracy vs F1 Score")
plt.legend()
plt.tight_layout()

plt.savefig("evaluation_plot.png")
plt.show()