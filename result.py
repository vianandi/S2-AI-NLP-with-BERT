import torch
import pandas as pd
import numpy as np
from transformers import (
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    BertTokenizerFast,
    AutoModelForSequenceClassification,
    AutoTokenizer
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

def ensemble_predictions(bert_model, distil_model, bert_tokenizer, distil_tokenizer, texts):
    predictions = []
    for text in tqdm(texts, desc="Predicting with Ensemble"):
        # Get inputs for BERT
        bert_inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        # Get inputs for DistilBERT
        distil_inputs = distil_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        if "token_type_ids" in distil_inputs:
            distil_inputs.pop("token_type_ids")
            
        with torch.no_grad():
            bert_logits = bert_model(**bert_inputs).logits
            distil_logits = distil_model(**distil_inputs).logits
        avg_logits = (bert_logits + distil_logits) / 2
        pred = torch.argmax(avg_logits, dim=1).item()
        predictions.append(pred)
    return predictions

def ensemble_predictions_indobert(indobert_lite_model, indobertweet_model, lite_tokenizer, tweet_tokenizer, texts):
    predictions = []
    for text in tqdm(texts, desc="Predicting with IndoBERT-Lite x IndoBERTweet Ensemble"):
        # Get inputs for IndoBERT-Lite
        lite_inputs = lite_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        # Get inputs for IndoBERTweet
        tweet_inputs = tweet_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        with torch.no_grad():
            lite_logits = indobert_lite_model(**lite_inputs).logits
            tweet_logits = indobertweet_model(**tweet_inputs).logits
        avg_logits = (lite_logits + tweet_logits) / 2
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

# Load models
bert_model = BertForSequenceClassification.from_pretrained("models/bert/final")
distil_model = DistilBertForSequenceClassification.from_pretrained("models/distilbert/final")
indobert_lite_model = AutoModelForSequenceClassification.from_pretrained("models/indobertlite/final")
indobertweet_model = BertForSequenceClassification.from_pretrained("models/indobertweet/final")

# Load correct tokenizers for each model
bert_tokenizer = AutoTokenizer.from_pretrained("models/bert/final")
distil_tokenizer = AutoTokenizer.from_pretrained("models/distilbert/final")
indobert_lite_tokenizer = AutoTokenizer.from_pretrained("models/indobertlite/final")
indobertweet_tokenizer = AutoTokenizer.from_pretrained("models/indobertweet/final")

# Get predictions using the appropriate tokenizer for each model
bert_preds = get_predictions(bert_model, bert_tokenizer, texts)
distil_preds = get_predictions(distil_model, distil_tokenizer, texts, remove_token_type_ids=True)
indobert_lite_preds = get_predictions(indobert_lite_model, indobert_lite_tokenizer, texts)
indobertweet_preds = get_predictions(indobertweet_model, indobertweet_tokenizer, texts)

# Ensemble predictions with appropriate tokenizers
ensemble_preds = ensemble_predictions(bert_model, distil_model, bert_tokenizer, distil_tokenizer, texts)
indobert_ensemble_preds = ensemble_predictions_indobert(indobert_lite_model, indobertweet_model, 
                                                       indobert_lite_tokenizer, indobertweet_tokenizer, texts)

results = [
    {"Model": "IndoBERT-Lite", **evaluate(true_labels, indobert_lite_preds), "Epoch": 2, "Notes": "indobert-lite-base-p1"},
    {"Model": "IndoBERTweet", **evaluate(true_labels, indobertweet_preds), "Epoch": 2, "Notes": "indobertweet-base-uncased"},
    {"Model": "BERT", **evaluate(true_labels, bert_preds), "Epoch": 2, "Notes": "indobert-base-p1"},
    {"Model": "DistilBERT", **evaluate(true_labels, distil_preds), "Epoch": 2, "Notes": "indobertweet-uncased"},
    {"Model": "Ensemble BERT+Distil", **evaluate(true_labels, ensemble_preds), "Epoch": "-", "Notes": "Average logits BERT & DistilBERT"},
    {"Model": "Ensemble IndoBERT", **evaluate(true_labels, indobert_ensemble_preds), "Epoch": "-", "Notes": "Average logits IndoBERT-Lite & IndoBERTweet"}
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