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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import load_and_clean_all_datasets
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
    for text in tqdm(texts, desc="Predicting with BERT x DistilBERT Ensemble"):
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
    """
    Evaluate predictions using F1 score.
    """
    try:
        return {
            "f1": f1_score(true_labels, preds, average="weighted", zero_division=0)  # Use 'weighted' average
        }
    except ValueError as e:
        if "pos_label" in str(e):
            # Handle missing labels by computing F1 for all present labels
            unique_true = set(true_labels)
            unique_pred = set(preds)
            present_labels = sorted(list(unique_true.union(unique_pred)))
            return {
                "f1": f1_score(true_labels, preds, average="weighted", labels=present_labels, zero_division=0)
            }
        else:
            raise e

df = load_and_clean_all_datasets()
NUM_SAMPLES = 1500 
df = df.sample(n=NUM_SAMPLES, random_state=42, replace=True)
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
indobert_ensemble_preds = ensemble_predictions_indobert(indobert_lite_model, indobertweet_model, indobert_lite_tokenizer, indobertweet_tokenizer, texts)

#calculate metrics
bert_accuracy = accuracy_score(true_labels, bert_preds)
bert_f1 = f1_score(true_labels, bert_preds, average="weighted")
bert_precision = precision_score(true_labels, bert_preds, average="weighted", zero_division=0)
bert_recall = recall_score(true_labels, bert_preds, average="weighted", zero_division=0)
distil_accuracy = accuracy_score(true_labels, distil_preds)
distil_f1 = f1_score(true_labels, distil_preds, average="weighted")
indobert_lite_accuracy = accuracy_score(true_labels, indobert_lite_preds)
indobert_lite_f1 = f1_score(true_labels, indobert_lite_preds, average="weighted")   
indobertweet_accuracy = accuracy_score(true_labels, indobertweet_preds)
indobertweet_f1 = f1_score(true_labels, indobertweet_preds, average="weighted")
ensemble_accuracy = accuracy_score(true_labels, ensemble_preds)
ensemble_f1 = f1_score(true_labels, ensemble_preds, average="weighted")
indobert_ensemble_accuracy = accuracy_score(true_labels, indobert_ensemble_preds)
indobert_ensemble_f1 = f1_score(true_labels, indobert_ensemble_preds, average="weighted")   

results = [
    {"Model": "IndoBERT-Lite", **evaluate(true_labels, indobert_lite_preds), "Epoch": 2, "Notes": "indobert-lite-base-p1"},
    {"Model": "IndoBERTweet", **evaluate(true_labels, indobertweet_preds), "Epoch": 2, "Notes": "indobertweet-base-uncased"},
    {"Model": "BERT", **evaluate(true_labels, bert_preds), "Epoch": 2, "Notes": "indobert-base-p1"},
    {"Model": "DistilBERT", **evaluate(true_labels, distil_preds), "Epoch": 2, "Notes": "indobertweet-uncased"},
    {"Model": "Ensemble BERT+Distil", **evaluate(true_labels, ensemble_preds), "Epoch": "-", "Notes": "Average logits BERT & DistilBERT"},
    {"Model": "Ensemble IndoBERT-Lite+IndoBERTweet", **evaluate(true_labels, indobert_ensemble_preds), "Epoch": "-", "Notes": "Average logits IndoBERT-Lite & IndoBERTweet"}
]

df_results = pd.DataFrame({
    "Model": ["BERT", "DistilBERT", "IndoBERT-Lite", "IndoBERTweet", "Ensemble BERT+Distil", "Ensemble IndoBERT-Lite+IndoBERTweet"],
    "accuracy": [bert_accuracy, distil_accuracy, indobert_lite_accuracy, indobertweet_accuracy, ensemble_accuracy, indobert_ensemble_accuracy],
    "f1": [bert_f1, distil_f1, indobert_lite_f1, indobertweet_f1, ensemble_f1, indobert_ensemble_f1],
    
})
df_results.to_csv("results/table_results.csv", index=False)
df_results.to_markdown("results/table_results.md", index=False)
print("✅ Hasil disimpan sebagai table_results.csv dan table_results.md")

plt.figure(figsize=(8, 5))
bar_width = 0.35
x = np.arange(len(df_results))

plt.bar(x - bar_width/2, df_results['accuracy'], width=bar_width, label='Accuracy', color='skyblue')
plt.bar(x + bar_width/2, df_results['f1'], width=bar_width, label='F1 Score', color='orange')



# Pastikan kolom 'accuracy' ada
if 'accuracy' not in df_results.columns:
    print("⚠️ 'accuracy' column is missing. Calculating accuracy...")
    # Hitung akurasi berdasarkan true_labels dan preds
    df_results['accuracy'] = df_results.apply(
        lambda row: np.mean(np.array(row['true_labels']) == np.array(row['preds'])), axis=1
    )

# Tampilkan tabel hasil
print("\n📊 Tabel Hasil:")
print(df_results)

# Visualisasi
bar_width = 0.35
x = np.arange(len(df_results['Model']))

plt.bar(x - bar_width/2, df_results['accuracy'], width=bar_width, label='Accuracy', color='skyblue')
plt.bar(x + bar_width/2, df_results['f1'], width=bar_width, label='F1 Score', color='orange')

plt.xlabel('Model')
plt.ylabel('Scores')
plt.title('Model Performance')
plt.xticks(x, df_results['Model'])
plt.legend()

plt.tight_layout()
plt.show()