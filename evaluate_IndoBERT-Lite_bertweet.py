import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizerFast
)
from sklearn.metrics import classification_report
from utils import load_and_clean_data
from tqdm import tqdm

# === PENGATURAN BOBOT ENSEMBLE ===
ALPHA = 0.5  # bobot IndoBERT Lite, sisanya IndoBERTweet

# === FUNGSI PREDIKSI ENSEMBLE ===
def ensemble_prediction(texts, model_lite, tokenizer_lite, model_bertweet, tokenizer_bertweet, alpha=0.5):
    predictions = []
    model_lite.eval()
    model_bertweet.eval()

    for text in tqdm(texts, desc="🔍 Ensemble Inference"):
        inputs_lite = tokenizer_lite(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs_bertweet = tokenizer_bertweet(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

        with torch.no_grad():
            logits_lite = model_lite(**inputs_lite).logits
            logits_bertweet = model_bertweet(**inputs_bertweet).logits

        avg_logits = alpha * logits_lite + (1 - alpha) * logits_bertweet
        pred = torch.argmax(avg_logits, dim=1).item()
        predictions.append(pred)

    return predictions

# === LOAD DATASET ===
df = load_and_clean_data("sentiment_ablation/data/INA_TweetsPPKM_Labeled_Pure.csv")
texts = df["text"].tolist()
true_labels = df["label"].tolist()

# === LOAD IndoBERT Lite ===
model_lite = AutoModelForSequenceClassification.from_pretrained("models/indobertlite/final")
tokenizer_lite = BertTokenizerFast.from_pretrained("models/indobertlite/final")

# === LOAD IndoBERTweet ===
model_bertweet = BertForSequenceClassification.from_pretrained("models/indobertweet/final")
tokenizer_bertweet = BertTokenizerFast.from_pretrained("models/indobertweet/final")

# === ENSEMBLE PREDIKSI ===
ensemble_preds = ensemble_prediction(
    texts,
    model_lite=model_lite,
    tokenizer_lite=tokenizer_lite,
    model_bertweet=model_bertweet,
    tokenizer_bertweet=tokenizer_bertweet,
    alpha=ALPHA
)

# === EVALUASI HASIL ===
print("\n📊 Evaluasi Ensemble: IndoBERT Lite + IndoBERTweet (alpha =", ALPHA, ")")
print(classification_report(true_labels, ensemble_preds, digits=4))