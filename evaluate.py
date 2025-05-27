import torch
from transformers import (
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    BertTokenizerFast
)
from utils import load_and_clean_data
from sklearn.metrics import classification_report
from tqdm import tqdm

def ensemble_prediction(texts, bert_model_path, distil_model_path, tokenizer_name):
    bert_model = BertForSequenceClassification.from_pretrained(bert_model_path)
    distilbert_model = DistilBertForSequenceClassification.from_pretrained(distil_model_path)
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)

    bert_model.eval()
    distilbert_model.eval()

    predictions = []

    for text in tqdm(texts, desc="🔍 Ensemble Inference"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

        with torch.no_grad():
            bert_logits = bert_model(**inputs).logits
            distil_inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
            distil_logits = distilbert_model(**distil_inputs).logits


        avg_logits = (bert_logits + distil_logits) / 2
        pred = torch.argmax(avg_logits, dim=1).item()
        predictions.append(pred)

    return predictions

df = load_and_clean_data("sentiment_ablation/data/INA_TweetsPPKM_Labeled_Pure.csv")

texts = df['text'].tolist()
true_labels = df['label'].tolist()

ensemble_preds = ensemble_prediction(
    texts,
    bert_model_path="models/bert/final", 
    distil_model_path="models/distilbert/final",
    tokenizer_name="indobenchmark/indobert-base-p1"
)

print("\n📊 Hasil Evaluasi Ensemble (BERT + DistilBERT):")
print(classification_report(true_labels, ensemble_preds, digits=4))