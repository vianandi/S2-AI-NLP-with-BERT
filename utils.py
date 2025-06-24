import pandas as pd
import re
import os
from sklearn.metrics import f1_score
import numpy as np

def standardize_label(label):
    label = str(label).strip().lower()
    # More comprehensive label mapping
    if any(word in label for word in ["pos", "positive", "1"]):
        return "positive"
    elif any(word in label for word in ["neg", "negative", "0"]):
        return "negative"
    elif any(word in label for word in ["net", "neutral", "netral", "2"]):
        return "neutral"
    else:
        return "neutral"  # Default to neutral instead of None

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.lower().strip()

def safe_f1_score(y_true, y_pred, average='weighted', labels=None):
    """
    Calculate F1 score with handling for missing labels
    """
    try:
        return f1_score(y_true, y_pred, average=average, labels=labels, zero_division=0)
    except ValueError as e:
        if "pos_label" in str(e):
            # Handle missing label case by using only present labels
            unique_true = set(y_true)
            unique_pred = set(y_pred)
            present_labels = sorted(list(unique_true.union(unique_pred)))
            return f1_score(y_true, y_pred, average=average, labels=present_labels, zero_division=0)
        else:
            raise e

def load_and_clean_all_datasets():
    all_data = []

    # === Dataset: Kurikulum 2013 ===
    try:
        df = pd.read_excel("sentiment_ablation/data/Dataset Sentimen kurikulum 2013.xlsx")
        df = df[["tweet", "sentiment"]].rename(columns={"tweet": "text"})
        print("✅ Loaded: Kurikulum 2013")
        all_data.append(df)
    except Exception as e:
        print("❌ Kurikulum:", e)

    # === Dataset: Tayangan TV ===
    try:
        df = pd.read_csv("sentiment_ablation/data/dataset_tweet_sentimen_tayangan_tv.csv")
        df = df[["Text Tweet", "Sentiment"]].rename(columns={"Text Tweet": "text", "Sentiment": "sentiment"})
        print("✅ Loaded: Tayangan TV")
        all_data.append(df)
    except Exception as e:
        print("❌ Tayangan TV:", e)

    # === Dataset: Opini Film ===
    try:
        df = pd.read_csv("sentiment_ablation/data/dataset_tweet_sentiment_opini_film.csv")
        df = df[["Text Tweet", "Sentiment"]].rename(columns={"Text Tweet": "text", "Sentiment": "sentiment"})
        print("✅ Loaded: Opini Film")
        all_data.append(df)
    except Exception as e:
        print("❌ Opini Film:", e)

    # === Dataset: Pariwisata ===
    try:
        df = pd.read_excel("sentiment_ablation/data/id-tourism-sentimentanalysis.xlsx")
        df = df[["review", "sentiment"]].rename(columns={"review": "text"})
        print("✅ Loaded: Pariwisata")
        all_data.append(df)
    except Exception as e:
        print("❌ Pariwisata:", e)
        
    # === Dataset: Sentiment Pilkada DKI ===
    try:
        df = pd.read_csv("sentiment_ablation/data/dataset_tweet_sentiment_pilkada_DKI_2017.csv")
        df = df[["Text Tweet", "Sentiment"]].rename(columns={"Text Tweet": "text", "Sentiment": "sentiment"})
        print("✅ Loaded: Pilkada DKI")
        all_data.append(df)
    except Exception as e:
        print("❌ Pilkada DKI:", e)

    # === Gabungkan Semua ===
    combined = pd.concat(all_data, ignore_index=True)
    combined.dropna(inplace=True)
    combined["text"] = combined["text"].apply(clean_text)
    combined["sentiment"] = combined["sentiment"].apply(standardize_label)
    combined = combined[combined["sentiment"].isin(["positive", "neutral", "negative"])]

    # Ensure balanced representation of all classes
    min_samples = combined["sentiment"].value_counts().min()
    if min_samples < 200:  
        print(f"⚠️  Warning: Minimum class has only {min_samples} samples")
    
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    combined["label"] = combined["sentiment"].map(label_map)

    print(f"\n✅ Total data gabungan setelah cleaning: {len(combined)}")
    print(f"📊 Label distribution:")
    print(combined["sentiment"].value_counts())
    
    return combined