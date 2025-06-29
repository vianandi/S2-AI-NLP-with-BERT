# File: utils.py

import pandas as pd
import re
import os
from sklearn.metrics import f1_score
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# Impor Sastrawi StopWordRemover
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- INISIALISASI ---
# Inisialisasi Stemmer
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# Inisialisasi StopWordRemover
stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

# Kamus slang
slang_dict = {
    'ga': 'tidak', 'gak': 'tidak', 'nggak': 'tidak', 'engga': 'tidak',
    'bgt': 'banget', 'jg': 'juga', 'aja': 'saja', 'sm': 'sama',
    'tp': 'tapi', 'dlm': 'dalam', 'utk': 'untuk', 'dg': 'dengan',
    'klo': 'kalau', 'krn': 'karena', 'sdh': 'sudah', 'blm': 'belum',
    'yg': 'yang', 'yaa': 'ya', 'wkwkwk': '', 'hehe': '', 'hahaha': '',
    'tdk': 'tidak', 'tks': 'terima kasih', 'dr': 'dari', 'kpd': 'kepada',
    'gue': 'saya', 'gw': 'saya', 'lu': 'kamu', 'lo': 'kamu',
    'karna': 'karena', 'gimana': 'bagaimana', 'gitu': 'begitu',
    'emang': 'memang', 'gini': 'begini', 'kalo': 'kalau'
}

def standardize_label(label):
    """Mengonversi berbagai format label menjadi label standar: 'positive', 'negative', 'neutral'."""
    label = str(label).strip().lower()
    if any(word in label for word in ["pos", "positive", "1", "positif"]):
        return "positive"
    elif any(word in label for word in ["neg", "negative", "0", "negatif"]):
        return "negative"
    elif any(word in label for word in ["net", "neutral", "netral", "2"]):
        return "neutral"
    else:
        return "neutral"

def preprocess_text_advanced(text):
    """
    Fungsi preprocessing lengkap sesuai paper referensi:
    1. Pembersihan dasar (URL, mention, simbol)
    2. Case folding (mengubah ke huruf kecil)
    3. Normalisasi kata slang
    4. Stemming (mengubah ke kata dasar)
    5. Stopword removal (menghapus kata umum)
    """
    text = str(text)
    # 1. Pembersihan dasar
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # 2. Case folding
    text = text.lower().strip()

    # 3. Normalisasi slang
    words = text.split()
    normalized_words = [slang_dict.get(word, word) for word in words]
    text = " ".join(normalized_words)

    # 4. Stemming
    text = stemmer.stem(text)

    # 5. Stopword removal
    text = stopword_remover.remove(text)

    return text

def load_and_clean_all_datasets():
    """Memuat semua dataset, membersihkannya, dan mengembalikan DataFrame gabungan."""
    all_data = []
    datasets_to_load = [
        {"path": "sentiment_ablation/data/Dataset Sentimen kurikulum 2013.xlsx - Sheet1.csv", "text_col": "tweet", "sentiment_col": "sentiment"},
        {"path": "sentiment_ablation/data/dataset_tweet_sentimen_tayangan_tv.csv", "text_col": "Text Tweet", "sentiment_col": "Sentiment"},
        {"path": "sentiment_ablation/data/dataset_tweet_sentiment_opini_film.csv", "text_col": "Text Tweet", "sentiment_col": "Sentiment"},
        {"path": "sentiment_ablation/data/id-tourism-sentimentanalysis.xlsx - id-tourism-sentimentanalysis.csv", "text_col": "review", "sentiment_col": "sentiment"},
        {"path": "sentiment_ablation/data/dataset_tweet_sentiment_pilkada_DKI_2017.csv", "text_col": "Text Tweet", "sentiment_col": "Sentiment"}
    ]

    print("Memuat dataset...")
    for dataset_info in datasets_to_load:
        try:
            df = pd.read_csv(dataset_info["path"])
            df = df[[dataset_info["text_col"], dataset_info["sentiment_col"]]].rename(columns={
                dataset_info["text_col"]: "text",
                dataset_info["sentiment_col"]: "sentiment"
            })
            print(f"✅ Berhasil memuat: {os.path.basename(dataset_info['path'])}")
            all_data.append(df)
        except Exception as e:
            print(f"❌ Gagal memuat {dataset_info['path']}: {e}")

    if not all_data:
        print("Peringatan: Tidak ada dataset yang berhasil dimuat.")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    combined.dropna(subset=['text', 'sentiment'], inplace=True)

    print("\n🔄 Menerapkan preprocessing lengkap (cleaning, slang, stemming, stopword removal)...")
    combined["text"] = combined["text"].apply(preprocess_text_advanced)
    
    print("🔄 Menstandardisasi label...")
    combined["sentiment"] = combined["sentiment"].apply(standardize_label)
    combined = combined[combined["sentiment"].isin(["positive", "negative", "neutral"])]
    combined = combined[combined['text'].str.strip().astype(bool)] # Hapus baris yang teksnya kosong setelah preprocessing

    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    combined["label"] = combined["sentiment"].map(label_map)
    combined.dropna(subset=['label'], inplace=True)
    combined['label'] = combined['label'].astype(int)


    print(f"\n✅ Total data setelah preprocessing lengkap: {len(combined)}")
    print("📊 Distribusi label:")
    print(combined["sentiment"].value_counts())
    
    return combined
