import pandas as pd
import re
import os
from sklearn.metrics import f1_score
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- INISIALISASI STEMMER & KAMUS SLANG ---

# Buat stemmer (lakukan sekali saja agar efisien)
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()
stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

# Buat kamus normalisasi slang (bisa Anda kembangkan lebih lanjut)
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
    Fungsi preprocessing lengkap:
    1. Pembersihan dasar (URL, mention, non-alfanumerik)
    2. Case folding 
    3. Normalisasi kata slang
    4. Stemming
    5. Stopword removal (BARU!)
    """
    # 1. Pembersihan dasar
    text = str(text)
    text = re.sub(r"http\S+", "", text)  # Hapus URL
    text = re.sub(r"@\w+", "", text)     # Hapus mention
    text = re.sub(r"[^a-zA-Z\s]", "", text) # Hapus karakter non-alfabet
    
    # 2. Case folding (BARU!)
    text = text.lower().strip()

    # 3. Normalisasi kata slang
    words = text.split()
    normalized_words = [slang_dict.get(word, word) for word in words]
    text = " ".join(normalized_words)

    # 4. Stemming
    text = stemmer.stem(text)

    # 5. Stopword removal (BARU!)
    text = stopword_remover.remove(text)

    return text

def safe_f1_score(y_true, y_pred, average='weighted', labels=None):
    """
    Calculate F1 score with handling for missing labels
    """
    try:
        return f1_score(y_true, y_pred, average=average, labels=labels, zero_division=0)
    except ValueError as e:
        if "pos_label" in str(e):
            unique_true = set(y_true)
            unique_pred = set(y_pred)
            present_labels = sorted(list(unique_true.union(unique_pred)))
            return f1_score(y_true, y_pred, average=average, labels=present_labels, zero_division=0)
        else:
            raise e

def load_and_clean_all_datasets():
    all_data = []

    # Struktur data baru yang lebih andal
    # Setiap item berisi: path file, nama kolom teks, dan nama kolom sentimen
    datasets_to_load = [
        {
            "path": "sentiment_ablation/data/Dataset Sentimen kurikulum 2013.xlsx",
            "text_col": "tweet",
            "sentiment_col": "sentiment"
        },
        {
            "path": "sentiment_ablation/data/dataset_tweet_sentimen_tayangan_tv.csv",
            "text_col": "Text Tweet",
            "sentiment_col": "Sentiment"
        },
        {
            "path": "sentiment_ablation/data/dataset_tweet_sentiment_opini_film.csv",
            "text_col": "Text Tweet",
            "sentiment_col": "Sentiment"
        },
        {
            "path": "sentiment_ablation/data/id-tourism-sentimentanalysis.xlsx",
            "text_col": "review",
            "sentiment_col": "sentiment"
        },
        {
            "path": "sentiment_ablation/data/dataset_tweet_sentiment_pilkada_DKI_2017.csv",
            "text_col": "Text Tweet",
            "sentiment_col": "Sentiment"
        }
    ]

    for dataset_info in datasets_to_load:
        file_path = dataset_info["path"]
        try:
            # Membaca file berdasarkan ekstensi
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path, encoding="utf-8")  # Tambahkan encoding jika diperlukan
            elif file_path.endswith(".xlsx"):
                df = pd.read_excel(file_path)  # Gunakan pd.read_excel untuk file Excel
            else:
                print(f"⚠️ Unsupported file format: {file_path}")
                continue

            # Mengambil hanya kolom yang diperlukan dan mengganti namanya menjadi "text" dan "sentiment"
            df = df[[dataset_info["text_col"], dataset_info["sentiment_col"]]].rename(columns={
                dataset_info["text_col"]: "text",
                dataset_info["sentiment_col"]: "sentiment"
            })
            
            print(f"✅ Loaded: {os.path.basename(file_path)}")
            all_data.append(df)

        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
        except KeyError as e:
            print(f"❌ Column error in {file_path}: {e}")
        except Exception as e:
            print(f"❌ General error loading {file_path}: {e}")

    # Gabungkan semua data
    if not all_data:
        print("❌ No datasets were loaded. Please check file paths and formats.")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    combined.dropna(subset=['text', 'sentiment'], inplace=True)

    print("\n🔄 Applying advanced text preprocessing (normalization & stemming)...")
    combined["text"] = combined["text"].apply(preprocess_text_advanced)
    
    combined["sentiment"] = combined["sentiment"].apply(standardize_label)
    combined = combined[combined["sentiment"].isin(["positive", "neutral", "negative"])]
    
    # Pastikan tidak ada teks kosong setelah preprocessing
    combined = combined[combined['text'].str.strip().astype(bool)]

    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    combined["label"] = combined["sentiment"].map(label_map)

    print(f"\n✅ Total data gabungan setelah preprocessing lanjutan: {len(combined)}")
    print(f"📊 Distribusi label:")
    print(combined["sentiment"].value_counts())
    
    return combined