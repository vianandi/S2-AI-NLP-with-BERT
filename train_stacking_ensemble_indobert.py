import torch
import numpy as np
import pandas as pd
import os
import joblib # Untuk menyimpan model meta-learner
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score

from transformers import AutoModelForSequenceClassification, BertTokenizerFast, BertForSequenceClassification
from utils import load_and_clean_all_datasets

print("🔧 PyTorch version:", torch.__version__)
print("🚀 CUDA available:", torch.cuda.is_available())
print("🖥️  Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

# --- Pengaturan ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_folder = 'resultmodels/stacking_ensemble'
os.makedirs(output_folder, exist_ok=True)
label_names = ['Negative', 'Neutral', 'Positive']

# --- 1. Memuat Dataset ---
print("\n📂 Memuat dan membersihkan dataset...")
# Menggunakan fungsi dari utils.py yang sudah mencakup preprocessing tingkat lanjut
df = load_and_clean_all_datasets() 
print(f"✅ Total data setelah cleaning: {len(df)}")

# --- 2. Memuat Model Dasar (Base Models) ---
def load_model(model_path, tokenizer_class, model_class, model_name):
    if not os.path.exists(model_path):
        print(f"❌ Model '{model_name}' tidak ditemukan di {model_path}")
        return None, None
    print(f"🔧 Memuat {model_name} dari {model_path}...")
    tokenizer = tokenizer_class.from_pretrained(model_path)
    model = model_class.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print(f"✅ {model_name} berhasil dimuat.")
    return model, tokenizer

print("\n🤖 Memuat Model-model Dasar...")
model_lite, tokenizer_lite = load_model("models/indobertlite/final", BertTokenizerFast, AutoModelForSequenceClassification, "IndoBERT Lite")
model_bertweet, tokenizer_bertweet = load_model("models/indobertweet/final", BertTokenizerFast, BertForSequenceClassification, "IndoBERTweet")

if model_lite is None or model_bertweet is None:
    print("❌ Gagal memuat salah satu atau kedua model dasar. Proses dihentikan.")
    exit()

# --- 3. Menghasilkan Prediksi Out-of-Fold (Meta-Features) ---
def get_oof_predictions(model, tokenizer, texts):
    """Fungsi untuk mendapatkan prediksi (probabilitas) dari satu model"""
    all_probs = []
    model.eval()
    with torch.no_grad():
        for text in tqdm(texts, desc=f"🔍 Prediksi dari {model.name_or_path.split('/')[-1]}"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs[0])
    return np.array(all_probs)

# Membagi data untuk training meta-learner dan final test
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
train_idx, test_idx = next(skf.split(df['text'], df['label']))

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

print(f"\n🔄 Menghasilkan meta-features untuk melatih Stacking model...")
print(f"📊 Ukuran data training untuk meta-learner: {len(train_df)}")
print(f"📊 Ukuran data test untuk evaluasi akhir: {len(test_df)}")

# Dapatkan probabilitas dari setiap model pada data training
oof_lite = get_oof_predictions(model_lite, tokenizer_lite, train_df['text'].tolist())
oof_bertweet = get_oof_predictions(model_bertweet, tokenizer_bertweet, train_df['text'].tolist())

# Gabungkan probabilitas menjadi fitur baru (meta-features)
# (Probabilitas kelas 0,1,2 dari model A) + (Probabilitas kelas 0,1,2 dari model B)
meta_features_train = np.hstack((oof_lite, oof_bertweet))
meta_labels_train = train_df['label'].values

# --- 4. Melatih Model Manajer (Meta-Learner) ---
print("\n💪 Melatih Meta-Learner (Logistic Regression) dengan meta-features...")
meta_learner = LogisticRegression(solver='liblinear', random_state=42)
meta_learner.fit(meta_features_train, meta_labels_train)
print("✅ Meta-Learner berhasil dilatih.")

# Simpan model meta-learner yang sudah dilatih
joblib.dump(meta_learner, os.path.join(output_folder, 'stacking_meta_learner.pkl'))
print(f"💾 Model ensemble (meta-learner) disimpan di: {os.path.join(output_folder, 'stacking_meta_learner.pkl')}")

# --- 5. Evaluasi Menyeluruh pada Test Set ---
print("\n🧪 Melakukan evaluasi akhir pada test set...")

# Pertama, dapatkan prediksi dari model dasar pada test set
test_preds_lite = get_oof_predictions(model_lite, tokenizer_lite, test_df['text'].tolist())
test_preds_bertweet = get_oof_predictions(model_bertweet, tokenizer_bertweet, test_df['text'].tolist())

# Buat meta-features untuk test set
meta_features_test = np.hstack((test_preds_lite, test_preds_bertweet))
true_labels_test = test_df['label'].values

# Gunakan meta-learner untuk membuat prediksi akhir
stacking_predictions = meta_learner.predict(meta_features_test)
stacking_probabilities = meta_learner.predict_proba(meta_features_test)

print("\n" + "="*60)
print("              HASIL EVALUASI ENSEMBLE STACKING")
print("="*60)

# Hitung semua metrik yang diminta
accuracy = accuracy_score(true_labels_test, stacking_predictions)
f1 = f1_score(true_labels_test, stacking_predictions, average='weighted')
report = classification_report(true_labels_test, stacking_predictions, target_names=label_names)
auc_score = roc_auc_score(true_labels_test, stacking_probabilities, multi_class='ovr', average='weighted')

print(f"🎯 Accuracy: {accuracy:.4f}")
print(f"🎯 F1-Score (Weighted): {f1:.4f}")
print(f"🎯 AUC (Weighted OVR): {auc_score:.4f}\n")
print("📊 Laporan Klasifikasi Lengkap:")
print(report)
print("="*60)

# Simpan hasil prediksi untuk analisis error
test_df['predicted_label_stacking'] = stacking_predictions
test_df.to_csv(os.path.join(output_folder, 'stacking_evaluation_results.csv'), index=False)
print(f"📄 Hasil prediksi detail disimpan di: {os.path.join(output_folder, 'stacking_evaluation_results.csv')}")