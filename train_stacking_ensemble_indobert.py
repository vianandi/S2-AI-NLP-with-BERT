import torch
import numpy as np
import pandas as pd
import os
import joblib
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
label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# --- 1. Memuat Dataset ---
print("\n📂 Memuat dan membersihkan dataset...")
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
    all_probs = []
    model.eval()
    with torch.no_grad():
        for text in tqdm(texts, desc=f"🔍 Prediksi dari {os.path.basename(model.name_or_path)}"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs[0])
    return np.array(all_probs)

skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
train_idx, test_idx = next(skf.split(df['text'], df['label']))

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

print(f"\n🔄 Menghasilkan meta-features untuk melatih Stacking model...")
print(f"📊 Ukuran data training untuk meta-learner: {len(train_df)}")
print(f"📊 Ukuran data test untuk evaluasi akhir: {len(test_df)}")

oof_lite = get_oof_predictions(model_lite, tokenizer_lite, train_df['text'].tolist())
oof_bertweet = get_oof_predictions(model_bertweet, tokenizer_bertweet, train_df['text'].tolist())

meta_features_train = np.hstack((oof_lite, oof_bertweet))
meta_labels_train = train_df['label'].values

# --- 4. Melatih Model Manajer (Meta-Learner) ---
print("\n💪 Melatih Meta-Learner (Logistic Regression) dengan meta-features...")
meta_learner = LogisticRegression(solver='liblinear', random_state=42)
meta_learner.fit(meta_features_train, meta_labels_train)
print("✅ Meta-Learner berhasil dilatih.")

joblib.dump(meta_learner, os.path.join(output_folder, 'stacking_meta_learner.pkl'))
print(f"💾 Model ensemble (meta-learner) disimpan di: {os.path.join(output_folder, 'stacking_meta_learner.pkl')}")

# --- 5. Evaluasi Menyeluruh pada Test Set (KODE YANG DIPERBAIKI) ---
print("\n🧪 Melakukan evaluasi akhir pada test set...")

test_preds_lite = get_oof_predictions(model_lite, tokenizer_lite, test_df['text'].tolist())
test_preds_bertweet = get_oof_predictions(model_bertweet, tokenizer_bertweet, test_df['text'].tolist())

meta_features_test = np.hstack((test_preds_lite, test_preds_bertweet))
true_labels_test = test_df['label'].values

stacking_predictions = meta_learner.predict(meta_features_test)
stacking_probabilities = meta_learner.predict_proba(meta_features_test)

print("\n" + "="*60)
print("              HASIL EVALUASI ENSEMBLE STACKING")
print("="*60)

# --- BAGIAN YANG DIPERBAIKI ---
unique_labels = np.unique(np.concatenate((true_labels_test, stacking_predictions)))
present_labels = sorted(unique_labels.tolist())
present_target_names = [label_map[label] for label in present_labels]

print(f"Info: Label yang terdeteksi di set tes ini: {present_target_names}")

accuracy = accuracy_score(true_labels_test, stacking_predictions)
f1 = f1_score(true_labels_test, stacking_predictions, average='weighted', labels=present_labels)
report = classification_report(true_labels_test, stacking_predictions, labels=present_labels, target_names=present_target_names, zero_division=0)

# Logika baru untuk menghitung AUC
if len(present_labels) <= 1:
    auc_score = 'N/A (hanya satu kelas terdeteksi)'
elif len(present_labels) == 2:
    # Kasus biner: gunakan probabilitas kelas positif (kolom kedua)
    # stacking_probabilities[:, 1] akan memilih probabilitas dari kelas kedua yang ada.
    auc_score = roc_auc_score(true_labels_test, stacking_probabilities[:, 1])
else:
    # Kasus multi-kelas: gunakan metode One-vs-Rest (OVR)
    auc_score = roc_auc_score(true_labels_test, stacking_probabilities, multi_class='ovr', average='weighted', labels=present_labels)

print(f"🎯 Accuracy: {accuracy:.4f}")
print(f"🎯 F1-Score (Weighted): {f1:.4f}")
# Gunakan f-string formatting yang lebih aman untuk menampilkan hasil AUC
print(f"🎯 AUC: {'{:.4f}'.format(auc_score) if isinstance(auc_score, float) else auc_score}")
print("\n📊 Laporan Klasifikasi Lengkap:")
print(report)
print("="*60)
# --- AKHIR DARI BAGIAN YANG DIPERBAIKI ---

test_df['predicted_label_stacking'] = stacking_predictions
test_df.to_csv(os.path.join(output_folder, 'stacking_evaluation_results.csv'), index=False)
print(f"📄 Hasil prediksi detail disimpan di: {os.path.join(output_folder, 'stacking_evaluation_results.csv')}")