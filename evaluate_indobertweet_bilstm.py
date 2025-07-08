import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import load_and_clean_all_datasets

# --- 1. Pengaturan Awal ---
print("🚀 Memulai Pelatihan Model Hibrida: IndoBERTweet + Bi-LSTM")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Menggunakan device: {device}")

MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# --- 2. Memuat dan Menyiapkan Data (Dengan Perbaikan) ---
print("\n📂 Memuat dan memproses dataset...")
df = load_and_clean_all_datasets()
df.dropna(subset=['label'], inplace=True)
df['label'] = df['label'].astype(int)

# ==================== BAGIAN YANG DIPERBAIKI ====================
# Re-map label agar berurutan (misal: [0, 2] menjadi [0, 1])
unique_labels_original = sorted(df.label.unique())
label_mapping = {original_label: i for i, original_label in enumerate(unique_labels_original)}

df['label'] = df['label'].map(label_mapping)
print(f"Melakukan re-mapping label. Mapping baru: {label_mapping}")

# Dapatkan jumlah kelas dari data yang sudah di-remap
num_classes = len(df.label.unique())
print(f"Jumlah kelas setelah re-mapping: {num_classes}")
print(f"Nilai unik di label setelah re-mapping: {df.label.unique()}")

# Pengecekan keamanan yang sekarang akan berhasil
min_label, max_label = df.label.min(), df.label.max()
if min_label < 0 or max_label >= num_classes:
    raise ValueError(f"Error: Label di luar rentang. Ditemukan min: {min_label}, max: {max_label}. Harusnya antara 0 dan {num_classes-1}")
else:
    print("✅ Pengecekan rentang label berhasil.")
# ==============================================================

texts = df.text.values
labels = df.label.values

tokenizer = BertTokenizerFast.from_pretrained('indolem/indobertweet-base-uncased')

# Tokenisasi semua data
input_ids, attention_masks = [], []
for text in tqdm(texts, desc="Tokenisasi Data"):
    encoded_dict = tokenizer.encode_plus(
                        text, add_special_tokens=True, max_length=MAX_LEN,
                        padding='max_length', truncation=True,
                        return_attention_mask=True, return_tensors='pt')
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Membagi data
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, random_state=42, test_size=0.1, stratify=labels)
train_masks, val_masks, _, _ = train_test_split(attention_masks, labels, random_state=42, test_size=0.1, stratify=labels)

# Membuat DataLoader
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)

# --- 3. Mendefinisikan Arsitektur Model Hibrida ---
class HybridClassifier(nn.Module):
    def __init__(self, n_classes):
        super(HybridClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('indolem/indobertweet-base-uncased')
        self.bilstm = nn.LSTM(input_size=self.bert.config.hidden_size, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(p=0.3)
        self.out = nn.Linear(128 * 2, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        lstm_output, (h_n, c_n) = self.bilstm(last_hidden_state)
        hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        output = self.dropout(hidden)
        return self.out(output)

print("\n🏗️ Membuat instance model hibrida...")
model = HybridClassifier(n_classes=num_classes)
model.to(device)

# --- 4. Menyiapkan Optimizer dan Scheduler ---
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_fn = nn.CrossEntropyLoss().to(device)

# --- 5. Training Loop ---
def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, n_examples):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)
        loss = loss_fn(outputs, b_labels)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return total_loss / len(data_loader)

def eval_model(model, data_loader, loss_fn, n_examples):
    model.eval()
    total_loss = 0
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)
            loss = loss_fn(outputs, b_labels)
            total_loss += loss.item()
            logits = outputs.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predictions.extend(np.argmax(logits, axis=1).flatten())
            true_labels.extend(label_ids.flatten())
    avg_loss = total_loss / len(data_loader)
    print("\nLaporan Klasifikasi Validasi:")
    print(classification_report(true_labels, predictions, zero_division=0))
    return avg_loss, true_labels, predictions

def save_evaluation_results(true_labels, predictions, train_losses, val_losses, output_folder="resultevaluateindobertweetbilstm"):
    """
    Membuat dan menyimpan hasil evaluasi dalam bentuk visualisasi
    """
    # Buat folder output jika belum ada
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"✅ Membuat folder output: {output_folder}")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(true_labels, predictions)
    
    # Map label numerik ke teks
    label_map_reverse = {v: k for k, v in label_mapping.items()}
    sentiment_labels = {
        0: 'Negatif',
        1: 'Netral', 
        2: 'Positif'
    }
    
    label_names = [sentiment_labels.get(label_map_reverse.get(i, i), f"Class {i}") for i in range(len(np.unique(true_labels)))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Prediksi')
    plt.ylabel('Label Asli')
    plt.title('Confusion Matrix - IndoBERTweet + BiLSTM')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'), dpi=300)
    print(f"✅ Confusion matrix disimpan ke {output_folder}/confusion_matrix.png")
    plt.close()
    
    # 2. Metrics Bar Chart
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted', zero_division=0)
    accuracy = np.mean(np.array(true_labels) == np.array(predictions))
    
    metrics = {
        'Akurasi': accuracy,
        'Presisi': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics.keys(), metrics.values(), color=['#106EC6', '#11A1C3', '#039CDA', '#13DCA5'])
    plt.ylim(0, 1.0)
    plt.title('Metrik Klasifikasi - IndoBERTweet + BiLSTM')
    plt.ylabel('Nilai')
    
    # Tambahkan nilai di atas bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'metrics.png'), dpi=300)
    print(f"✅ Grafik metrik disimpan ke {output_folder}/metrics.png")
    plt.close()
    
    # 3. Loss Curves
    if len(train_losses) > 1:
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')
        plt.title('Kurva Loss Training dan Validasi - IndoBERTweet + BiLSTM')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'loss_curves.png'), dpi=300)
        print(f"✅ Kurva loss disimpan ke {output_folder}/loss_curves.png")
        plt.close()
    
    # 4. Classification Report sebagai tabel
    from sklearn.metrics import classification_report
    report = classification_report(true_labels, predictions, 
                                labels=range(num_classes),
                                target_names=label_names,
                                output_dict=True, 
                                zero_division=0)
    
    report_df = pd.DataFrame(report).transpose()
    
    # Simpan sebagai CSV
    report_df.to_csv(os.path.join(output_folder, 'classification_report.csv'))
    
    # Visualisasi tabel
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=report_df.round(4).values,
                    rowLabels=report_df.index,
                    colLabels=report_df.columns,
                    cellLoc='center', 
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title('Classification Report - IndoBERTweet + BiLSTM')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'classification_report.png'), dpi=300)
    print(f"✅ Classification report disimpan ke {output_folder}/classification_report.png")
    plt.close()

# Modifikasi training loop untuk melacak loss
train_losses = []
val_losses = []

print("\n🚀 Memulai proses training...")
for epoch in range(EPOCHS):
    print(f'\n--- Epoch {epoch + 1}/{EPOCHS} ---')
    train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer, scheduler, len(train_data))
    print(f'Train loss: {train_loss}')
    train_losses.append(train_loss)
    
    val_loss, true_vals, predictions = eval_model(model, val_dataloader, loss_fn, len(val_data))
    print(f'Validation loss: {val_loss}')
    val_losses.append(val_loss)

# Evaluasi akhir
print("\n🔍 Melakukan evaluasi akhir model...")
final_val_loss, final_true_labels, final_predictions = eval_model(model, val_dataloader, loss_fn, len(val_data))

# Simpan hasil evaluasi sebagai PNG
print("\n📊 Menyimpan hasil evaluasi dalam bentuk visualisasi...")
save_evaluation_results(
    final_true_labels, 
    final_predictions, 
    train_losses, 
    val_losses
)

# --- 6. Simpan Model Hibrida ---
output_dir = './models/hybrid_indobertweet_bilstm/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f"\n💾 Menyimpan model hibrida terlatih ke {output_dir}")
torch.save(model.state_dict(), os.path.join(output_dir, 'model.bin'))
tokenizer.save_pretrained(output_dir)

print("\n✅ Pelatihan model hibrida selesai.")