import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np
import os

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
        self.bilstm = nn.LSTM(input_size=self.bert.config.hidden_size, hidden_size=128, num_layers=2,
                              bidirectional=True, batch_first=True, dropout=0.2)
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

print("\n🚀 Memulai proses training...")
for epoch in range(EPOCHS):
    print(f'\n--- Epoch {epoch + 1}/{EPOCHS} ---')
    train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer, scheduler, len(train_data))
    print(f'Train loss: {train_loss}')
    val_loss, _, _ = eval_model(model, val_dataloader, loss_fn, len(val_data))
    print(f'Validation loss: {val_loss}')

# --- 6. Simpan Model Hibrida ---
output_dir = '/models/hybrid_indobertweet_bilstm/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f"\n💾 Menyimpan model hibrida terlatih ke {output_dir}")
torch.save(model.state_dict(), os.path.join(output_dir, 'model.bin'))
tokenizer.save_pretrained(output_dir)

print("\n✅ Pelatihan model hibrida selesai.")