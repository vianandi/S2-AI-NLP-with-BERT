import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizerFast, BertModel, DistilBertModel, DistilBertTokenizerFast, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np
import os

from utils import load_and_clean_all_datasets

# ====================================================================================
# --- 1. PENGATURAN UTAMA: PILIH MODEL YANG AKAN DILATIH DI SINI ---
#
# Ganti nilai 'model_name' ke salah satu dari:
# 'indobertweet', 'indobert-lite', 'bert-base', atau 'distilbert'
#
model_name_to_train = 'indobert-lite' # <--- GANTI DI SINI UNTUK EKSPERIMEN BERBEDA
# ====================================================================================


# --- Konfigurasi Detail untuk Setiap Model ---
MODEL_CONFIG = {
    'indobertweet': {
        'path': 'indolem/indobertweet-base-uncased',
        'model_class': BertModel,
        'tokenizer_class': BertTokenizerFast
    },
    'indobert-lite': {
        'path': 'indobenchmark/indobert-lite-base-p1',
        'model_class': BertModel,
        'tokenizer_class': BertTokenizerFast
    },
    'bert-base': {
        'path': 'indobenchmark/indobert-base-p1',
        'model_class': BertModel,
        'tokenizer_class': BertTokenizerFast
    },
    'distilbert': {
        'path': 'distilbert-base-multilingual-cased',
        'model_class': DistilBertModel,
        'tokenizer_class': DistilBertTokenizerFast
    }
}

# --- Memuat Konfigurasi yang Dipilih ---
if model_name_to_train not in MODEL_CONFIG:
    raise ValueError(f"Nama model '{model_name_to_train}' tidak valid. Pilih dari: {list(MODEL_CONFIG.keys())}")

config = MODEL_CONFIG[model_name_to_train]
print(f"🚀 Memulai Pelatihan Model Hibrida untuk: {model_name_to_train.upper()}")

# --- Pengaturan Pelatihan ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Menggunakan device: {device}")

MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# --- 2. Memuat dan Menyiapkan Data ---
print("\n📂 Memuat dan memproses dataset...")
df = load_and_clean_all_datasets()
df.dropna(subset=['label'], inplace=True)
df['label'] = df['label'].astype(int)

# Re-map label agar berurutan
unique_labels_original = sorted(df.label.unique())
label_mapping = {original_label: i for i, original_label in enumerate(unique_labels_original)}
df['label'] = df['label'].map(label_mapping)
num_classes = len(df.label.unique())
print(f"Jumlah kelas yang terdeteksi: {num_classes}")

# --- Tokenisasi Data ---
tokenizer = config['tokenizer_class'].from_pretrained(config['path'])
texts = df.text.values
labels = df.label.values

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

# --- Membagi Data ---
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, random_state=42, test_size=0.1, stratify=labels)
train_masks, val_masks, _, _ = train_test_split(attention_masks, labels, random_state=42, test_size=0.1, stratify=labels)

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)

# --- 3. Mendefinisikan Arsitektur Model Hibrida ---
class HybridClassifier(nn.Module):
    def __init__(self, n_classes):
        super(HybridClassifier, self).__init__()
        self.transformer = config['model_class'].from_pretrained(config['path'])
        self.bilstm = nn.LSTM(input_size=self.transformer.config.hidden_size, hidden_size=128, num_layers=2,
                              bidirectional=True, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(p=0.3)
        self.out = nn.Linear(128 * 2, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        lstm_output, (h_n, c_n) = self.bilstm(last_hidden_state)
        hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        output = self.dropout(hidden)
        return self.out(output)

print("\n🏗️ Membuat instance model hibrida...")
model = HybridClassifier(n_classes=num_classes)
model.to(device)

# --- 4. Optimizer dan Scheduler ---
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_fn = nn.CrossEntropyLoss().to(device)

# --- 5. Training Loop (Fungsi tidak berubah) ---
def train_epoch(model, data_loader, loss_fn, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        b_input_ids, b_input_mask, b_labels = [b.to(device) for b in batch]
        outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)
        loss = loss_fn(outputs, b_labels)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return total_loss / len(data_loader)

def eval_model(model, data_loader, loss_fn):
    model.eval()
    total_loss = 0
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            b_input_ids, b_input_mask, b_labels = [b.to(device) for b in batch]
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
    return avg_loss

# --- 6. Proses Training Utama ---
print("\n🚀 Memulai proses training...")
for epoch in range(EPOCHS):
    print(f'\n--- Epoch {epoch + 1}/{EPOCHS} ---')
    train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer, scheduler)
    print(f'Train loss: {train_loss:.4f}')
    val_loss = eval_model(model, val_dataloader, loss_fn)
    print(f'Validation loss: {val_loss:.4f}')

# --- 7. Simpan Model Hibrida ---
output_dir = f'./models/hybrid_{model_name_to_train}_bilstm/'
os.makedirs(output_dir, exist_ok=True)
print(f"\n💾 Menyimpan model hibrida terlatih ke {output_dir}")
torch.save(model.state_dict(), os.path.join(output_dir, 'model.bin'))
tokenizer.save_pretrained(output_dir)

print("\n✅ Pelatihan model hibrida selesai.")