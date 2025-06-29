import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertTokenizerFast, DistilBertModel, DistilBertTokenizerFast, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np
import os
import pandas as pd

from utils import load_and_clean_all_datasets

# --- 1. Definisi Arsitektur Model Hibrida ---

class TCN(nn.Module):
    """Definisi sederhana dari Temporal Convolutional Network."""
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        # Lapisan konvolusi 1D untuk menangkap pola temporal
        self.conv1 = nn.Conv1d(input_size, num_channels, kernel_size, padding='same')
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(num_channels, output_size, kernel_size, padding='same')
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Input shape: (batch, seq_len, features) -> (batch, features, seq_len) untuk konvolusi
        x = x.permute(0, 2, 1)
        out = self.dropout1(self.relu1(self.conv1(x)))
        out = self.dropout2(self.relu2(self.conv2(out)))
        # Mengembalikan ke shape semula
        return out.permute(0, 2, 1)

class HybridClassifierHead(nn.Module):
    """Kepala klasifikasi yang menggabungkan Bi-LSTM dan TCN."""
    def __init__(self, input_size, num_classes):
        super(HybridClassifierHead, self).__init__()
        self.bilstm = nn.LSTM(input_size, 128, bidirectional=True, batch_first=True)
        self.tcn = TCN(input_size, 128, 256, kernel_size=3, dropout=0.2)
        # Output BiLSTM (128*2) + Output TCN (128) = 384
        self.fc = nn.Linear(256 + 128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x adalah last_hidden_state dari Transformer
        lstm_out, _ = self.bilstm(x)
        lstm_features = torch.mean(lstm_out, 1) # Menggunakan rata-rata output sebagai representasi
        
        tcn_out = self.tcn(x)
        tcn_features = torch.mean(tcn_out, 1)

        combined_features = torch.cat((lstm_features, tcn_features), dim=1)
        return self.fc(self.dropout(combined_features))

class FullHybridModel(nn.Module):
    """Model lengkap yang menggabungkan Transformer dengan Hybrid Classifier Head."""
    def __init__(self, transformer_model, num_classes):
        super(FullHybridModel, self).__init__()
        self.transformer = transformer_model
        self.classifier_head = HybridClassifierHead(
            input_size=self.transformer.config.hidden_size,
            num_classes=num_classes
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        return self.classifier_head(last_hidden_state)

# --- 2. Fungsi Helper untuk Pelatihan dan Evaluasi ---

def prepare_dataloader(df, tokenizer, max_len, batch_size):
    """Menyiapkan DataLoader dari DataFrame."""
    input_ids, attention_masks = [], []
    for text in df['text'].values:
        encoded = tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=max_len,
            padding='max_length', truncation=True, return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(df['label'].values)
    
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_pipeline(model, train_dataloader, val_dataloader, epochs, lr, device):
    """Fungsi untuk melatih satu pipeline model."""
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            b_input_ids, b_mask, b_labels = [b.to(device) for b in batch]
            model.zero_grad()
            outputs = model(input_ids=b_input_ids, attention_mask=b_mask)
            loss = loss_fn(outputs, b_labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Rata-rata loss training: {avg_train_loss:.4f}")

        # Evaluasi di setiap akhir epoch
        evaluate_pipeline(model, val_dataloader, device)

def evaluate_pipeline(model, data_loader, device):
    """Mengevaluasi model dan mengembalikan probabilitas."""
    model.eval()
    predictions, true_labels, probabilities = [], [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            b_input_ids, b_mask, b_labels = [b.to(device) for b in batch]
            outputs = model(input_ids=b_input_ids, attention_mask=b_mask)
            
            logits = outputs.detach()
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            probabilities.extend(probs)
            predictions.extend(preds)
            true_labels.extend(b_labels.cpu().numpy())
            
    print("\nLaporan Klasifikasi:")
    print(classification_report(true_labels, predictions, zero_division=0))
    return np.array(probabilities)

# --- 3. Proses Eksekusi Utama ---

if __name__ == '__main__':
    # Pengaturan
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")
    
    MAX_LEN = 128
    BATCH_SIZE = 16 # Turunkan jika terjadi Out-of-Memory
    EPOCHS = 3      # Paper menggunakan 10, kita coba 3 untuk efisiensi
    LR = 2e-5
    
    # Memuat dan menyiapkan data
    df = load_and_clean_all_datasets()
    
    # ==================== BAGIAN YANG DIPERBAIKI ====================
    # Re-map label agar berurutan (misal: [0, 2] menjadi [0, 1]) untuk menghindari error CUDA
    unique_labels_original = sorted(df['label'].unique())
    label_mapping = {original_label: i for i, original_label in enumerate(unique_labels_original)}
    
    df['label'] = df['label'].map(label_mapping)
    print(f"Melakukan re-mapping label. Mapping baru: {label_mapping}")
    
    num_classes = len(df['label'].unique())
    print(f"Jumlah kelas setelah re-mapping: {num_classes}")
    # ==============================================================
    
    # Split data 80% training, 20% testing
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    # Split data training lagi untuk validation
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['label'])

    print(f"\nUkuran Data: Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")

    # --- Pipeline 1: BERT-base ---
    print("\n" + "="*50)
    print(" PIPELINE 1: Melatih BERT-base + BiLSTM-TCN")
    print("="*50)
    
    bert_tokenizer = BertTokenizerFast.from_pretrained('indobenchmark/indobert-base-p1')
    bert_transformer = BertModel.from_pretrained('indobenchmark/indobert-base-p1')
    bert_hybrid_model = FullHybridModel(bert_transformer, num_classes)
    
    bert_train_dataloader = prepare_dataloader(train_df, bert_tokenizer, MAX_LEN, BATCH_SIZE)
    bert_val_dataloader = prepare_dataloader(val_df, bert_tokenizer, MAX_LEN, BATCH_SIZE)
    bert_test_dataloader = prepare_dataloader(test_df, bert_tokenizer, MAX_LEN, BATCH_SIZE)

    train_pipeline(bert_hybrid_model, bert_train_dataloader, bert_val_dataloader, EPOCHS, LR, device)

    # --- Pipeline 2: DistilBERT ---
    print("\n" + "="*50)
    print(" PIPELINE 2: Melatih DistilBERT + BiLSTM-TCN")
    print("="*50)

    distil_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-multilingual-cased')
    distil_transformer = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
    distil_hybrid_model = FullHybridModel(distil_transformer, num_classes)

    distil_train_dataloader = prepare_dataloader(train_df, distil_tokenizer, MAX_LEN, BATCH_SIZE)
    distil_val_dataloader = prepare_dataloader(val_df, distil_tokenizer, MAX_LEN, BATCH_SIZE)
    distil_test_dataloader = prepare_dataloader(test_df, distil_tokenizer, MAX_LEN, BATCH_SIZE)

    train_pipeline(distil_hybrid_model, distil_train_dataloader, distil_val_dataloader, EPOCHS, LR, device)
    
    # --- 4. Evaluasi Ensemble ---
    print("\n" + "="*50)
    print(" EVALUASI ENSEMBLE AKHIR PADA TEST SET")
    print("="*50)

    # Dapatkan probabilitas dari kedua model pada data test
    bert_probs = evaluate_pipeline(bert_hybrid_model, bert_test_dataloader, device)
    distil_probs = evaluate_pipeline(distil_hybrid_model, distil_test_dataloader, device)
    
    # Gabungkan dengan simple averaging
    ensemble_probs = (bert_probs + distil_probs) / 2.0
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    
    true_labels = test_df['label'].values
    
    print("\n--- HASIL AKHIR ENSEMBLE (BERT+DISTILBERT dengan HYBRID CLASSIFIER) ---")
    print(classification_report(true_labels, ensemble_preds, zero_division=0))

