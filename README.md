# Buat environment Python (opsional tapi direkomendasikan)
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

1. Persiapan Data
Pastikan data sentiment berada di lokasi yang sesuai (sentiment_ablation/data/INA_TweetsPPKM_Labeled_Pure.csv)


2. Melatih Model (waktu eksekusi lama)
# Model BERT
python train_bert.py

# Model DistilBERT
python train_distilbert.py

# Model IndoBERT Lite
python train_indobert_lite.py

# Model IndoBERTweet
python train_indobertweet.py


3. Evaluasi Model
# Evaluasi BERT & DistilBERT
python evaluate.py

# Evaluasi IndoBERT Lite & IndoBERTweet
python evaluate_IndoBERT-Lite_bertweet.py


4. Benchmark Model
# Benchmark BERT & DistilBERT
python benchmark_models.py

# Benchmark IndoBERT Lite & IndoBERTweet
python benchmark_IndoBERT-Lite_indobertweet.py


5. Hasil dan Laporan
# Hasilkan tabel dan visualisasi awal
python result.py

# Hasilkan hasil akhir dan perbandingan menyeluruh
python generate_final_results.py


Output yang Dihasilkan
Setelah menjalankan semua kode, Anda akan mendapatkan:
1. Model terlatih di folder models
2. Tabel perbandingan (CSV dan Markdown)
3. Visualisasi performa model (PNG)
4. Log pelatihan dan evaluasi

Catatan
1. Proses pelatihan membutuhkan GPU untuk kinerja optimal
2. Pastikan memiliki ruang penyimpanan yang cukup (16GB) untuk menyimpan model dan requirements python
3. Seluruh proses bisa memakan waktu 3-5 jam tergantung spesifikasi hardware
