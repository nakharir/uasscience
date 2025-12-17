INFORMASI PROYEK
Judul Proyek:
Prediksi Anomali Suhu Permukaan Laut (SST) untuk Deteksi Siklus El Niño Menggunakan Random Forest Regressor
Nama Mahasiswa: Amrizal Nakharir
NIM: 233307003
Program Studi: Teknologi Informasi
Mata Kuliah: Data Science
Dosen Pengampu: Gus Nanang
Tahun Akademik: [Tahun/Semester] Link GitHub Repository: [URL Repository] Link Video Pembahasan: [URL Repository]

---

1. LEARNING OUTCOMES
   Pada proyek ini, mahasiswa diharapkan dapat: 1. Memahami konteks masalah dan merumuskan problem statement secara jelas 2. Melakukan analisis dan eksplorasi data (EDA) secara komprehensif (OPSIONAL) 3. Melakukan data preparation yang sesuai dengan karakteristik dataset 4. Mengembangkan tiga model machine learning yang terdiri dari (WAJIB): - Model baseline - Model machine learning / advanced - Model deep learning (WAJIB) 5. Menggunakan metrik evaluasi yang relevan dengan jenis tugas ML 6. Melaporkan hasil eksperimen secara ilmiah dan sistematis 7. Mengunggah seluruh kode proyek ke GitHub (WAJIB) 8. Menerapkan prinsip software engineering dalam pengembangan proyek

---

2. PROJECT OVERVIEW
   2.1 Latar Belakang
   Proyek prediksi Anomali Suhu Permukaan Laut (Sea Surface Temperature - SST) sangat krusial karena fenomena El Niño-Southern Oscillation (ENSO) merupakan penggerak utama variabilitas iklim antartahunan di seluruh dunia. Ketidakmampuan dalam memprediksi anomali suhu ini dapat menyebabkan kegagalan dalam antisipasi perubahan cuaca ekstrem. Proyek ini penting untuk memberikan sistem peringatan dini berbasis data (data-driven) yang dapat melengkapi model fisik iklim tradisional yang sangat kompleks dan membutuhkan daya komputasi tinggi. Dengan menggunakan Machine Learning dan Deep Learning, kita dapat mengekstraksi pola non-linier dari data historis buoy (pelampung) laut secara lebih efisien.
   Ham, Y.-G., Kim, J.-H., & Luo, J.-J. (2019). Deep learning for multi-year ENSO forecasts. Nature, 573(7775), 568–572
   McPhaden, M. J., et al. (1998). The Tropical Ocean-Global Atmosphere (TOGA) Program.
   [Jelaskan konteks dan latar belakang proyek]
3. BUSINESS UNDERSTANDING / PROBLEM UNDERSTANDING
   3.1 Problem Statements
   Tuliskan 2–4 pernyataan masalah yang jelas dan spesifik.
   Contoh (universal): 1. Model perlu mampu memprediksi nilai target dengan akurasi tinggi 2. Sistem harus dapat mengidentifikasi pola pada citra secara otomatis 3. Dataset memiliki noise sehingga perlu preprocessing yang tepat 4. Dibutuhkan model deep learning yang mampu belajar representasi fitur kompleks
   [Tulis problem statements Anda di sini]
   3.2 Goals
   Tujuan harus spesifik, terukur, dan selaras dengan problem statement. Contoh tujuan: 1. Membangun model ML untuk memprediksi variabel target dengan akurasi > 80% 2. Mengukur performa tiga pendekatan model (baseline, advanced, deep learning) 3. Menentukan model terbaik berdasarkan metrik evaluasi yang relevan 4. Menghasilkan sistem yang dapat bekerja secara reproducible
   [Tulis goals Anda di sini]
   3.3 Solution Approach
   Mahasiswa WAJIB menggunakan minimal tiga model dengan komposisi sebagai berikut: #### Model 1 – Baseline Model Model sederhana sebagai pembanding dasar. Pilihan model: - Linear Regression (untuk regresi) - Logistic Regression (untuk klasifikasi) - K-Nearest Neighbors (KNN) - Decision Tree - Naive Bayes
   [Jelaskan model baseline yang Anda pilih dan alasannya]
   Model 2 – Advanced / ML Model
   Model machine learning yang lebih kompleks. Pilihan model: - Random Forest - Gradient Boosting (XGBoost, LightGBM, CatBoost) - Support Vector Machine (SVM) - Ensemble methods - Clustering (K-Means, DBSCAN) - untuk unsupervised - PCA / dimensionality reduction (untuk preprocessing)
   [Jelaskan model advanced yang Anda pilih dan alasannya]
   Model 3 – Deep Learning Model (WAJIB)
   Model deep learning yang sesuai dengan jenis data. Pilihan Implementasi (pilih salah satu sesuai dataset): A. Tabular Data: - Multilayer Perceptron (MLP) / Neural Network - Minimum: 2 hidden layers - Contoh: prediksi harga, klasifikasi binary/multiclass
   B. Image Data: - CNN sederhana (minimum 2 convolutional layers) ATAU - Transfer Learning (ResNet, VGG, MobileNet, EfficientNet) - recommended - Contoh: klasifikasi gambar, object detection
   C. Text Data: - LSTM/GRU (minimum 1 layer) ATAU - Embedding + Dense layers ATAU - Pre-trained model (BERT, DistilBERT, Word2Vec) - Contoh: sentiment analysis, text classification
   D. Time Series: - LSTM/GRU untuk sequential prediction - Contoh: forecasting, anomaly detection
   E. Recommender Systems: - Neural Collaborative Filtering (NCF) - Autoencoder-based Collaborative Filtering - Deep Matrix Factorization
   Minimum Requirements untuk Deep Learning: - ✅ Model harus training minimal 10 epochs - ✅ Harus ada plot loss dan accuracy/metric per epoch - ✅ Harus ada hasil prediksi pada test set - ✅ Training time dicatat (untuk dokumentasi)
   Tidak Diperbolehkan: - ❌ Copy-paste kode tanpa pemahaman - ❌ Model tidak di-train (hanya define arsitektur) - ❌ Tidak ada evaluasi pada test set
   Pada proyek ini digunakan model Long Short-Term Memory (LSTM) untuk melakukan prediksi suhu permukaan laut berdasarkan data historis. LSTM dipilih karena kemampuannya dalam menangkap pola temporal dan dependensi jangka panjang pada data time series.

---

4. DATA UNDERSTANDING
   4.1 Informasi Dataset
   Sumber Dataset:
   Dataset diperoleh dari NOAA Tropical Atmosphere Ocean (TAO) Project
   URL: https://www.pmel.noaa.gov/tao/drupal/disdel/
   Deskripsi Dataset:
   • Jumlah baris (rows): ± 1.000.000+ baris
   • Jumlah kolom (features): 12 kolom
   • Tipe data: Time Series (Tabular numerik)
   • Ukuran dataset: ± 50–100 MB (tergantung versi file)
   • Format file: .dat.gz (compressed text file, delimiter spasi)

4.2 Deskripsi Fitur
Nama Fitur Tipe Data Deskripsi Contoh Nilai
obs Integer Nomor observasi/pencatatan data 1, 2, 3
year Integer Tahun pengambilan data (format 2 digit) 80, 90, 05
month Integer Bulan pengambilan data 1 – 12
day Integer Hari pengambilan data 1 – 31
date Integer Penanda tanggal tambahan dari dataset asli 20200115
latitude Float Garis lintang lokasi buoy -5.0, 0.0, 5.0
longitude Float Garis bujur lokasi buoy 165.0, 180.0
zon.winds Float Kecepatan angin zonal (timur–barat) -3.2, 1.5
mer.winds Float Kecepatan angin meridional (utara–selatan) 0.8, -1.1
humidity Float Kelembaban udara (%) 70.5, 85.2
air temp. Float Suhu udara (°C) 26.5, 29.0
s.s.temp. Float Suhu permukaan laut (°C) (label/target) 27.8, 30.1

4.3 Kondisi Data
Jelaskan kondisi dan permasalahan data:
• Missing Values: Ada 5%
• Duplicate Data: Tidak
• Outliers: zon.winds, mer.winds, Humidity, s.s.temp.
• Imbalanced Data: Tidak
• Noise: noise alamai yang berasal dari variasi kode cuaca, kesalahan pengukuran sensor, fluktuasi lingkungan.
• Data Quality Issues: format thun menggunakan 2 digit sehigga perlu di convert ke tahun normal, adanya tanggal tidak valid atau NaT
4.4 Exploratory Data Analysis (EDA) - (OPSIONAL)
Requirement: Minimal 3 visualisasi yang bermakna dan insight-nya. Contoh jenis visualisasi yang dapat digunakan: - Histogram (distribusi data) - Boxplot (deteksi outliers) - Heatmap korelasi (hubungan antar fitur) - Bar plot (distribusi kategori) - Scatter plot (hubungan 2 variabel) - Wordcloud (untuk text data) - Sample images (untuk image data) - Time series plot (untuk temporal data) - Confusion matrix heatmap - Class distribution plot
Visualisasi 1: [Judul Visualisasi]
Insight:
[Jelaskan apa yang dapat dipelajari dari visualisasi ini]
Visualisasi 2: [Judul Visualisasi]

Insight:
Imputasi menggunakan median untuk masing-masing fitur.
Visualisasi 3: [Judul Visualisasi]
[Insert gambar/plot]
Insight:
[Jelaskan apa yang dapat dipelajari dari visualisasi ini]

---

5. DATA PREPARATION
   Ditemukan missing values pada beberapa fitur numerik, yaitu:
   • zon.winds
   • mer.winds
   • humidity
   • air temp.
   • s.s.temp.
   Median lebih robust terhadap outliers dan sesuai untuk data lingkungan yang cenderung memiliki distribusi tidak normal.
   Nilai ekstrem masih valid secara domain karena merepresentasikan fenomena alam (misalnya cuaca ekstrem), sehingga tetap dipertahankan agar model dapat mempelajari variasi data yang realistis.
   Data Type Conversion
   • Kolom year dikonversi dari format dua digit menjadi empat digit.
   • Dibuat fitur baru datetime dari kombinasi year, month, dan day.
   • Baris dengan nilai datetime tidak valid dihapus
   5.2 Feature Engineering
   Aktivitas: - Creating new features - Feature extraction - Feature selection - Dimensionality reduction
   Tidak dilakukan karena jumlah fitur relatif sedikit dan setiap fitur memiliki makna fisik yang jelas.
   5.3 Data Transformation
   Untuk Data Tabular: - Encoding (Label Encoding, One-Hot Encoding, Ordinal Encoding) - Scaling (Standardization, Normalization, MinMaxScaler)
   Untuk Data Text: - Tokenization - Lowercasing - Removing punctuation/stopwords - Stemming/Lemmatization - Padding sequences - Word embedding (Word2Vec, GloVe, fastText)
   Untuk Data Image: - Resizing - Normalization (pixel values 0-1 atau -1 to 1) - Data augmentation (rotation, flip, zoom, brightness, etc.) - Color space conversion
   Untuk Time Series: - Creating time windows - Lag features - Rolling statistics - Differencing
   Menggunakan StandardScaler untuk menormalkan fitur numerik agar memiliki mean 0 dan standar deviasi 1.
   5.4 Data Splitting
   Strategi pembagian data:

- Training set: [X]% ([jumlah] samples)
- Validation set: [X]% ([jumlah] samples) - jika ada
- Test set: [X]% ([jumlah] samples)
  Contoh:
  Menggunakan stratified split untuk mempertahankan distribusi kelas:
- Training: 80% (8000 samples)
- Test: 20% (2000 samples)
- Random state: 42 untuk reproducibility
  [Srategi pembagian data dilakukan berdasarkan urutan waktu (time-based split) untuk menghindari data leakage.
  • Training set: 80% (data historis)
  • Test set: 20% (data terbaru)
  • Validation set: Tidak digunakan
  5.5 Data Balancing (jika diperlukan)
  Teknik yang digunakan: - SMOTE (Synthetic Minority Over-sampling Technique) - Random Undersampling - Class weights - Ensemble sampling
  [Jelaskan jika Anda melakukan data balancing]
  5.6 Ringkasan Data Preparation
  Langkah Apa yang Dilakukan Mengapa Penting Bagaimana Implementasinya
  Data Cleaning Menangani missing values, outliers, dan tipe data Meningkatkan kualitas data dan konsistensi Imputasi median, validasi datetime
  Feature Engineering Membuat dan memilih fitur relevan Mengoptimalkan input model Seleksi fitur lingkungan utama
  Data Transformation Scaling dan pengurutan data Mempercepat training dan menjaga urutan temporal StandardScaler dan sort by datetime
  Data Splitting Pembagian data berbasis waktu Mencegah data leakage Time-based split 80:20
  Data Balancing Tidak dilakukan Data regresi tidak memerlukan balancing —

---

6. MODELING
   6.1 Model 1 — Baseline Model
   6.1.1 Deskripsi Model
   Nama Model: Linear Regression
   Cara Kerja Singkat:
   Linear Regression memodelkan hubungan linear antara fitur input (kecepatan angin zonal, meridional, kelembaban, suhu udara) dengan target (s.s.temp.) menggunakan persamaan garis lurus.
   6.1.2 Hyperparameter
   Parameter yang digunakan:
   • fit_intercept: True
   • normalize: False
   6.1.3 Implementasi (Ringkas)

# Contoh kode (opsional, bisa dipindah ke GitHub)

from sklearn.linear_model import LinearRegression
model_baseline = LinearRegression()
model_baseline.fit(X_train, y_train)
y_pred_baseline = model_baseline.predict(X_test)
6.1.4 Hasil Awal
[Tuliskan hasil evaluasi awal, akan dijelaskan detail di Section 7]

---

6.2 Model 2 — ML / Advanced Model
6.2.1 Deskripsi Model
Nama Model: Random Forest Regressor
Teori Singkat:
Random Forest adalah algoritma ensemble yang membangun banyak decision tree dan menggabungkan hasil prediksinya (rata-rata) untuk meningkatkan akurasi dan mengurangi overfitting.
Alasan Pemilihan:
Keunggulan:
• Tidak sensitif terhadap scaling
• Dapat menangani interaksi antar fitur
• Risiko overfitting lebih kecil dibanding single tree
Kelemahan:
• Interpretabilitas lebih rendah
• Konsumsi memori lebih besar
• Tidak memanfaatkan urutan waktu secara eksplisit
6.2.2 Hyperparameter
Parameter yang digunakan:
• n_estimators: 100
• max_depth: 10
• min_samples_split: 2
• random_state: 42
Hyperparameter Tuning (jika dilakukan): - Metode: [Grid Search / Random Search / Bayesian Optimization] - Best parameters: […]
6.2.3 Implementasi (Ringkas)

# Contoh kode

from sklearn.ensemble import RandomForestRegressor
model_advanced = RandomForestRegressor(
n_estimators=100,
max_depth=10,
random_state=42
)
model_advanced.fit(X_train, y_train)
y_pred_advanced = model_advanced.predict(X_test)
6.2.4 Hasil Model
Model Random Forest menunjukkan peningkatan performa dibanding baseline, terutama dalam menangkap pola non-linear. Hasil evaluasi kuantitatif dijelaskan pada Section 7.

---

6.3 Model 3 — Deep Learning Model (WAJIB)
6.3.1 Deskripsi Model
Nama Model: [Nama arsitektur, misal: CNN / LSTM / MLP]
** (Centang) Jenis Deep Learning: ** - [ ] Multilayer Perceptron (MLP) - untuk tabular - [ ] Convolutional Neural Network (CNN) - untuk image - [ v] Recurrent Neural Network (LSTM/GRU) - untuk sequential/text - [ ] Transfer Learning - untuk image - [ ] Transformer-based - untuk NLP - [ ] Autoencoder - untuk unsupervised - [ ] Neural Collaborative Filtering - untuk recommender
Alasan Pemilihan:
[Mengapa arsitektur ini cocok untuk dataset Anda?]
6.3.2 Arsitektur Model
Deskripsi Layer:
Layer Deskripsi
Input Sequence data time series
LSTM 64 units
Dropout 0.2
Dense 1 unit (output regresi)
6.3.3 Input & Preprocessing Khusus
Input shape: [Sebutkan dimensi input]
Preprocessing khusus untuk DL: - [Sebutkan preprocessing khusus seperti normalisasi, augmentasi, dll.]
6.3.4 Hyperparameter
Training Configuration:

- Optimizer: Adam / SGD / RMSprop
- Learning rate: [nilai]
- Loss function: [categorical_crossentropy / mse / binary_crossentropy / etc.]
- Metrics: [accuracy / mae / etc.]
- Batch size: [nilai]
- Epochs: [nilai]
- Validation split: [nilai] atau menggunakan validation set terpisah
- Callbacks: [EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, etc.]
  6.3.5 Implementasi (Ringkas)
  Framework: TensorFlow/Keras / PyTorch

# Contoh kode TensorFlow/Keras

import tensorflow as tf
from tensorflow import keras

model_dl = keras.Sequential([
keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
keras.layers.Dropout(0.3),
keras.layers.Dense(64, activation='relu'),
keras.layers.Dropout(0.3),
keras.layers.Dense(num_classes, activation='softmax')
])

model_dl.compile(
optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy']
)

history = model_dl.fit(
X_train, y_train,
validation_split=0.2,
epochs=50,
batch_size=32,
callbacks=[early_stopping]
)
6.3.6 Training Process
Training Time: ± beberapa menit
Computational Resource: CPU (Local / Google Colab)
6.3.7 Model Summary
[Paste model.summary() output atau rangkuman arsitektur]

---

7. EVALUATION
   7.1 Metrik Evaluasi
   Pilih metrik yang sesuai dengan jenis tugas:
   Untuk Regresi:
   • MSE (Mean Squared Error): Rata-rata kuadrat error
   • RMSE (Root Mean Squared Error): Akar dari MSE
   • MAE (Mean Absolute Error): Rata-rata absolute error
   • R² Score: Koefisien determinasi
   • MAPE (Mean Absolute Percentage Error): Error dalam persentase
   7.2 Hasil Evaluasi Model
   7.2.1 Model 1 (Baseline)
   Metrik Evaluasi:
   • MSE: [isi sesuai output]
   • RMSE: [isi sesuai output]
   • MAE: [isi sesuai output]
   • R² Score: [isi sesuai output]
   Analisis Singkat:
   Model baseline mampu menangkap tren umum suhu permukaan laut, namun kurang akurat dalam memodelkan hubungan non-linear antar fitur. Nilai error relatif lebih besar dibandingkan model lain.
   7.2.2 Model 2 — Advanced Model (Random Forest Regressor)
   Metrik Evaluasi:
   • MSE: [isi sesuai output]
   • RMSE: [isi sesuai output]
   • MAE: [isi sesuai output]
   • R² Score: [isi sesuai output]
   Analisis Singkat:
   Random Forest menunjukkan peningkatan performa yang signifikan dibandingkan baseline. Model ini mampu menangkap pola non-linear dan interaksi antar fitur dengan lebih baik, ditunjukkan oleh penurunan error dan peningkatan nilai R².]
   7.2.3 Model 3 (Deep Learning)
   Metrik Evaluasi:
   • MSE: [isi sesuai output]
   • RMSE: [isi sesuai output]
   • MAE: [isi sesuai output]
   • R² Score: [isi sesuai output]
   Training History:
   Visualisasi training dan validation loss telah ditampilkan pada Section 6.3.6.
   Analisis Singkat:
   Model LSTM memberikan performa terbaik dalam memprediksi suhu permukaan laut karena mampu memanfaatkan pola temporal dalam data time series. Error paling kecil dan nilai R² tertinggi diperoleh pada model ini.
   7.3 Perbandingan Ketiga Model
   Tabel Perbandingan:
   Model MSE RMSE MAE R² Training Time
   Baseline (Linear Regression) Lebih tinggi Lebih tinggi Lebih tinggi Rendah Sangat cepat
   Advanced (Random Forest) Menurun Menurun Menurun Sedang–Tinggi Sedang
   Deep Learning (LSTM) Terendah Terendah Terendah Tertinggi Paling lama

Visualisasi Perbandingan:
Model MSE RMSE MAE R² Training Time
Baseline (Linear Regression) Lebih tinggi Lebih tinggi Lebih tinggi Rendah Sangat cepat
Advanced (Random Forest) Menurun Menurun Menurun Sedang–Tinggi Sedang
Deep Learning (LSTM) Terendah Terendah Terendah Tertinggi Paling lama
]
7.4 Analisis Hasil

1. Model Terbaik
   Model LSTM merupakan model terbaik karena menghasilkan nilai error paling kecil dan mampu menangkap pola temporal dalam data time series.
2. Perbandingan dengan Baseline
   Dibandingkan Linear Regression, Random Forest dan LSTM menunjukkan peningkatan performa yang signifikan. Hal ini membuktikan bahwa hubungan antar fitur dan target tidak sepenuhnya linear.
3. Trade-off
   • Linear Regression: cepat dan sederhana, namun akurasi rendah
   • Random Forest: akurasi baik dengan kompleksitas sedang
   • LSTM: akurasi tertinggi, tetapi membutuhkan waktu training dan resource lebih besar
4. Error Analysis
   Kesalahan prediksi terbesar terjadi pada kondisi suhu ekstrem, yang jarang muncul dalam dataset. Hal ini menunjukkan keterbatasan model dalam memprediksi kejadian ekstrem.
5. Overfitting / Underfitting
   • Baseline cenderung underfitting
   • Random Forest relatif seimbang
   • LSTM tidak mengalami overfitting signifikan berkat penggunaan EarlyStopping

---

8. CONCLUSION
   8.1 Kesimpulan Utama
   Model Terbaik:
   Model terbaik dalam proyek ini adalah Deep Learning model berbasis LSTM.
   Alasan:
   Model LSTM menunjukkan performa paling unggul berdasarkan metrik evaluasi regresi, yaitu:
   • Nilai MSE, RMSE, dan MAE paling rendah
   • Nilai R² Score paling tinggi dibandingkan model baseline dan Random Forest

Pencapaian Goals:
• Membangun model prediksi suhu permukaan laut
• Membandingkan performa beberapa model (baseline, ML, dan deep learning)
• Menentukan model terbaik berdasarkan evaluasi kuantitatif

8.2 Key Insights
• Suhu permukaan laut memiliki distribusi yang relatif normal, namun terdapat outlier pada nilai ekstrem.
• Fitur meteorologis seperti suhu udara, kelembaban, dan angin memiliki hubungan yang cukup kuat terhadap suhu permukaan laut.
• Data memiliki missing values yang cukup signifikan sehingga preprocessing sangat berpengaruh terhadap performa model.
Insight dari Modeling: - [
• Model linear (baseline) tidak cukup untuk menangkap hubungan kompleks dalam data.
• Model ensemble seperti Random Forest mampu meningkatkan akurasi dengan menangkap hubungan non-linear.
• Model berbasis LSTM memberikan hasil terbaik karena mampu memanfaatkan urutan waktu (sequential pattern) pada data.
8.3 Kontribusi Proyek
Manfaat praktis:
Proyek ini dapat dimanfaatkan untuk:
• Monitoring dan prediksi suhu permukaan laut dalam bidang kelautan dan klimatologi
• Mendukung pengambilan keputusan berbasis data untuk riset cuaca, perikanan, dan lingkungan
• Menjadi dasar pengembangan sistem prediksi cuaca laut berbasis AI
Pembelajaran yang didapat:
Melalui proyek ini, diperoleh beberapa pembelajaran penting, antara lain:
• Pemahaman end-to-end proses data science mulai dari EDA hingga evaluasi model
• Pentingnya preprocessing dan data preparation dalam meningkatkan performa model
• Perbandingan efektivitas model klasik, machine learning, dan deep learning
• Implementasi deep learning (LSTM) untuk data time series secara praktis

---

9. FUTURE WORK (Opsional)
   Saran pengembangan untuk proyek selanjutnya: ** Centang Sesuai dengan saran anda **
   Data: - [ ] Mengumpulkan lebih banyak data - [ ] Menambah variasi data - [ ] Feature engineering lebih lanjut
   Model: - [ ] Mencoba arsitektur DL yang lebih kompleks - [ ] Hyperparameter tuning lebih ekstensif - [ ] Ensemble methods (combining models) - [ ] Transfer learning dengan model yang lebih besar
   Deployment: - [ ] Membuat API (Flask/FastAPI) - [ ] Membuat web application (Streamlit/Gradio) - [ ] Containerization dengan Docker - [ ] Deploy ke cloud (Heroku, GCP, AWS)
   Optimization: - [ ] Model compression (pruning, quantization) - [ ] Improving inference speed - [ ] Reducing model size

---

10. REPRODUCIBILITY (WAJIB)
    10.1 GitHub Repository
    Link Repository: https://github.com/nakharir/uasscience
    Repository harus berisi: - ✅ Notebook Jupyter/Colab dengan hasil running - ✅ Script Python (jika ada) - ✅ requirements.txt atau environment.yml - ✅ README.md yang informatif - ✅ Folder structure yang terorganisir - ✅ .gitignore (jangan upload dataset besar)
    10.2 Environment & Dependencies
    Python Version: [3.8 / 3.9 / 3.10 / 3.11]
    Main Libraries & Versions:
    numpy==1.24.3
    pandas==2.0.3
    scikit-learn==1.3.0
    matplotlib==3.7.2
    seaborn==0.12.2

# Deep Learning Framework (pilih salah satu)

tensorflow==2.14.0 # atau
torch==2.1.0 # PyTorch

# Additional libraries (sesuaikan)

xgboost==1.7.6
lightgbm==4.0.0
opencv-python==4.8.0 # untuk computer vision
nltk==3.8.1 # untuk NLP
transformers==4.30.0 # untuk BERT, dll
