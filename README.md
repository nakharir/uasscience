---

# 3. 📊 Dataset
- **Sumber:** NOAA TAO Project  
- **Jumlah Data:** > 1 juta baris  
- **Tipe Data:** Time Series (Tabular)

### Fitur Utama
| Fitur | Deskripsi |
|------|----------|
| zon.winds | Kecepatan angin zonal |
| mer.winds | Kecepatan angin meridional |
| humidity | Kelembaban udara |
| air temp. | Suhu udara |
| s.s.temp. | Suhu permukaan laut (target) |
| datetime | Waktu observasi |

---

# 4. 🔧 Data Preparation

Tahapan data preparation meliputi:

- **Handling missing values** menggunakan median
- **Outlier analysis** menggunakan boxplot
- **Standardization** menggunakan StandardScaler
- **Time-based splitting** untuk data latih dan uji (80% : 20%)

---

# 5. 🤖 Modeling

Model yang digunakan dalam proyek ini:

- **Model 1 – Baseline:** Linear Regression
- **Model 2 – Advanced ML:** Random Forest Regressor
- **Model 3 – Deep Learning:** LSTM (Long Short-Term Memory)

---

# 6. 🧪 Evaluation

**Metrik Evaluasi (Regresi):**

- MSE
- RMSE
- MAE
- R² Score

### Hasil Singkat

| Model         | R² Score   | Catatan                    |
| ------------- | ---------- | -------------------------- |
| Baseline      | Rendah     | Underfitting               |
| Random Forest | Lebih baik | Menangkap non-linearitas   |
| LSTM          | Tertinggi  | Memanfaatkan pola temporal |

---

# 7. 🏁 Kesimpulan

- **Model terbaik:** LSTM
- **Alasan:** Memiliki error paling kecil dan R² Score tertinggi
- **Insight utama:** Pola temporal sangat berpengaruh dalam prediksi suhu permukaan laut

---

# 8. 🔮 Future Work

- [ ] Menambah data observasi
- [ ] Hyperparameter tuning lanjutan
- [ ] Eksperimen arsitektur DL lain
- [ ] Deployment sebagai web application

---

# 9. 🔁 Reproducibility

Untuk menjalankan proyek ini:

```bash
pip install -r requirements.txt
```
