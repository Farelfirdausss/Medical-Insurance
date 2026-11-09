# Medical Insurance Cost Prediction

End-to-end Machine Learning project untuk memprediksi **biaya asuransi kesehatan (`charges`)** berdasarkan karakteristik peserta, seperti usia, jenis kelamin, BMI, jumlah anak, status merokok, dan wilayah tempat tinggal.

Project ini menggunakan beberapa algoritma regresi untuk membandingkan performa model dan memilih algoritma terbaik berdasarkan hasil evaluasi. Model terpilih kemudian diimplementasikan ke dalam aplikasi web interaktif menggunakan **Streamlit**.

---

## 📊 Dataset

Dataset yang digunakan adalah **Medical Insurance Dataset**.

| Informasi                  | Keterangan      |
| -------------------------- | --------------- |
| Nama File                  | `insurance.csv` |
| Jumlah Data Awal           | 1.338 baris     |
| Jumlah Kolom               | 7               |
| Jumlah Fitur               | 6               |
| Target                     | `charges`       |
| Missing Values             | 0               |
| Data Setelah Preprocessing | 1.192 baris     |

### Features

| Fitur      | Keterangan                             |
| ---------- | -------------------------------------- |
| `age`      | Usia peserta                           |
| `sex`      | Jenis kelamin                          |
| `bmi`      | Body Mass Index                        |
| `children` | Jumlah anak atau tanggungan            |
| `smoker`   | Status merokok                         |
| `region`   | Wilayah tempat tinggal                 |
| `charges`  | Biaya asuransi sebagai target prediksi |

---

## 🔬 Machine Learning Workflow

Tahapan yang dilakukan dalam project ini meliputi:

```text
Dataset
   ↓
Data Loading
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Outlier Detection (IQR)
   ↓
Outlier Filtering (1,338 → 1,192 baris)
   ↓
Standardization (StandardScaler)
   ↓
Duplicate Checking
   ↓
Encoding (One-Hot Encoding)
   ↓
EDA & Correlation Analysis
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Selection
   ↓
Streamlit Deployment
```

---

## 🤖 Machine Learning Models

Tiga algoritma regresi digunakan untuk membandingkan performa prediksi biaya asuransi kesehatan.

1. Least Angle Regression (LARS)
2. Linear Regression
3. Gradient Boosting Regressor

### Model Evaluation

Hasil evaluasi masing-masing model adalah sebagai berikut:

| Model             |        MAE |        MSE |         R² |
| ----------------- | ---------: | ---------: | ---------: |
| LARS              |     0.7391 |     0.9353 |     0.1567 |
| Linear Regression |     0.3798 |     0.4420 |     0.6014 |
| Gradient Boosting | **0.3323** | **0.3946** | **0.6442** |

Berdasarkan hasil evaluasi, **Gradient Boosting Regressor** menjadi model terbaik karena menghasilkan nilai MAE dan MSE paling rendah serta nilai R² paling tinggi dibandingkan kedua model lainnya.

---

## 📈 Feature Analysis

Berdasarkan analisis korelasi pada dataset, diperoleh hasil sebagai berikut:

| Variabel   | Korelasi |
| ---------- | -------: |
| `smoker`   | 0.602377 |
| `age`      | 0.436059 |
| `region`   | 0.090858 |
| `children` | 0.082308 |
| `bmi`      | 0.060969 |
| `sex`      | 0.021548 |

Hasil analisis menunjukkan bahwa variabel `smoker` memiliki hubungan paling kuat dengan biaya asuransi (`charges`), kemudian diikuti oleh variabel `age`.

---

## 💾 Model Implementation

Model terbaik yang digunakan dalam aplikasi adalah **Gradient Boosting Regressor**.

Pada versi deployment ini, model tidak lagi dimuat dari file `.joblib`, melainkan dilatih langsung di dalam `app.py` ketika aplikasi dijalankan.

Contoh implementasi:

```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(random_state=42)
model.fit(X, y)
```

Pendekatan ini digunakan agar aplikasi tidak bergantung pada file model hasil training yang berpotensi mengalami ketidakcocokan versi Python atau library antara lingkungan lokal dan Streamlit Cloud.

File `gbr_model_medical_insurance.joblib` tetap disimpan di repository sebagai arsip hasil training dari notebook, tetapi tidak digunakan oleh aplikasi Streamlit.

---

## 🚀 Deployment - Streamlit

Aplikasi prediksi biaya asuransi kesehatan dikembangkan menggunakan **Streamlit** agar pengguna dapat melakukan prediksi melalui browser.

### 1. Menjalankan Aplikasi Secara Lokal

Install seluruh library yang dibutuhkan:

```bash
pip install -r requirements.txt
```

Jalankan aplikasi:

```bash
streamlit run app.py
```

Setelah berhasil dijalankan, buka browser melalui alamat:

```text
http://localhost:8501
```

### 2. Deployment Menggunakan Streamlit Cloud

Langkah-langkah deployment:

1. Pastikan seluruh file project sudah diunggah ke GitHub.
2. Buka [Streamlit Community Cloud](https://share.streamlit.io/).
3. Login menggunakan akun GitHub.
4. Pilih repository `Medical-Insurance`.
5. Pilih branch `main`.
6. Tentukan file utama `app.py`.
7. Klik **Deploy**.

### 3. Cara Menggunakan Aplikasi

1. Masukkan usia (`age`).
2. Masukkan nilai BMI (`bmi`).
3. Masukkan jumlah anak (`children`).
4. Pilih jenis kelamin (`sex`).
5. Pilih status merokok (`smoker`).
6. Pilih wilayah tempat tinggal (`region`).
7. Klik tombol **Prediksi Biaya**.
8. Hasil estimasi biaya asuransi akan ditampilkan dalam mata uang Dollar ($).

---

## 📁 Project Structure

Struktur folder repository:

```text
Medical-Insurance/
│
├── insurance.csv
├── Medical Insurance.ipynb
├── gbr_model_medical_insurance.joblib
├── app.py
├── requirements.txt
├── medical_insurance.md
└── README.md
```

### Keterangan File

| File                                 | Fungsi                                                  |
| ------------------------------------ | ------------------------------------------------------- |
| `insurance.csv`                      | Dataset Medical Insurance                               |
| `Medical Insurance.ipynb`            | Notebook proses analisis dan pemodelan Machine Learning |
| `gbr_model_medical_insurance.joblib` | Arsip model hasil training                              |
| `app.py`                             | Aplikasi prediksi menggunakan Streamlit                 |
| `requirements.txt`                   | Daftar library yang dibutuhkan                          |
| `medical_insurance.md`               | Dokumentasi lengkap project                             |
| `README.md`                          | Informasi utama repository                              |

---

## 📚 Documentation

Dokumentasi lengkap mengenai tahapan pengerjaan project dapat dilihat pada file:

**`medical_insurance.md`**

Dokumentasi tersebut mencakup:

- Data Loading
- Data Understanding
- Data Cleaning
- Outlier Detection
- Standardization
- Exploratory Data Analysis (EDA)
- Encoding
- Correlation Analysis
- Data Splitting
- Model Training
- Model Evaluation
- Model Saving
- Streamlit Deployment

---

## 🛠️ Technologies

Teknologi dan library yang digunakan dalam project ini:

- **Python** — Bahasa pemrograman utama.
- **Pandas** — Pengolahan dan manipulasi dataset.
- **NumPy** — Operasi numerik.
- **Scikit-Learn** — Preprocessing dan pemodelan Machine Learning.
- **Joblib** — Penyimpanan model hasil training.
- **Streamlit** — Pengembangan aplikasi web interaktif.

---

## 🎯 Conclusion

Project ini berhasil membangun sistem Machine Learning untuk memprediksi biaya asuransi kesehatan berdasarkan karakteristik peserta.

Dari tiga algoritma regresi yang diuji, Gradient Boosting Regressor memberikan performa terbaik dengan nilai MAE sebesar **0.3323**, MSE sebesar **0.3946**, dan R² sebesar **0.6442**.

Model tersebut kemudian diimplementasikan ke dalam aplikasi web interaktif menggunakan Streamlit. Dengan demikian, pengguna dapat memasukkan karakteristik peserta dan memperoleh estimasi biaya asuransi secara langsung melalui aplikasi.

---

## 👨‍💻 Project Information

**Project:** Medical Insurance Cost Prediction  
**Category:** Machine Learning - Regression  
**Dataset:** Medical Insurance Dataset  
**Deployment:** Streamlit  
**Repository:** GitHub
