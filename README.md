# Medical Insurance Cost Prediction

End-to-end Machine Learning project untuk memprediksi **biaya asuransi kesehatan (`charges`)** berdasarkan karakteristik peserta seperti usia, jenis kelamin, BMI, jumlah anak, status merokok, dan wilayah tempat tinggal.

Project ini menggunakan beberapa algoritma regresi dan memilih model dengan performa terbaik berdasarkan hasil evaluasi.

---

## 📊 Dataset

Dataset yang digunakan adalah **Medical Insurance Dataset**.

| Informasi                  | Keterangan      |
| -------------------------- | --------------- |
| File                       | `insurance.csv` |
| Jumlah data awal           | 1,338 baris     |
| Jumlah kolom               | 7               |
| Jumlah fitur               | 6               |
| Target                     | `charges`       |
| Missing Value              | 0               |
| Data setelah preprocessing | 1,192 baris     |

### Features

- `age` — usia peserta
- `sex` — jenis kelamin
- `bmi` — Body Mass Index
- `children` — jumlah anak/dependen
- `smoker` — status merokok
- `region` — wilayah tempat tinggal
- `charges` — biaya asuransi sebagai target

---

## 🔬 Machine Learning Workflow

Project ini dilakukan melalui beberapa tahapan:

```text
Dataset
   ↓
Data Loading
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Outlier Detection
   ↓
Outlier Filtering
   ↓
Standardization
   ↓
Duplicate Checking
   ↓
Encoding
   ↓
EDA & Correlation
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Selection
   ↓
Model Saving
   ↓
Flask Deployment
```

---

## 🤖 Models

Tiga model regresi digunakan untuk membandingkan performa:

- Least Angle Regression (LARS)
- Linear Regression
- Gradient Boosting Regressor

### Model Evaluation

| Model                 |        MAE |        MSE |         R² |
| --------------------- | ---------: | ---------: | ---------: |
| LARS                  |     0.7391 |     0.9353 |     0.1567 |
| Linear Regression     |     0.3798 |     0.4420 |     0.6014 |
| **Gradient Boosting** | **0.3323** | **0.3946** | **0.6442** |

Berdasarkan hasil evaluasi, **Gradient Boosting Regressor** menjadi model terbaik karena menghasilkan nilai MAE dan MSE paling rendah serta R² paling tinggi.

---

## 📈 Feature Analysis

Berdasarkan analisis korelasi pada dataset, variabel yang memiliki hubungan paling kuat dengan `charges` adalah:

| Variable   | Correlation |
| ---------- | ----------: |
| `smoker`   |    0.602377 |
| `age`      |    0.436059 |
| `region`   |    0.090858 |
| `children` |    0.082308 |
| `bmi`      |    0.060969 |
| `sex`      |    0.021548 |

Variabel `smoker` memiliki korelasi paling kuat terhadap biaya asuransi, kemudian diikuti oleh `age`.

---

## 💾 Model

Model terbaik disimpan menggunakan Joblib:

```text
gbr_model_medical_insurance.joblib
```

Model dapat dimuat kembali menggunakan:

```python
import joblib

model = joblib.load(
    'gbr_model_medical_insurance.joblib'
)
```

---

## 🚀 Deployment

Model disiapkan untuk digunakan melalui **Flask REST API**.

Endpoint:

```text
POST /predict
```

Server lokal:

```text
http://127.0.0.1:5000
```

API menerima data peserta dan menghasilkan prediksi biaya asuransi.

Contoh input:

```json
{
  "data": [[19, 0, 27.9, 0, 1, 0]]
}
```

Urutan fitur:

```text
age
bmi
children
sex
smoker
region
```

Contoh menjalankan Flask:

```bash
python app.py
```

---

## 📁 Project Structure

```text
Medical-Insurance/
│
├── insurance.csv
├── Medical Insurance.ipynb
├── gbr_model_medical_insurance.joblib
├── app.py
├── medical_insurance.md
└── README.md
```

---

## 📚 Documentation

Dokumentasi lengkap mengenai proses:

- Data Loading
- Data Cleaning
- EDA
- Outlier Detection
- Standardization
- Encoding
- Correlation Analysis
- Data Splitting
- Modeling
- Evaluation
- Model Saving
- Flask Deployment

dapat dilihat pada:

**[`medical_insurance.md`](medical_insurance.md)**

---

## 🛠️ Technologies

Project ini menggunakan:

```text
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-Learn
Joblib
Flask
```

---

## 🎯 Conclusion

Project ini berhasil membangun model Machine Learning untuk memprediksi biaya asuransi kesehatan.

Dari tiga model yang diuji, **Gradient Boosting Regressor** memberikan hasil terbaik dengan **MAE 0.3323, MSE 0.3946, dan R² 0.6442**.

Model tersebut kemudian disimpan menggunakan Joblib dan disiapkan untuk digunakan melalui Flask REST API.

---

## 👨‍💻 Project

**Medical Insurance Cost Prediction**

Machine Learning — Regression  
Dataset: Medical Insurance
