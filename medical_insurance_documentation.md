# Medical Insurance Cost Prediction - Deployment Ready

End-to-end ML regression project untuk memprediksi biaya asuransi kesehatan (`charges`) berdasarkan usia, jenis kelamin, BMI, jumlah anak, status perokok, dan wilayah tempat tinggal. Project ini menggunakan beberapa algoritma regresi dan menyiapkan model terbaik untuk disimpan menggunakan Joblib serta digunakan melalui Flask API.

**Dataset:** Medical Insurance  
**File:** `insurance.csv`  
**Task:** Regression  
**Best Model:** Gradient Boosting Regressor

**Best Model Result: MAE: 0.3323 | MSE: 0.3946 | R²: 0.6442**

> Catatan: dokumentasi ini dibuat berdasarkan notebook `Medical Insurance.ipynb`, sehingga tahapan preprocessing, encoding, model, dan deployment mengikuti implementasi yang terdapat di notebook.

---

## 📊 Dataset Overview

- **File:** `insurance.csv`
- **Ukuran awal:** 1,338 baris × 7 kolom
- **Jumlah fitur:** 6
- **Target:** `charges`
- **Numerik:** `age`, `bmi`, `children`, `charges`
- **Kategorikal:** `sex`, `smoker`, `region`
- **Missing Value:** 0
- **Data setelah filter outlier:** 1,193 baris
- **Data setelah penghapusan duplikat:** 1,192 baris

### Feature Description

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Usia peserta asuransi |
| `sex` | Categorical | Jenis kelamin peserta |
| `bmi` | Numeric | Body Mass Index |
| `children` | Numeric | Jumlah anak/dependen |
| `smoker` | Categorical | Status merokok |
| `region` | Categorical | Wilayah tempat tinggal |
| `charges` | Numeric / Target | Biaya asuransi kesehatan |

Tujuan project adalah membangun model regresi yang dapat memperkirakan
nilai `charges` berdasarkan karakteristik masing-masing peserta asuransi.

---

## 🔬 Methodology

### BAB 1: Data Loading & Understanding

Dataset dibaca menggunakan Pandas dengan `read_csv()`.

```python
import pandas as pd

insurance = pd.read_csv("insurance.csv")
insurance.head()
```

Pemeriksaan struktur data dilakukan menggunakan:

```python
insurance.info()
```

Hasil menunjukkan bahwa dataset mempunyai **1,338 baris dan 7 kolom**.
Seluruh kolom memiliki jumlah data lengkap sehingga tidak ditemukan
nilai kosong pada tahap pemeriksaan awal.

Statistik deskriptif juga diperiksa menggunakan:

```python
insurance.describe(include="all")
```

Tahap ini digunakan untuk memahami tipe data, jumlah observasi,
distribusi awal, dan karakteristik setiap variabel.

---

### BAB 2: Data Cleaning

Pemeriksaan missing value dilakukan dengan:

```python
missing_values = insurance.isnull().sum()
missing_values[missing_values > 0]
```

Hasil:

```text
Series([], dtype: int64)
```

Artinya tidak terdapat missing value pada dataset.

Notebook juga menyediakan proses pengisian missing value sebagai
langkah preprocessing:

- Variabel numerik menggunakan median.
- Variabel kategorikal menggunakan mode.

Karena dataset tidak memiliki missing value, proses tersebut tidak
mengubah jumlah maupun isi data.

Setelah itu dilakukan pemeriksaan kolom dengan jumlah missing value
yang tinggi. Tidak terdapat kolom yang masuk kategori tersebut.

---

### BAB 3: EDA - Outlier & Distribution

Exploratory Data Analysis dilakukan untuk memahami distribusi data dan
mendeteksi kemungkinan outlier.

Boxplot dibuat untuk:

```python
for feature in ['age', 'bmi', 'charges']:
    plt.figure(figsize=(10, 2))
    sns.boxplot(x=df[feature])
    plt.title(f'Box Plot of {feature}')
    plt.show()
```

Identifikasi outlier menggunakan metode Interquartile Range (IQR).

```python
Q1 = df[numeric_features].quantile(0.25)
Q3 = df[numeric_features].quantile(0.75)
IQR = Q3 - Q1
```

Batas outlier menggunakan:

```text
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Data kemudian difilter agar baris yang mengandung outlier pada fitur
numerik tidak digunakan pada dataset berikutnya.

Hasil filtering:

```text
Data sebelum filter: 1338
Data setelah filter outlier: 1193
```

Dengan demikian, sebanyak **145 baris** tidak digunakan setelah proses
filter outlier.

---

### BAB 4: Standardization & Duplicate Checking

Standardisasi dilakukan menggunakan `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df[numeric_features] = scaler.fit_transform(
    df[numeric_features]
)
```

Pada notebook, `numeric_features` mencakup kolom numerik termasuk
`charges`. Dengan demikian, standardisasi dilakukan terhadap variabel
numerik yang terdapat pada dataframe sebelum proses pemisahan fitur dan
target.

Setelah standardisasi dilakukan pemeriksaan data duplikat.

```python
duplicates = df.duplicated()

print(df[duplicates])
```

Ditemukan **1 baris duplikat**.

Baris tersebut kemudian dihapus:

```python
df = df.drop_duplicates()
```

Hasil akhir:

```text
(1192, 7)
```

Setelah proses penghapusan duplikat, dataset yang digunakan untuk
pemodelan terdiri dari **1,192 baris dan 7 kolom**.

---

### BAB 5: Encoding Data Kategorikal

Variabel kategorikal yang digunakan adalah:

```text
sex
smoker
region
```

Notebook menggunakan dua pendekatan encoding.

#### One Hot Encoding

One Hot Encoding dibuat menggunakan:

```python
df_one_hot = pd.get_dummies(
    df,
    columns=category_features
)
```

Pendekatan ini menghasilkan kolom biner untuk masing-masing kategori.

#### Label Encoding

Untuk proses modeling yang digunakan di notebook, LabelEncoder
diterapkan pada variabel kategorikal.

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df_lencoder = pd.DataFrame(df)

for col in category_features:
    df_lencoder[col] = label_encoder.fit_transform(df[col])
```

Hasil akhirnya membuat variabel kategorikal menjadi nilai numerik
sehingga dapat digunakan oleh algoritma machine learning.

Variabel yang diubah adalah:

```text
sex
smoker
region
```

---

### BAB 6: Distribution & Correlation Analysis

Distribusi beberapa variabel divisualisasikan menggunakan histogram
dan KDE.

Variabel yang dianalisis:

```text
age
bmi
children
charges
smoker
```

Contoh kode:

```python
columns_to_plot = [
    'age',
    'bmi',
    'children',
    'charges',
    'smoker'
]

for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(2, 3, i)
    sns.histplot(
        df_lencoder[column],
        kde=True,
        bins=30
    )
```

Analisis korelasi kemudian dilakukan menggunakan:

```python
correlation_matrix = df_lencoder.corr()

sns.heatmap(
    correlation_matrix,
    annot=False,
    cmap='coolwarm',
    vmin=-1,
    vmax=1
)
```

Korelasi terhadap target `charges` menghasilkan:

| Variable | Correlation |
|---|---:|
| `charges` | 1.000000 |
| `smoker` | 0.602377 |
| `age` | 0.436059 |
| `region` | 0.090858 |
| `children` | 0.082308 |
| `bmi` | 0.060969 |
| `sex` | 0.021548 |

Berdasarkan hasil tersebut, variabel `smoker` memiliki hubungan paling
kuat dengan `charges`, diikuti oleh `age`.

---

### BAB 7: Data Preparation - Splitting

Fitur dan target dipisahkan menggunakan:

```python
X = df_lencoder.drop(columns='charges')
y = df_lencoder['charges']
```

Fitur yang digunakan:

```text
age
bmi
children
sex
smoker
region
```

Target:

```text
charges
```

Dataset kemudian dibagi menjadi data training dan testing:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Hasil pembagian:

```text
Jumlah data      : 1192
Jumlah data latih: 953
Jumlah data test : 239
```

Dengan demikian, sekitar 80% data digunakan untuk melatih model dan
20% digunakan untuk menguji performa model.

---

### BAB 8: Modeling

Tiga algoritma regresi digunakan untuk membandingkan performa model:

1. Least Angle Regression (LARS)
2. Linear Regression
3. Gradient Boosting Regressor

#### Least Angle Regression

```python
from sklearn import linear_model

lars = linear_model.Lars(
    n_nonzero_coefs=1
)

lars.fit(X_train, y_train)
```

LARS digunakan sebagai salah satu model pembanding dalam proses
pemodelan regresi.

#### Linear Regression

```python
from sklearn.linear_model import LinearRegression

LR = LinearRegression()

LR.fit(X_train, y_train)
```

Linear Regression digunakan sebagai model regresi linear dasar untuk
membandingkan hasil prediksi dengan model lainnya.

#### Gradient Boosting Regressor

```python
from sklearn.ensemble import GradientBoostingRegressor

GBR = GradientBoostingRegressor(
    random_state=184
)

GBR.fit(X_train, y_train)
```

Gradient Boosting membangun model secara bertahap dengan menggunakan
model sebelumnya untuk memperbaiki kesalahan prediksi.

---

## 📈 BAB 9: Validation & Evaluation

Evaluasi dilakukan menggunakan tiga metrik:

### Mean Absolute Error (MAE)

MAE menunjukkan rata-rata selisih absolut antara nilai aktual dan
nilai prediksi.

```text
MAE = mean(|y - ŷ|)
```

Semakin kecil nilai MAE, semakin kecil rata-rata kesalahan prediksi.

### Mean Squared Error (MSE)

MSE menghitung rata-rata kuadrat kesalahan prediksi.

```text
MSE = mean((y - ŷ)²)
```

Nilai MSE yang lebih kecil menunjukkan model memiliki kesalahan
prediksi yang lebih rendah.

### R² Score

R² menunjukkan seberapa besar variasi target dapat dijelaskan oleh
model.

Nilai yang semakin mendekati 1 menunjukkan performa model yang semakin
baik.

---

## 📊 Model Comparison

Berdasarkan proses evaluasi pada notebook, diperoleh hasil berikut:

| Model | MAE | MSE | R² |
|---|---:|---:|---:|
| LARS | 0.7391 | 0.9353 | 0.1567 |
| Linear Regression | 0.3798 | 0.4420 | 0.6014 |
| **GradientBoostingRegressor** | **0.3323** | **0.3946** | **0.6442** |

Berdasarkan hasil tersebut, **GradientBoostingRegressor** menghasilkan
performa terbaik karena memiliki nilai MAE dan MSE paling rendah serta
nilai R² paling tinggi dibandingkan dua model lainnya.

Model tersebut kemudian dipilih sebagai model utama untuk proses
penyimpanan dan deployment.

---

## 💾 BAB 10: Saving Model

Model Gradient Boosting disimpan menggunakan Joblib.

```python
import joblib

joblib.dump(
    GBR,
    'gbr_model_medical_insurance.joblib'
)
```

File model:

```text
gbr_model_medical_insurance.joblib
```

Model kemudian dimuat kembali menggunakan:

```python
joblib_model = joblib.load(
    'gbr_model_medical_insurance.joblib'
)
```

Penyimpanan model memungkinkan model yang telah dilatih digunakan
kembali tanpa melakukan proses training dari awal.

---

## 🚀 BAB 11: Deployment

Notebook menggunakan Flask sebagai framework untuk menyediakan REST API.

```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

joblib_model = joblib.load(
    'gbr_model_medical_insurance.joblib'
)

@app.route('/predict', methods=['POST'])
def predict():

    data = request.json['data']

    prediction = joblib_model.predict(data)

    return jsonify({
        'prediction': prediction.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
```

Endpoint yang digunakan:

```text
POST /predict
```

Server Flask dijalankan pada:

```text
http://127.0.0.1:5000
```

API menerima data melalui JSON pada bagian:

```text
data
```

Kemudian data tersebut diteruskan ke model untuk menghasilkan nilai
prediksi.

---

## 🧪 Prediction Testing

Notebook juga melakukan pengujian prediksi secara langsung menggunakan:

```python
data = [[19, 0, 27.9, 0, 1, 0]]

hasil = joblib_model.predict(data)

print(hasil)
```

Hasil prediksi pada notebook:

```text
[2.77634046]
```

Input tersebut menggunakan enam fitur dengan urutan:

```text
age
bmi
children
sex
smoker
region
```

Karena model pada notebook dilatih menggunakan hasil Label Encoding,
input yang dikirim ke model harus menggunakan bentuk numerik yang sesuai
dengan encoding tersebut.

---

## 🔄 Machine Learning Workflow

Alur lengkap project:

```text
DATASET
   ↓
DATA LOADING
   ↓
DATA UNDERSTANDING
   ↓
MISSING VALUE CHECK
   ↓
OUTLIER DETECTION
   ↓
OUTLIER FILTERING
   ↓
STANDARDIZATION
   ↓
DUPLICATE CHECK
   ↓
DUPLICATE REMOVAL
   ↓
LABEL ENCODING
   ↓
EDA & CORRELATION
   ↓
TRAIN / TEST SPLIT
   ↓
MODEL TRAINING
   ↓
MODEL EVALUATION
   ↓
MODEL SELECTION
   ↓
MODEL SAVING
   ↓
FLASK API
   ↓
PREDICTION
```

---

## 📁 Project Structure

```text
medical-insurance/
│
├── insurance.csv
├── Medical Insurance.ipynb
├── gbr_model_medical_insurance.joblib
├── app.py
└── README.md
```

---

## 📦 Requirements

Library yang digunakan dalam notebook:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
flask
```

Instalasi:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib flask
```

---

## 🎯 Conclusion

Project ini menggunakan dataset Medical Insurance untuk membangun model
regresi yang dapat digunakan untuk memprediksi biaya asuransi kesehatan.

Dataset awal terdiri dari **1,338 baris dan 7 kolom**. Setelah proses
filtering outlier dan penghapusan data duplikat, diperoleh **1,192 baris
data** yang digunakan dalam proses modeling.

Tiga algoritma dibandingkan, yaitu LARS, Linear Regression, dan
Gradient Boosting Regressor. Berdasarkan hasil evaluasi, Gradient
Boosting Regressor memberikan performa terbaik dengan **MAE 0.3323,
MSE 0.3946, dan R² 0.6442**.

Model terbaik kemudian disimpan menggunakan Joblib dengan nama
`gbr_model_medical_insurance.joblib` dan disiapkan untuk digunakan
melalui Flask REST API.

Secara keseluruhan, project mencakup tahapan mulai dari data loading,
cleaning, EDA, outlier filtering, standardisasi, encoding,
correlation analysis, pembagian data, modeling, evaluasi, penyimpanan
model, hingga deployment menggunakan Flask.
