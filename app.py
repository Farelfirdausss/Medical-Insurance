import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

st.title("Prediksi Biaya Asuransi Kesehatan")

# Baca data & latih langsung di sini, gak pake .joblib
df = pd.read_csv("insurance.csv")
df = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)

X = df.drop("charges", axis=1)
y = df["charges"]
model = GradientBoostingRegressor(random_state=42)
model.fit(X, y)

age = st.slider("Umur", 18, 64, 30)
bmi = st.number_input("BMI", 15.0, 50.0, 25.0)
children = st.slider("Jumlah Anak", 0, 5, 0)
sex = st.selectbox("Jenis Kelamin", ["male", "female"])
smoker = st.selectbox("Perokok?", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

if st.button("Prediksi Biaya"):
    data = {
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex_male": [1 if sex == "male" else 0],
        "smoker_yes": [1 if smoker == "yes" else 0],
        "region_northwest": [1 if region == "northwest" else 0],
        "region_southeast": [1 if region == "southeast" else 0],
        "region_southwest": [1 if region == "southwest" else 0],
    }
    # Samakan kolom dengan yang dipakai training
    input_df = pd.DataFrame(data)
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[X.columns]

    pred = model.predict(input_df)[0]
    st.success(f"💰 Estimasi Biaya Asuransi: ${pred:,.2f}")