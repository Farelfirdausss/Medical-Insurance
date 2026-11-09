import streamlit as st
import pandas as pd
import joblib

model = joblib.load("gbr_model_medical_insurance.joblib")

st.set_page_config(page_title="Medical Insurance Prediction", page_icon="🏥")
st.title("🏥 Prediksi Biaya Asuransi Kesehatan")
st.write("Masukkan data pasien untuk memprediksi biaya asuransi (charges)")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Umur (age)", 18, 100, 30)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    children = st.number_input("Jumlah Anak (children)", 0, 5, 0)

with col2:
    sex = st.selectbox("Jenis Kelamin (sex)", ["male", "female"])
    smoker = st.selectbox("Perokok? (smoker)", ["no", "yes"])
    region = st.selectbox("Wilayah (region)", ["southwest", "southeast", "northwest", "northeast"])

if st.button("Prediksi Biaya", use_container_width=True):
    # 1. Ngintip dulu, model lu dilatih pake kolom apa aja?
    fitur_model = list(model.feature_names_in_)

    # 2. Bikin semua kolom jadi 0 dulu
    data_dict = {f: [0] for f in fitur_model}

    # 3. Isi yang angka
    if 'age' in data_dict:
        data_dict['age'] = [age]
    if 'bmi' in data_dict:
        data_dict['bmi'] = [bmi]
    if 'children' in data_dict:
        data_dict['children'] = [children]

    # 4. Isi yang kategori, diubah jadi 1 kalo cocok
    for f in fitur_model:
        if f == f"sex_{sex}":
            data_dict[f] = [1]
        if f == f"smoker_{smoker}":
            data_dict[f] = [1]
        if f == f"region_{region}":
            data_dict[f] = [1]

    data = pd.DataFrame(data_dict)[fitur_model]
    prediksi = model.predict(data)[0]

    st.success(f"💰 Estimasi Biaya Asuransi: **${prediksi:,.2f}**")
    st.write(f"Atau sekitar Rp {prediksi * 16000:,.0f}")