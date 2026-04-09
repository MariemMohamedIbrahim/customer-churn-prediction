import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# إعداد الصفحة
st.set_page_config(page_title="Churn Dashboard", layout="wide")

# تحميل الملفات
model = joblib.load("light_churn_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("one_hot_encoder.pkl")
model_features = joblib.load("final_model_features.pkl")

# ==== CSS تنسيق ====
st.markdown("""
    <style>
    h1, h2, h3 {
        color: #1f2937;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #1f77b4;  
        color: white;
        border-radius: 8px;
        padding: 0.5em 2em;
    }
    </style>
""", unsafe_allow_html=True)


# ==== الصورة الصغيرة في الـ Sidebar ====
with st.sidebar:
    st.image("WhatsApp Image 2025-04-21 at 1.10.13 AM.jpeg", use_column_width=True)
    st.markdown("## 🧾 User Inputs")

    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", 18, 100, value=30)
    num_dependents = st.number_input("Number of Dependents", min_value=0, value=2)
    estimated_salary = st.number_input("Estimated Salary", value=50000)
    calls_made = st.number_input("Calls Made", value=50)
    sms_sent = st.number_input("SMS Sent", value=30)
    data_used = st.number_input("Data Used (MB)", value=1024.0)
    monthly_bill = st.number_input("Monthly Bill (USD)", value=40.0)
    registration_year = st.selectbox("Registration Year", list(range(2018, 2026)))
    registration_month = st.selectbox("Registration Month", list(range(1, 13)))
    registration_day = st.selectbox("Registration Day", list(range(1, 32)))
    day_of_week = st.selectbox("Day of Week (1=Mon, 7=Sun)", list(range(1, 8)))
    telecom_partner = st.selectbox("Telecom Partner", ['Airtel', 'Jio', 'Vodafone', 'BSNL'])
    state = st.selectbox("State", [
        'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat',
        'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala',
        'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
        'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
        'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
    ])
    city = st.selectbox("City", ['Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai'])

# ==== الصورة الكبيرة ====
st.image("telecom.jpg", use_column_width=True)

st.markdown("""
    <div style='text-align: center; padding-top: 1rem;'>
        <h1 style='font-size: 40px;'>📊 Customer Churn Prediction</h1>
    </div>
""", unsafe_allow_html=True)

def preprocess_input():
    gender_encoded = 1 if gender == "Male" else 0
    calls_per_sms = calls_made / sms_sent if sms_sent != 0 else 0
    data_per_dependent = data_used / num_dependents if num_dependents != 0 else data_used
    data_per_call = data_used / calls_made if calls_made != 0 else data_used

    cat_df = pd.DataFrame([[telecom_partner, state, city]], columns=['telecom_partner', 'state', 'city'])
    encoded_array = encoder.transform(cat_df)
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out())

    numeric_data = {
        'customer_id': 0,
        'gender': gender_encoded,
        'age': age,
        'num_dependents': num_dependents,
        'estimated_salary': estimated_salary,
        'calls_made': calls_made,
        'sms_sent': sms_sent,
        'data_used': data_used,
        'registration_year': registration_year,
        'registration_month': registration_month,
        'registration_day': registration_day,
        'calls_per_sms': calls_per_sms,
        'data_per_dependent': data_per_dependent,
        'data_per_call': data_per_call,
        'monthly_bill': monthly_bill,
        'day_of_week': day_of_week
    }

    df = pd.DataFrame([numeric_data])
    df = pd.concat([df, encoded_df], axis=1)

    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    df = df[model_features]

    return scaler.transform(df), numeric_data.keys()

# ==== المحتوى ====
col_chart, col_pred = st.columns([2, 1], gap="large")

with col_chart:
    st.markdown("<div style='text-align: center;'><h2>💡 Top Feature Importance</h2></div>", unsafe_allow_html=True)
    try:
        importances = model.feature_importances_
        input_data_scaled, user_features = preprocess_input()

        feature_df = pd.DataFrame({
            "feature": model_features,
            "importance": importances
        })
        filtered = feature_df[feature_df["feature"].isin(user_features) & (feature_df["feature"] != "customer_id")]
        top_feats = filtered.sort_values("importance", ascending=False)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(top_feats["feature"][::-1], top_feats["importance"][::-1], color="#1f77b4")
        ax.set_title("Top Feature Importances")
        ax.set_xlabel("Importance")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Feature importance not available: {e}")

with col_pred:
    st.markdown("### 🔍 Prediction")
    if st.button("✅ Predict", use_container_width=True):
        try:
            X, _ = preprocess_input()
            pred = model.predict(X)[0]
            prob = model.predict_proba(X)[0][1]

            st.markdown("---")
            if pred == 1:
                st.error(f"✅ Predicted: Churn\n\n📉 Probability: {prob:.2%}")
            else:
                st.success(f"✅ Predicted: Retain\n\n📈 Probability: {1 - prob:.2%}")
        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")

#streamlit run churn_prediction.py

