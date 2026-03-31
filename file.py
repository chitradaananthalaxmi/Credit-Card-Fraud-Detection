import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------
# PAGE CONFIG
# ---------------------------------------------
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------
# FUTURISTIC CSS
# ---------------------------------------------
page_bg = """
<style>
body {
    background: linear-gradient(135deg, #0f0f0f 0%, #1b1b1b 50%, #0d0d0d 100%);
    color: #e0e0e0;
}

.sidebar .sidebar-content {
    background: linear-gradient(180deg, #111111, #0a0a0a);
}

.block-container {
    padding-top: 2rem;
}

h1, h2, h3 {
    color: #00eaff;
    text-shadow: 0px 0px 10px #00eaff;
}

.stButton>button {
    background: linear-gradient(90deg, #00eaff, #0066ff);
    color: white;
    border-radius: 8px;
    transition: 0.3s;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #0099ff, #0044ff);
    transform: scale(1.03);
}

.dataframe {
    color: white !important;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------------------------------------------
# TITLE + HEADER
# ---------------------------------------------
st.markdown("<h1 style='text-align:center;'>🛡️ FraudShield AI</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>Advanced Credit Card Fraud Detection System</h3>", unsafe_allow_html=True)

st.write("---")

# ---------------------------------------------
# LOAD MODELS
# ---------------------------------------------
try:
    LR = joblib.load("LR_model.pkl")
    DT = joblib.load("DT_model.pkl")
    RF = joblib.load("RF_model.pkl")
except:
    LR = DT = RF = None
    st.warning("Models not found. Upload LR_model.pkl, DT_model.pkl, RF_model.pkl to repository.")

# ---------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------
menu = st.sidebar.radio("🔍 Navigate", ["Dashboard", "Predict"])

# ---------------------------------------------
# DASHBOARD SECTION
# ---------------------------------------------
if menu == "Dashboard":
    st.subheader("📊 Model Performance Comparison")
    st.write("Results based on your ML model training:")

    accuracy_data = {
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
        "Accuracy": [0.999, 0.99925, 0.99925],
    }

    df_acc = pd.DataFrame(accuracy_data)

    st.bar_chart(df_acc, x="Model", y="Accuracy", color="#00eaff")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Best Model", "Random Forest")
    with col2:
        st.metric("Accuracy", "99.9%")

    st.write("---")
    st.write("### 🔬 Classification Report (Random Forest)")
    st.code("""
Class 0 → Precision 1.00 | Recall 1.00 | F1 1.00  
Class 1 → Precision 0.83 | Recall 0.71 | F1 0.77
    """)

# ---------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------
if menu == "Predict":
    st.subheader("🔮 Predict Credit Card Fraud")

    st.info("Enter the feature values of a transaction below.")

    columns = [
        "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
        "V10","V11","V12","V13","V14","V15","V16","V17","V18",
        "V19","V20","V21","V22","V23","V24","V25","V26","V27",
        "V28","Amount"
    ]

    input_values = []

    col1, col2, col3 = st.columns(3)

    for i, col in enumerate(columns):
        if i % 3 == 0:
            with col1:
                v = st.number_input(col, value=0.0)
                input_values.append(v)
        elif i % 3 == 1:
            with col2:
                v = st.number_input(col, value=0.0)
                input_values.append(v)
        else:
            with col3:
                v = st.number_input(col, value=0.0)
                input_values.append(v)

    if st.button("🚀 Predict"):
        x = np.array(input_values).reshape(1, -1)

        if RF:
            pred = RF.predict(x)[0]
            if pred == 0:
                st.success("✅ Legitimate Transaction")
            else:
                st.error("❌ Fraudulent Transaction Detected!")
        else:
            st.error("Model missing! Upload trained model files.")

# ---------------------------------------------
# FOOTER
# ---------------------------------------------
st.write("---")
st.markdown(
    "<p style='text-align:center; color: #888;'>Created with ⚡ Streamlit | FraudShield AI</p>",
    unsafe_allow_html=True,
)
