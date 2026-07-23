import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.datasets import load_breast_cancer

# --- Setup Page ---
st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {width: 100%; border-radius: 5px; height: 3em; background-color: #3498db; color: white; font-weight: bold;}
    .stButton>button:hover {background-color: #2980b9;}
    .malignant-card {background-color: #ffebee; padding: 20px; border-radius: 10px; border-left: 5px solid #e74c3c; color: #c0392b;}
    .benign-card {background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #2ecc71; color: #27ae60;}
    </style>
""", unsafe_allow_html=True)

# --- Load Artifacts ---
@st.cache_resource
def load_models():
    model = joblib.load(os.path.join('models', 'model.pkl'))
    scaler = joblib.load(os.path.join('models', 'scaler.pkl'))
    feature_names = joblib.load(os.path.join('models', 'feature_names.pkl'))
    importances = joblib.load(os.path.join('models', 'feature_importances.pkl'))
    
    # Get mean values for defaults
    dataset = load_breast_cancer()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['label'] = dataset.target
    
    means = df.mean().to_dict()
    benign_means = df[df['label'] == 1].mean().to_dict()
    malignant_means = df[df['label'] == 0].mean().to_dict()
    
    return model, scaler, feature_names, importances, means, benign_means, malignant_means

model, scaler, feature_names, importances, means, benign_means, malignant_means = load_models()

# Sort features by importance
top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
top_feature_names = [f[0] for f in top_features]

# --- UI Header ---
st.title("🧬 Breast Cancer Prediction Engine")
st.markdown("Enter the tumor characteristics in the sidebar to predict whether the tumor is **Benign** or **Malignant** using a Logistic Regression model (98.25% Accuracy).")

# --- Sidebar Inputs ---
st.sidebar.header("🔬 Input Features")
st.sidebar.markdown("For simplicity, you only need to adjust the **Top 5 most important features**. The remaining 25 features are set to their mean values automatically.")

# Quick fill buttons
col1, col2 = st.sidebar.columns(2)
if col1.button("Sample Benign"):
    st.session_state.preset = 'benign'
if col2.button("Sample Malignant"):
    st.session_state.preset = 'malignant'

# Initialize session state for inputs
if 'preset' not in st.session_state:
    st.session_state.preset = 'mean'

input_data = {}

for feature in feature_names:
    if feature in top_feature_names:
        # Determine default value based on preset
        if st.session_state.preset == 'benign':
            default_val = float(benign_means[feature])
        elif st.session_state.preset == 'malignant':
            default_val = float(malignant_means[feature])
        else:
            default_val = float(means[feature])
            
        input_data[feature] = st.sidebar.slider(
            label=feature.capitalize(),
            min_value=float(means[feature] * 0.1),
            max_value=float(means[feature] * 3.0),
            value=default_val,
            format="%.4f"
        )
    else:
        # Keep background features at mean
        if st.session_state.preset == 'benign':
            input_data[feature] = benign_means[feature]
        elif st.session_state.preset == 'malignant':
            input_data[feature] = malignant_means[feature]
        else:
            input_data[feature] = means[feature]

# --- Prediction Logic ---
def predict(input_dict):
    input_df = pd.DataFrame([input_dict], columns=feature_names)
    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0]
    return prediction, probability

prediction, probability = predict(input_data)

# --- Main Layout ---
tab1, tab2 = st.tabs(["🔮 Prediction Result", "📊 Model Insights"])

with tab1:
    st.subheader("Diagnostic Prediction")
    
    # 0 is Malignant, 1 is Benign
    if prediction == 0:
        st.markdown(f"""
            <div class="malignant-card">
                <h2>⚠️ Malignant Tumor Detected</h2>
                <p>The model predicts that the tumor is cancerous with <strong>{probability[0]*100:.2f}% confidence</strong>. Immediate medical review is recommended.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="benign-card">
                <h2>✅ Benign Tumor Detected</h2>
                <p>The model predicts that the tumor is non-cancerous with <strong>{probability[1]*100:.2f}% confidence</strong>. Routine check-ups are advised.</p>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.subheader("Confidence Breakdown")
    colA, colB = st.columns(2)
    with colA:
        st.metric(label="Probability of Malignant", value=f"{probability[0]*100:.2f}%")
        st.progress(float(probability[0]))
    with colB:
        st.metric(label="Probability of Benign", value=f"{probability[1]*100:.2f}%")
        st.progress(float(probability[1]))

with tab2:
    st.subheader("Top 5 Feature Importances (Logistic Regression)")
    st.markdown("These are the most heavily weighted features the model uses to make its decision.")
    
    import_df = pd.DataFrame(top_features, columns=['Feature', 'Importance (Absolute Weight)'])
    st.bar_chart(import_df.set_index('Feature'))
    
    st.markdown("""
    ### About the Model
    - **Algorithm:** Logistic Regression
    - **Preprocessing:** StandardScaler
    - **Test Accuracy:** 98.25%
    - **Precision/Recall:** 98.61%
    """)
