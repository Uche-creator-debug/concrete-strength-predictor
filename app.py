import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Concrete Strength Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main {padding: 2rem;}
    .stButton>button {width: 100%; height: 3rem; font-size: 1.1rem;}
    .prediction {font-size: 2.5rem; font-weight: bold; text-align: center;}
</style>
""", unsafe_allow_html=True)

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_model():
    model = joblib.load("model/xgboost_concrete_model.pkl")
    features = joblib.load("model/feature_names.pkl")
    return model, features

model, feature_names = load_model()

# ====================== SIDEBAR ======================
with st.sidebar:
    st.title("🏗️ Concrete AI")
    st.markdown("### Predict 28-day strength")
    
    st.markdown("---")
    st.markdown("**Model Info**")
    st.write("• XGBoost Regressor")
    st.write("• R² Score: **0.9445**")
    st.write("• MAE: **2.57 MPa**")
    
    if st.button("Reset All Inputs"):
        st.rerun()

# ====================== MAIN APP ======================
st.title("🏗️ Concrete Compressive Strength Predictor")
st.markdown("**Professional mix design predictor powered by Machine Learning**")

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Performance", "ℹ️ About"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Mix Ingredients")
        cement = st.number_input("Cement (kg/m³)", 100.0, 600.0, 300.0, step=10.0)
        slag = st.number_input("Blast Furnace Slag (kg/m³)", 0.0, 400.0, 0.0, step=10.0)
        flyash = st.number_input("Fly Ash (kg/m³)", 0.0, 400.0, 0.0, step=10.0)
        water = st.number_input("Water (kg/m³)", 100.0, 300.0, 180.0, step=5.0)
        superplasticizer = st.number_input("Superplasticizer (kg/m³)", 0.0, 40.0, 5.0, step=1.0)
    
    with col2:
        st.subheader("Aggregates & Age")
        coarseaggregate = st.number_input("Coarse Aggregate (kg/m³)", 800.0, 1200.0, 1000.0, step=10.0)
        fineaggregate = st.number_input("Fine Aggregate (kg/m³)", 600.0, 1000.0, 800.0, step=10.0)
        age = st.number_input("Age (days)", 1, 365, 28, step=1)
        
        st.markdown("---")
        st.subheader("Quick Presets")
        preset = st.selectbox("Load sample mix", 
            ["Custom", "Standard M25", "High Strength M60", "Low Cement"])
        
        if preset == "Standard M25":
            cement, water, coarseaggregate, fineaggregate = 300, 180, 1050, 780
        elif preset == "High Strength M60":
            cement, slag, water, superplasticizer = 450, 50, 160, 12
        elif preset == "Low Cement":
            cement, flyash, water = 220, 100, 190

    if st.button("🚀 Predict Strength", type="primary", use_container_width=True):
        input_data = pd.DataFrame({
            'cement': [cement], 'slag': [slag], 'flyash': [flyash],
            'water': [water], 'superplasticizer': [superplasticizer],
            'coarseaggregate': [coarseaggregate], 'fineaggregate': [fineaggregate],
            'age': [age]
        })
        
        input_data['water_cement_ratio'] = input_data['water'] / input_data['cement']
        input_data['total_binder'] = input_data['cement'] + input_data['slag'] + input_data['flyash']
        
        input_data = input_data[feature_names]
        
        prediction = model.predict(input_data)[0]
        
        st.success(f"**Predicted Compressive Strength: {prediction:.2f} MPa**", icon="🏆")
        
        if prediction >= 55:
            st.balloons()
            st.success("**High Performance Concrete** — Excellent!")
        elif prediction >= 35:
            st.info("**Normal Structural Concrete** — Good")
        else:
            st.warning("**Lower Strength** — Consider mix adjustment")

with tab2:
    st.subheader("Model Performance Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", "0.9445", "Excellent")
    col2.metric("MAE", "2.57 MPa", "Very Accurate")
    col3.metric("RMSE", "4.07 MPa")
    
    st.markdown("### Feature Importance")
    # You can add a static image or generate plot
    st.info("Top Features: **Age**, **Total Binder**, **Water-Cement Ratio**")
    
    st.markdown("---")
    st.write("This model was trained on 1005 samples from the UCI Concrete dataset.")

with tab3:
    st.subheader("How it Works")
    st.write("""
    This app uses **XGBoost** — one of the most powerful gradient boosting algorithms — 
    trained on real experimental concrete data.
    
    Key engineered features:
    - Water-Cement Ratio
    - Total Binder (Cement + Slag + Fly Ash)
    """)
    
    st.markdown("### Tips for Accurate Prediction")
    st.info("• Age has very strong effect\n• Lower water-cement ratio = higher strength\n• Superplasticizer helps reduce water")

# Footer
st.markdown("---")
st.markdown(
    "Built by **Uche** • [GitHub](https://github.com/yourusername/concrete-strength-predictor) "
    "• Powered by XGBoost + Streamlit"
)