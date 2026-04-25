import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page config
st.set_page_config(page_title="Concrete Strength Predictor", page_icon="🏗️", layout="centered")

st.title("🏗️ Concrete Compressive Strength Predictor")
st.markdown("""
### Enter the concrete mix ingredients and age to predict 28-day strength.
""")

# Load model and features
@st.cache_resource
def load_model():
    model = joblib.load("model/xgboost_concrete_model.pkl")
    features = joblib.load("model/feature_names.pkl")
    return model, features

model, feature_names = load_model()

# Input fields
col1, col2 = st.columns(2)

with col1:
    cement = st.number_input("Cement (kg/m³)", min_value=100.0, max_value=600.0, value=300.0, step=10.0)
    slag = st.number_input("Blast Furnace Slag (kg/m³)", min_value=0.0, max_value=400.0, value=0.0, step=10.0)
    flyash = st.number_input("Fly Ash (kg/m³)", min_value=0.0, max_value=400.0, value=0.0, step=10.0)
    water = st.number_input("Water (kg/m³)", min_value=100.0, max_value=300.0, value=180.0, step=5.0)
    superplasticizer = st.number_input("Superplasticizer (kg/m³)", min_value=0.0, max_value=40.0, value=5.0, step=1.0)

with col2:
    coarseaggregate = st.number_input("Coarse Aggregate (kg/m³)", min_value=800.0, max_value=1200.0, value=1000.0, step=10.0)
    fineaggregate = st.number_input("Fine Aggregate (kg/m³)", min_value=600.0, max_value=1000.0, value=800.0, step=10.0)
    age = st.number_input("Age (days)", min_value=1, max_value=365, value=28, step=1)

st.subheader("Mix Design Inputs")

# Predict button
if st.button("🚀 Predict Compressive Strength", type="primary"):
    # Create input dataframe
    input_data = pd.DataFrame({
        'cement': [cement],
        'slag': [slag],
        'flyash': [flyash],
        'water': [water],
        'superplasticizer': [superplasticizer],
        'coarseaggregate': [coarseaggregate],
        'fineaggregate': [fineaggregate],
        'age': [age]
    })
    
    # Feature engineering (same as training)
    input_data['water_cement_ratio'] = input_data['water'] / input_data['cement']
    input_data['total_binder'] = input_data['cement'] + input_data['slag'] + input_data['flyash']
    
    # Reorder columns to match training
    input_data = input_data[feature_names]
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    st.success(f"**Predicted Compressive Strength: {prediction:.2f} MPa**")
    
    # Simple gauge / interpretation
    if prediction >= 50:
        st.balloons()
        st.success("Excellent High-Strength Concrete! 🏆")
    elif prediction >= 30:
        st.info("Good Normal Concrete")
    else:
        st.warning("Lower Strength Concrete")

# Add footer
st.markdown("---")
st.markdown("Built with ❤️ using XGBoost + Streamlit")