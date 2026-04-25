import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="Concrete Strength Predictor",
    page_icon="🏗️",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load("model/xgboost_concrete_model.pkl")
    features = joblib.load("model/feature_names.pkl")
    return model, features

model, feature_names = load_model()

# ====================== SIDEBAR ======================
with st.sidebar:
    st.title("🏗️ Concrete AI")
    st.caption("28-day strength predictor")
    st.markdown("---")
    st.write("**Model Metrics**")
    st.success("R² = 0.9445")
    st.info("MAE = 2.57 MPa")

# ====================== MAIN ======================
st.title("🏗️ Concrete Compressive Strength Predictor")
st.markdown("**Predict concrete strength from mix design using XGBoost**")

tab1, tab2 = st.tabs(["🔮 Make Prediction", "ℹ️ About the Model"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        cement = st.number_input("Cement (kg/m³)", 100.0, 600.0, 300.0, step=5.0)
        slag = st.number_input("Slag (kg/m³)", 0.0, 400.0, 0.0, step=5.0)
        flyash = st.number_input("Fly Ash (kg/m³)", 0.0, 400.0, 0.0, step=5.0)
        water = st.number_input("Water (kg/m³)", 100.0, 300.0, 180.0, step=5.0)
        superplasticizer = st.number_input("Superplasticizer (kg/m³)", 0.0, 40.0, 5.0, step=0.5)

    with col2:
        coarseaggregate = st.number_input("Coarse Aggregate (kg/m³)", 800.0, 1300.0, 1000.0, step=10.0)
        fineaggregate = st.number_input("Fine Aggregate (kg/m³)", 600.0, 1000.0, 800.0, step=10.0)
        age = st.number_input("Age (days)", 1, 365, 28, step=1)

    if st.button("🚀 Predict Compressive Strength", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            'cement': cement, 'slag': slag, 'flyash': flyash,
            'water': water, 'superplasticizer': superplasticizer,
            'coarseaggregate': coarseaggregate, 'fineaggregate': fineaggregate,
            'age': age
        }])
        
        input_df['water_cement_ratio'] = input_df['water'] / input_df['cement']
        input_df['total_binder'] = input_df['cement'] + input_df['slag'] + input_df['flyash']
        
        input_df = input_df[feature_names]
        
        pred = model.predict(input_df)[0]
        
        st.success(f"**Predicted Strength: {pred:.2f} MPa**", icon="🔥")
        
        if pred > 55:
            st.balloons()
            st.success("High-Strength Concrete! Excellent mix.")
        elif pred > 35:
            st.info("Good structural concrete")
        else:
            st.warning("Consider increasing cement or reducing water")

with tab2:
    st.subheader("Model Information")
    st.write("- **Algorithm**: XGBoost Regressor")
    st.write("- **Training Data**: 1005 samples (UCI Concrete Dataset)")
    st.write("- **Performance**: R² = 0.9445 | MAE = 2.57 MPa")
    st.write("- **Key Features**: Age, Water-Cement Ratio, Total Binder")

st.markdown("---")
st.markdown("Made with ❤️ by Uche • [GitHub Repo](https://github.com/yourusername/concrete-strength-predictor)")
