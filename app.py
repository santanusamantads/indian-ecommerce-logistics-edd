import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

# Page Configuration
st.set_page_config(page_title="Indian E-Commerce EDD Predictor", page_icon="📦", layout="wide")

st.title("📦 Indian E-Commerce Logistics: EDD & SLA Risk Predictor")
st.markdown("**M.Sc. Data Science & AI Project — RKMVERI** | Interactive Inference Dashboard")

# Load Data and Train Models (Cached for performance)
@st.cache_resource
def load_and_train():
    df = pd.read_csv('india_multi_agency_edd_dataset.csv')
    
    features = [
        'Delivery_Partner', 'Logistics_Zone', 'Distance_km', 'Shipment_Mode',
        'Weight_kg', 'Is_COD', 'Shipping_Cost_INR', 'Weather_Condition',
        'Festival_Peak', 'Pickup_Delay_Hours', 'SLA_Promised_Days'
    ]
    
    X = pd.get_dummies(df[features], drop_first=True)
    y_reg = df['Transit_Time_Hours']
    y_clf = df['Is_Delayed_SLA']
    
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train Random Forest for Regression & Logistic Regression for Classification
    rf_reg = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train, y_reg_train)
    
    log_clf = LogisticRegression(max_iter=1000, random_state=42)
    log_clf.fit(X_train_scaled, y_clf_train)
    
    return rf_reg, log_clf, scaler, X.columns, df

rf_reg, log_clf, scaler, model_columns, df = load_and_train()

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Shipment Parameters")

partner = st.sidebar.selectbox("Delivery Partner", df['Delivery_Partner'].unique())
zone = st.sidebar.selectbox("Logistics Zone", df['Logistics_Zone'].unique())
mode = st.sidebar.selectbox("Shipment Mode", df['Shipment_Mode'].unique())
weather = st.sidebar.selectbox("Weather Condition", df['Weather_Condition'].unique())

distance = st.sidebar.slider("Distance (km)", min_value=10, max_value=2500, value=750, step=10)
weight = st.sidebar.slider("Package Weight (kg)", min_value=0.1, max_value=15.0, value=1.5, step=0.1)
cost = st.sidebar.slider("Shipping Cost (INR)", min_value=50.0, max_value=500.0, value=150.0, step=5.0)
pickup_delay = st.sidebar.slider("Warehouse Pickup Delay (Hours)", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
sla_days = st.sidebar.slider("Promised SLA (Days)", min_value=1, max_value=10, value=4, step=1)

is_cod = st.sidebar.radio("Payment Method", ['Prepaid (0)', 'Cash on Delivery (1)'])
is_cod_val = 1 if 'Cash on Delivery' in is_cod else 0

festival = st.sidebar.radio("Festival Peak Surge?", ['No (0)', 'Yes (1)'])
festival_val = 1 if 'Yes' in festival else 0

# --- Build Input DataFrame for Prediction ---
input_data = pd.DataFrame({
    'Delivery_Partner': [partner],
    'Logistics_Zone': [zone],
    'Distance_km': [distance],
    'Shipment_Mode': [mode],
    'Weight_kg': [weight],
    'Is_COD': [is_cod_val],
    'Shipping_Cost_INR': [cost],
    'Weather_Condition': [weather],
    'Festival_Peak': [festival_val],
    'Pickup_Delay_Hours': [pickup_delay],
    'SLA_Promised_Days': [sla_days]
})

# Align columns with training dummy variables
input_encoded = pd.get_dummies(input_data)
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

# --- Make Predictions ---
predicted_hours = rf_reg.predict(input_encoded)[0]
predicted_days = round(predicted_hours / 24, 1)

input_scaled = scaler.transform(input_encoded)
sla_prob = log_clf.predict_proba(input_scaled)[0][1] # Probability of delay

# --- Display Outputs on Main Page ---
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="⏱️ Predicted Transit Time", value=f"{predicted_hours:.1f} Hours", delta=f"{predicted_days} Days")

with col2:
    risk_label = "🚨 High Risk (Delayed)" if sla_prob > 0.5 else "✅ Safe (On-Time)"
    st.metric(label="📊 SLA Breach Risk", value=f"{sla_prob*100:.1f}%", delta=risk_label)
with col3:
    st.metric(label="🚚 Selected Courier", value=partner)

st.markdown("---")
if sla_prob > 0.5:
    st.warning("⚠️ **Warning:** This shipment has a high probability of breaching its promised SLA deadline based on current distance, weather, and pickup delays. Consider switching to **Air** mode or upgrading to **Blue Dart**.")
else:
    st.success("✨ **Optimized:** This shipment is projected to meet its delivery deadline successfully!")