import streamlit as st
import joblib
import os
import numpy as np

BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "wisdm_model.pkl"))
le = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

st.set_page_config(
    page_title="Real-Time Activity Recognition",
    layout="centered"
)

st.markdown(
    "<h1 style='text-align: center; color: #1f77b4;'>"
    "Real-Time Human Activity Recognition</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>"
    "Enter real-time accelerometer readings to predict human activity</p>",
    unsafe_allow_html=True
)

st.divider()

x_acc = st.number_input("X Acceleration", value=0.0, format="%.4f")
y_acc = st.number_input("Y Acceleration", value=0.0, format="%.4f")
z_acc = st.number_input("Z Acceleration", value=0.0, format="%.4f")

st.divider()

if st.button(" Predict Activity", use_container_width=True):
    input_data = np.array([[x_acc, y_acc, z_acc]])

    pred_label = model.predict(input_data)[0]
    activity_name = le.inverse_transform([pred_label])[0]

    st.success(f"✅ Predicted Activity: **{activity_name}**")
