# src/demo_streamlit.py
import streamlit as st
import joblib
import cv2
import numpy as np
from preprocess import preprocess_image
from features import extract_features

st.title("Handwriting Mood Classifier — Demo")
model_info = joblib.load("models/handwriting_mood_rf.pkl")
model = model_info['model']

uploaded = st.file_uploader("Upload an image (jpg/png)", type=['jpg','png','jpeg'])
if uploaded is not None:
    img_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
    proc = preprocess_image(img)
    feat = extract_features(proc).reshape(1, -1)
    pred = model.predict(feat)[0]
    st.image(img, caption="Uploaded image", use_column_width=True)
    st.markdown(f"**Predicted mood:** {pred}")
