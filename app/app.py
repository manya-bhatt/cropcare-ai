import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

st.set_page_config(page_title="CropCare AI", page_icon="🌿", layout="centered")
st.title("🌿 CropCare AI — Plant Disease Detector")
st.write("Upload a photo of a plant leaf, and the AI will diagnose potential diseases.")

@st.cache_resource
def load_trained_model():
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model = load_model(os.path.join(BASE_DIR, 'best_model.keras'))
    with open(os.path.join(BASE_DIR, 'class_names.json'), 'r') as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_trained_model()

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    with st.spinner("Analyzing leaf..."):
        prediction = model.predict(img_array)
        predicted_index = int(np.argmax(prediction[0]))
        predicted_class = class_names[str(predicted_index)]
        confidence = round(float(np.max(prediction[0])) * 100, 2)

    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**Confidence:** {confidence}%")

    if confidence < 60:
        st.warning("Confidence is low — try a clearer, well-lit photo of a single leaf for better accuracy.")