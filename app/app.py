import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from groq import Groq

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
def get_treatment_advice(disease_name):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    clean_name = disease_name.replace("___", " - ").replace("_", " ")
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": f"A plant has been diagnosed with: {clean_name}. In under 100 words, give a farmer practical, actionable treatment and prevention advice."}
        ],
        max_tokens=200
    )
    return response.choices[0].message.content

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    with st.spinner("Analyzing leaf..."):
        prediction = model.predict(img_array)[0]
        top_3_indices = np.argsort(prediction)[-3:][::-1]

    st.success(f"**Top Prediction:** {class_names[str(top_3_indices[0])]}")
    st.info(f"**Confidence:** {round(float(prediction[top_3_indices[0]]) * 100, 2)}%")

    st.write("---")
    st.subheader("🩺 Treatment Advice")
    with st.spinner("Getting advice..."):
        top_disease = class_names[str(top_3_indices[0])]
        advice = get_treatment_advice(top_disease)
    st.write(advice)

    if prediction[top_3_indices[0]] * 100 < 60:
        st.warning("Confidence is low — try a clearer, well-lit photo of a single leaf for better accuracy.")