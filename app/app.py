import streamlit as st
import numpy as np
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from groq import Groq
import cv2
import tensorflow as tf
from fpdf import FPDF
from datetime import datetime
import tempfile
import csv

LAST_CONV_LAYER = "Conv_1"

st.set_page_config(page_title="CropCare AI", page_icon="🌿", layout="centered")
st.title("🌿 CropCare AI — Plant Disease Detector")
st.write("Upload a photo of a plant leaf, and the AI will diagnose potential diseases.")


@st.cache_resource
def load_trained_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model = load_model(os.path.join(BASE_DIR, 'best_model.keras'))
    with open(os.path.join(BASE_DIR, 'class_names.json'), 'r') as f:
        class_names = json.load(f)
    return model, class_names


model, class_names = load_trained_model()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_diagnosis" not in st.session_state:
    st.session_state.last_diagnosis = None


def get_treatment_advice(disease_name, language="English"):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    clean_name = disease_name.replace("___", " - ").replace("_", " ")
    prompt = (
        f"A plant has been diagnosed with: {clean_name}. "
        f"In under 100 words, give a farmer practical, actionable treatment and prevention advice. "
        f"Respond entirely in {language}, using simple, everyday language a farmer would understand."
    )
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content

def get_chatbot_response(chat_history, current_diagnosis=None):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    system_context = (
        "You are an agricultural assistant helping farmers with plant health, "
        "diseases, pesticides, fertilizers, and crop care questions. "
        "Keep answers practical, concise (under 150 words), and easy to understand."
    )
    if current_diagnosis:
        system_context += f" The user's most recently diagnosed plant issue was: {current_diagnosis}."

    messages = [{"role": "system", "content": system_context}]
    for msg in chat_history[-6:]:
        if msg.get("content"):
            messages.append(msg)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=300
    )

    answer = response.choices[0].message.content
    return answer if answer else "Sorry, I couldn't generate a response. Please try rephrasing your question."

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_gradcam(original_img, heatmap, alpha=0.4):
    img = np.array(original_img.resize((224, 224)))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed


def generate_pdf_report(original_img, gradcam_img, disease_name, confidence, top_3_list, advice_text):
    clean_name = disease_name.replace("___", " - ").replace("_", " ")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as orig_file:
        original_img.resize((300, 300)).save(orig_file.name)
        orig_path = orig_file.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as heatmap_file:
        Image.fromarray(gradcam_img).resize((300, 300)).save(heatmap_file.name)
        heatmap_path = heatmap_file.name

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "CropCare AI - Diagnosis Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, datetime.now().strftime("Generated on %B %d, %Y at %I:%M %p"), ln=True, align="C")
    pdf.ln(6)

    pdf.image(orig_path, x=20, y=pdf.get_y(), w=80)
    pdf.image(heatmap_path, x=110, y=pdf.get_y(), w=80)
    pdf.ln(85)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, f"Diagnosis: {clean_name}", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Confidence: {confidence}%", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Other Possibilities:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for name, conf in top_3_list:
        pdf.cell(0, 7, f"- {name}: {conf}%", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Treatment Advice:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, advice_text.encode('latin-1', 'replace').decode('latin-1'))

    return bytes(pdf.output())

def save_to_history(disease_name, confidence):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    history_path = os.path.join(BASE_DIR, 'history.csv')
    clean_name = disease_name.replace("___", " - ").replace("_", " ")
    
    file_exists = os.path.exists(history_path)
    with open(history_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "disease", "confidence"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), clean_name, confidence])


def load_history():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    history_path = os.path.join(BASE_DIR, 'history.csv')
    if not os.path.exists(history_path):
        return []
    with open(history_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)




language = st.selectbox(
    "Choose advice language:",
    ["English", "Hindi", "Marathi", "Tamil", "Telugu", "Bengali"]
)

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
    st.write("**Other possibilities:**")
    for idx in top_3_indices[1:]:
        name = class_names[str(idx)]
        conf = round(float(prediction[idx]) * 100, 2)
        st.write(f"- {name}: {conf}%")

    
    st.write("---")
    st.subheader("🔍 What the AI Focused On")
    heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER, pred_index=int(top_3_indices[0]))
    overlayed_img = overlay_gradcam(img, heatmap)
    st.image(overlayed_img, caption="Red/yellow areas influenced the diagnosis most", use_container_width=True)

    
    st.write("---")
    st.subheader("🩺 Treatment Advice")
    with st.spinner("Getting advice..."):
        top_disease = class_names[str(top_3_indices[0])]
        advice = get_treatment_advice(top_disease, language)
    st.write(advice)
    save_to_history(top_disease, round(float(prediction[top_3_indices[0]]) * 100, 2))
    st.session_state.last_diagnosis = top_disease.replace("___", " - ").replace("_", " ")

    if prediction[top_3_indices[0]] * 100 < 60:
        st.warning("Confidence is low — try a clearer, well-lit photo of a single leaf for better accuracy.")

   
    st.write("---")
    top_3_list = [
        (class_names[str(idx)].replace("___", " - ").replace("_", " "), round(float(prediction[idx]) * 100, 2))
        for idx in top_3_indices[1:]
    ]
    pdf_bytes = generate_pdf_report(
        img,
        overlayed_img,
        top_disease,
        round(float(prediction[top_3_indices[0]]) * 100, 2),
        top_3_list,
        advice
    )
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name="cropcare_diagnosis_report.pdf",
        mime="application/pdf"
    )
    st.write("---")
    st.subheader("📊 Recent Diagnosis History")
    st.caption("Note: history reflects recent activity on this app instance and may reset periodically.")

    history = load_history()
    if len(history) == 0:
        st.write("No diagnoses recorded yet.")
    else:
        import pandas as pd
        df = pd.DataFrame(history)
        df['confidence'] = df['confidence'].astype(float)

        st.dataframe(df.tail(10).iloc[::-1], use_container_width=True)

        st.write("**Most Frequently Detected:**")
        disease_counts = df['disease'].value_counts().head(5)
        st.bar_chart(disease_counts)
    st.write("---")
    st.subheader("💬 Ask CropCare AI")
    st.caption("Ask about plant diseases, pesticides, fertilizers, or general crop care.")

    if st.session_state.last_diagnosis:
        st.caption(f"Currently discussing: {st.session_state.last_diagnosis}")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
           st.write(msg["content"])

    user_question = st.chat_input("Type your question here...")

    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = get_chatbot_response(
                st.session_state.chat_history,
                st.session_state.last_diagnosis
            )
        st.write(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})





















