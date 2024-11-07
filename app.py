import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# โหลดโมเดลและ tokenizer จาก Hugging Face
model_name = "Imboon/distilbert-phishing-email"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained("1aurent/distilbert-base-multilingual-cased-finetuned-email-spam")

# สร้างฟังก์ชันสำหรับการทำนาย
def predict_phishing(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=1)
    return predictions[0][1].item()  # ดึงค่า confidence ของคลาส phishing

st.title("Phishing Email Detector")
user_input = st.text_area("Enter the email content:")

if st.button("Predict"):
    probability = predict_phishing(user_input)
    if probability > 0.5:
        st.write(f"Warning! This email is likely a phishing attempt. Confidence: {probability:.2f}")
    else:
        st.write(f"This email appears safe. Confidence: {1 - probability:.2f}")
