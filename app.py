import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

@st.cache_resource
def load_model():
    model_name = "ชื่อโมเดลบนHuggingFaceของคุณ"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

id2label = {0: "neutral", 1: "depression"}

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = int(torch.argmax(outputs.logits, dim=1))
    return id2label[pred_id]

st.title("WangchanBERTa Sentiment Prediction")

user_input = st.text_area("กรอกข้อความภาษาไทยที่ต้องการทำนาย")

if st.button("ทำนาย"):
    if user_input.strip() == "":
        st.warning("กรุณากรอกข้อความก่อนกดทำนาย")
    else:
        result = predict_sentiment(user_input)
        st.success(f"ผลการทำนาย: {result}")
