import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

@st.cache_resource
def load_model():
    model_name = "lnwoak/wangchanberta-sentiment"  # ชื่อ repo โมเดลใน Hugging Face ของคุณ
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

st.title("WangchanBERTa Sentiment Analysis")

text = st.text_area("กรุณากรอกข้อความภาษาไทย")

if st.button("ทำนาย"):
    if text.strip() == "":
        st.warning("กรุณากรอกข้อความก่อนกดทำนาย")
    else:
        label = predict_sentiment(text)
        st.success(f"ผลการทำนาย: {label}")
