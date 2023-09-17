import requests
import os
import streamlit as st

BACKEND_PORT = os.environ.get("BACKEND_PORT", 8000)
BACKEND_URL = f"http://backend:{BACKEND_PORT}"

def process_text(text):
    response = requests.get(f"{BACKEND_URL}/process_text", params={"text": text})
    return response.json()["processed_text"]


st.title("Text CLeaner and HTML App")

input_text = st.text_area("Введите текст для обработки:")

if st.button("Обработать"):
    output_text = process_text(input_text)
    st.text_area("Обработанный текст:", value=output_text)
