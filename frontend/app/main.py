import requests
import os
import streamlit as st

BACKEND_PORT = os.environ.get("BACKEND_PORT", 8000)
BACKEND_URL = f"http://backend:{BACKEND_PORT}"


def process_text(text):
    response = requests.get(f"{BACKEND_URL}/process_text", params={"text": text})
    return response.json()["processed_text"]


def create_markdown(text):
    return f"""
    <div style="background-color: #FFFFFF; padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
    {text}
    </div>
    """


st.title("Text CLeaner and HTML App")

input_text = st.text_area("Введите текст для обработки:")

if st.button("Обработать"):
    output_text = process_text(input_text)
    st.caption('Обработанный текст:')
    st.code(output_text)

    st.caption('Обработанный текст в виде HTML:')
    st.markdown(create_markdown(output_text), unsafe_allow_html=True)
