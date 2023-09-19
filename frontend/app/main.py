import requests
import os
import streamlit as st

BACKEND_PORT = os.environ.get("BACKEND_PORT", 8000)
BACKEND_URL = f"http://backend:{BACKEND_PORT}"


def process_text(text, spelling_threshold, punctuation_threshold):
    response = requests.get(f"{BACKEND_URL}/process_text", params={
        "input_text": text,
        "spelling_threshold": spelling_threshold,
        "punctuation_threshold": punctuation_threshold
    })
    return response.json()["processed_text"]


def create_markdown(text):
    return f"""
    <div style="background-color: #FFFFFF; padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
    {text}
    </div>
    """


st.title("Text CLeaner and HTML App")

spelling_threshold = st.slider(
    "Порог для исправления орфографии",
    min_value=0.,
    max_value=1.0,
    value=st.session_state.get("spelling_threshold", 0.5),
    step=0.01,
    help="Чем больше, тем меньше модель будет исправлять орфографию"
)

punctuation_threshold = st.slider(
    "Порог для исправления пунктуации",
    min_value=0.,
    max_value=1.0,
    value=st.session_state.get("punctuation_threshold", 0.3),
    step=0.01,
    help="Чем больше, тем меньше модель будет исправлять пунктуацию"
)

input_text = st.text_area("Введите текст для обработки:")

if st.button("Обработать"):
    output_text = process_text(input_text, spelling_threshold, punctuation_threshold)
    st.caption('Обработанный текст:')
    st.code(output_text)

    st.caption('Обработанный текст в виде HTML:')
    st.markdown(create_markdown(output_text), unsafe_allow_html=True)
