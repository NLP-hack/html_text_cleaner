from fastapi import FastAPI
from app.src.model.model_manipulation import load_model

model = load_model()
app = FastAPI()


@app.get("/")
def main():
    return {"message": "Text cleaner is running"}


@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}


@app.get("/process_text")
def process_text(text: str):
    return {"processed_text": model.correct(text)}
