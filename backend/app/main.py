from fastapi import FastAPI
import logging

from app.src.model.model_manipulation import load_model
from app.src.hltm_annotator.html_manipulation import load_annotator
from app.src.utils import get_result, postprocess_text

model = load_model('norvig')
annotator = load_annotator()
app = FastAPI()


@app.get("/")
def main():
    return {"message": "Text cleaner is running"}


@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}


@app.get("/process_text")
def process_text(input_text: str, spelling_threshold: float = 0.5,
                 punctuation_threshold: float = 0.5):
    res_wo_model = annotator.annotate(input_text)
    res_model = model.correct(input_text, spelling_threshold, punctuation_threshold)

    res = get_result(annotator, model, input_text, spelling_threshold, punctuation_threshold)
    return {
        'only_model': res_model,
        "only_html": postprocess_text(res_wo_model),
        "processed_text": postprocess_text(res),
    }
