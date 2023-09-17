from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def main():
    return {"message": "Text cleaner is running"}


@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}


@app.get("/process_text")
def process_text(text: str):
    # TODO добавить обработку текста
    return {"processed_text": text.upper()}

# TODO добавить обработку файла
