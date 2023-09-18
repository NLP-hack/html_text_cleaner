
import os
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from app.src.model.model_classes import Model, MODEL_NAME, MaskCorrector

model_path = os.environ.get("MODEL_PATH", "./model_files/bert_punct_spelling.pt")

PUNCT = ',;:.'
tag2punct = dict(enumerate(PUNCT))
punct2tag = dict(zip(tag2punct.values(), tag2punct.keys()))


def load_model():
    import __main__; setattr(__main__, "Model", Model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Модель определяет слова с ошибками и знаки препинания
    model = torch.load(model_path, map_location=device)

    # Модель заменяет слова с ошибками на ближайшие по левенштейну
    mlm = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    mask_corrector = MaskCorrector(model=model, tokenizer=tokenizer, mlm=mlm, tag2punct=tag2punct,
                                   device=device,
                                   k=10000,
                                   spelling_threshold=0.5, punct_threshold=0.6)
    return mask_corrector
