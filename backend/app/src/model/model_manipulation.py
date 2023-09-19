import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from app.src.model.base_model import Model, MODEL_NAME
from app.src.model.bert_corrector import MaskCorrector
from app.src.model.norvig_corrector import NorvigAlgorithm, NorvigCorrector

MODEL_PATH = os.environ.get("MODEL_PATH", "./model_files/bert_punct_spelling.pt")
DICTIONARY_PATH = os.environ.get("DICTIONARY_PATH", "./model_files/norvig_vocabulary.json")

PUNCT = ',;:.'
tag2punct = dict(enumerate(PUNCT))
punct2tag = dict(zip(tag2punct.values(), tag2punct.keys()))


def load_model(type='bert'):
    import __main__;
    setattr(__main__, "Model", Model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Модель определяет слова с ошибками и знаки препинания
    model = torch.load(MODEL_PATH, map_location=device)

    if type == 'bert':
        mlm = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

        corrector = MaskCorrector(model=model, tokenizer=tokenizer, mlm=mlm,
                                  tag2punct=tag2punct,
                                  device=device,
                                  k=10000)
    elif type == 'norvig':
        with open(DICTIONARY_PATH) as jsfile:
            dictionary = json.load(jsfile)
        algorithm = NorvigAlgorithm(dictionary)
        corrector = NorvigCorrector(model=model, tokenizer=tokenizer, tag2punct=tag2punct,
                                    device=device, corrector=algorithm)
    else:
        raise ValueError("Unknown type of model")
    return corrector
