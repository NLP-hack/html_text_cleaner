import re

import torch
from string import punctuation
from torch.nn.functional import sigmoid
from nltk.tokenize.treebank import TreebankWordDetokenizer


class NorvigAlgorithm:
    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.N = sum(self.dictionary.values())
        self.letters = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'

    def P(self, word):
        return self.dictionary.get(word, 0) / self.N

    def correction(self, word):
        return max(self.candidates(word), key=self.P)

    def candidates(self, word):
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(
            self.edits2(word)) or [word])

    def known(self, words):
        return set(w for w in words if w in self.dictionary)

    def edits1(self, word):
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in self.letters]
        inserts = [L + c + R for L, R in splits for c in self.letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))


class NorvigCorrector:
    """
    Алгоритм Норвига для исправления опечаток
    Соединяет результат с пунктуацией
    """

    def __init__(self, model, tokenizer, corrector, tag2punct, device='cpu',
                 spelling_threshold=0.5, punct_threshold=0.5):
        self.extended_punct = list(tag2punct.values()) + ['!', '?']
        self.TOKEN_SEP = '##'
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.tag2punct = tag2punct
        self.spelling_threshold = spelling_threshold
        self.punct_threshold = punct_threshold
        self.detokenizer = TreebankWordDetokenizer()
        self.corrector = corrector

    def get_predictions(self, tokenized_text):
        self.model.eval()
        punct_output, spelling_output = self.model(
            **{k: v.to(self.device) for k, v in tokenized_text.items()})
        punct_preds = sigmoid(punct_output.cpu())
        punct_preds = punct_preds.where(punct_preds >= self.punct_threshold, 0)
        spelling_preds = (sigmoid(spelling_output.cpu()) >= self.spelling_threshold).int().squeeze(
            -1)
        return punct_preds, spelling_preds

    def merge_tokens(self, tokenized_text, punct_preds, spelling_preds):
        bpe_tokens = self.tokenizer.convert_ids_to_tokens(tokenized_text['input_ids'][0],
                                                          skip_special_tokens=True)
        tokens, misspelled, next_puncts = [], [], []
        token = ''
        is_misspelled = 0
        next_punct = ''
        for bpe_token, punct_pred, spelling_pred in zip(bpe_tokens, punct_preds[0],
                                                        spelling_preds[0]):
            if spelling_pred.item() == 1:
                is_misspelled = 1
            if punct_pred.any() > 0:
                tag = int(torch.argmax(punct_pred))
                next_punct = self.tag2punct[tag]
            if bpe_token.startswith(self.TOKEN_SEP):
                token += bpe_token.replace(self.TOKEN_SEP, '')
            else:
                tokens.append(token)
                misspelled.append(is_misspelled)
                next_puncts.append(next_punct)
                token = bpe_token
                is_misspelled = 0
                next_punct = ''
        tokens.append(token)
        misspelled.append(is_misspelled)
        next_puncts.append(next_punct)
        tokens.pop(0)
        misspelled.pop(0)
        next_puncts.pop(0)
        assert len(tokens) == len(misspelled) == len(next_puncts)
        return tokens, misspelled, next_puncts

    def correct_spelling(self, tokens, misspelled):
        corrected_tokens = []
        for token, is_misspelled in zip(tokens, misspelled):
            if is_misspelled:
                corrected_tokens.append(self.replace(token))
            else:
                corrected_tokens.append(token)
        return corrected_tokens

    def replace(self, token):
        capital = token[0].isupper()
        replacement = self.corrector.correction(token)
        if capital:
            replacement = replacement.capitalize()
        return replacement

    def change_punctuation(self, tokens, next_puncts):
        text = ['']
        model_punct = False
        for token, next_punct in zip(tokens, next_puncts):
            if next_punct and text[-1] not in self.extended_punct:
                if token in punctuation:
                    text.append(next_punct)
                else:
                    text.append(token)
                    text.append(next_punct)
                model_punct = True
            elif token not in self.extended_punct or not model_punct:
                text.append(token)
                model_punct = False
        return [token for token in text if token]

    def detokenize(self, tokens):
        text = self.detokenizer.detokenize(tokens)
        text = (
            text
            .replace(' ,', ',')
            .replace(' .', '.')
            .replace(' :', ':')
            .replace(' ;', ';')
            .replace(' !', '!')
            .replace(' ?', '?')
            .replace(' %', '%')
            .replace(' )', ')')
            .replace('( ', '(')
            .replace(' } ', '}')
            .replace('{ ', '{')
            .replace(' ] ', ']')
            .replace('[ ', '[')
            .replace(' »', '»')
            .replace('« ', '«')
        )

        left_quoted = re.findall('(?<=" )[А-я|A-z|\d|,|\.]+', text)
        right_quoted = re.findall('[А-я|A-z|\d|,|\.]+(?= ")', text)

        for lq in left_quoted:
            text = text.replace(f'" {lq}', f'"{lq}')
        for rq in right_quoted:
            text = text.replace(f'{rq} "', f'{rq}"')
        return text

    def correct(self, text, spelling_threshold=0.5, punct_threshold=0.5):
        self.spelling_threshold = spelling_threshold
        self.punct_threshold = punct_threshold
        text = str(text)
        tokenized_text = self.tokenizer(text, add_special_tokens=False, return_tensors='pt')
        punct_preds, spelling_preds = self.get_predictions(tokenized_text)
        tokens, misspelled, next_puncts = self.merge_tokens(tokenized_text, punct_preds,
                                                            spelling_preds)
        tokens_corrected_spelling = self.correct_spelling(tokens, misspelled)
        corrected_text = self.change_punctuation(tokens_corrected_spelling, next_puncts)
        return self.detokenize(corrected_text)
