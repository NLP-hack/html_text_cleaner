import re

from transformers import AutoModel
from torch.nn import Module, Dropout, Linear, Sequential
import torch
from torch.nn.functional import sigmoid
from Levenshtein import distance as lev_distance
from nltk.tokenize.treebank import TreebankWordDetokenizer
from string import punctuation

MODEL_NAME = 'DeepPavlov/rubert-base-cased'


class Model(Module):
    def __init__(self, pretrained_model_name, num_punct_classes, freeze=True, **kwargs):
        super().__init__()
        self.emb = AutoModel.from_pretrained(pretrained_model_name, output_attentions=False,
                                             output_hidden_states=False)

        self.emb_size = list(self.emb.parameters())[-1].shape[0]
        if freeze:
            for param in self.emb.parameters():
                param.requires_grad = False

        self.punct_hid_size = 392
        self.spelling_hid_size = 392
        self.punctuation_head = Sequential(
            Dropout(p=0.1),
            Linear(self.emb_size, self.punct_hid_size),
            Linear(self.punct_hid_size, num_punct_classes)
        )

        self.spelling_head = Sequential(
            Dropout(p=0.1),
            Linear(self.emb_size, self.spelling_hid_size),
            Linear(self.spelling_hid_size, 1)
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        emb = self.emb(input_ids=input_ids, attention_mask=attention_mask)[0]
        punct_output = self.punctuation_head(emb)
        spelling_output = self.spelling_head(emb)
        return punct_output, spelling_output


class MaskCorrector:
    def __init__(self, model, tokenizer, mlm, tag2punct, device='cpu', k=10,
                 spelling_threshold=0.5, punct_threshold=0.5):
        self.TOKEN_SEP = '##'
        self.MASK = '[MASK]'
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.mlm = mlm.to(self.device)
        self.tag2punct = tag2punct
        self.spelling_threshold = spelling_threshold
        self.punct_threshold = punct_threshold
        self.k = k
        self.mask_id = tokenizer(self.MASK, add_special_tokens=False).input_ids[0]
        self.detokenizer = TreebankWordDetokenizer()

    def get_predictions(self, tokenized_text):
        self.model.eval()
        with torch.no_grad():
            punct_output, spelling_output = self.model(
                **{k: v.to(self.device) for k, v in tokenized_text.items()})
        return punct_output, spelling_output

    def mask_mistakes(self, tokenized_text, spelling_output):
        spelling_pred = (sigmoid(spelling_output.cpu()) >= self.spelling_threshold).int().squeeze(
            -1)
        masked_text = tokenized_text['input_ids'].clone()
        masked_text[spelling_pred == 1] = self.mask_id
        return masked_text, tokenized_text['input_ids'][spelling_pred == 1]

    def select_correction(self, mistakes, candidates, probabilities):
        corrections = []
        for mistake, cands, probs in zip(mistakes, candidates, probabilities):
            lev_threshold = len(mistake)
            levs = []
            for cand, prob in zip(cands, probs):
                lev = lev_distance(cand, mistake)
                if lev < lev_threshold:
                    levs.append((lev, prob.item(), cand))
            levs = sorted(levs, key=lambda x: (x[0], -x[1]))
            if levs:
                corrections.append(levs[0][2])
            else:
                corrections.append(mistake)
        return corrections

    def get_corrections(self, masked_text, masked_tokens):
        self.mlm.eval()
        logits = self.mlm(input_ids=masked_text.to(self.device)).logits
        mask_token_index = torch.where(masked_text == self.mask_id)[1]
        mask_token_logits = logits[0, mask_token_index]
        candidates = torch.topk(mask_token_logits, self.k, dim=1)
        mistakes = self.tokenizer.batch_decode(masked_tokens)
        correction_candidates = [self.tokenizer.batch_decode(c) for c in candidates.indices]
        return mistakes, correction_candidates, candidates.values

    def join_tokens(self, tokens, corrections):
        text = []
        full_token = ''
        for token in tokens:
            if token == self.MASK:
                token = corrections.pop(0)
            if token.startswith(self.TOKEN_SEP):
                full_token += token.replace(self.TOKEN_SEP, '')
            else:
                text.append(full_token)
                full_token = token
        text.append(full_token)
        text.pop(0)
        return text

    def correct_spelling(self, tokenized_text, spelling_output):
        masked_text, masked_tokens = self.mask_mistakes(tokenized_text, spelling_output)
        mistakes, candidates, probabilities = self.get_corrections(masked_text, masked_tokens)
        corrections = self.select_correction(mistakes, candidates, probabilities)
        tokens = self.tokenizer.convert_ids_to_tokens(masked_text[0])
        return tokens, corrections

    # def change_punctuation(self, tokenized_text, punct_output):
    #     tokens = self.tokenizer.convert_ids_to_tokens(tokenized_text['input_ids'][0])
    #     punct_pred = sigmoid(punct_output.cpu())
    #     punct_pred = punct_pred.where(punct_pred >= self.punct_threshold, 0)
    #     text = []
    #     next_punct = ''
    #     for token, pred in zip(tokens, punct_pred[0]):
    #         if token in self.tag2punct.values():
    #             text.append(next_punct)
    #         else:
    #             text.append(next_punct)
    #             text.append(token)
    #         if pred.any() > 0:
    #             tag = int(torch.argmax(pred))
    #             next_punct = self.tag2punct[tag]
    #         else:
    #             next_punct = ''
    #     return [token for token in text if token]
    def change_punctuation(self, tokenized_text, punct_output):
        tokens = self.tokenizer.convert_ids_to_tokens(tokenized_text['input_ids'][0])
        punct_pred = sigmoid(punct_output.cpu())
        punct_pred = punct_pred.where(punct_pred >= self.punct_threshold, 0)
        text = []
        model_punct = False
        for token, pred in zip(tokens, punct_pred[0]):
            if pred.any() > 0:
                tag = int(torch.argmax(pred))
                next_punct = self.tag2punct[tag]
                if token in punctuation:
                    text.append(next_punct)
                else:
                    text.append(token)
                    text.append(next_punct)
                model_punct = True
            elif token not in list(self.tag2punct.values()) + ['!', '?'] or not model_punct:
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
        )
        return text

    def preprocess_text(self, text):
        return re.sub('([а-я.!?])([А-Я])', '\\1 \\2', text)

    def correct(self, text):
        text = str(text)
        text = self.preprocess_text(text)
        tokenized_text = self.tokenizer(text, add_special_tokens=False, return_tensors='pt')
        punct_output, spelling_output = self.get_predictions(tokenized_text)
        tokens, corrections = self.correct_spelling(tokenized_text, spelling_output)
        tokens_corrected_punct = self.change_punctuation(tokenized_text, punct_output)
        corrected_text = self.join_tokens(tokens_corrected_punct, corrections)
        return self.detokenize(corrected_text)
