from transformers import AutoModel
from torch.nn import Module, Dropout, Linear, Sequential

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
