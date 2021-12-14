from transformers import DistilBertForSequenceClassification
from torch import nn
import torch

class DistilBertClassifier(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.embDim = config.dim

    def forward(self, x, last=False, freeze=False):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        # if not last:
        #     outputs = super().__call__(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #     )[0]
        #     return outputs
        if freeze:
            with torch.no_grad():
                distilbert_output = self.distilbert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
                pooled_output = hidden_state[:, 0]  # (bs, dim)
                pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
                pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
                e = self.dropout(pooled_output)  # (bs, dim)
                logits = self.classifier(e)  # (bs, num_labels)
        else:
            distilbert_output = self.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
            pooled_output = hidden_state[:, 0]  # (bs, dim)
            pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
            pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
            e = self.dropout(pooled_output)  # (bs, dim)
            logits = self.classifier(e)  # (bs, num_labels)
        if last:
            return logits, e
        else:
            return logits
        return

    def get_embedding_dim(self):
        return self.embDim