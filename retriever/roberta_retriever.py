from transformers import AutoModel
import torch.nn as nn
import torch


class RobertaRetriever(nn.Module):

    def __init__(self,
                 config,
                 model_name,
                 cache_dir=None
                 ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.project = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                     nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))

    def encode_seq(self, input_ids, mask):
        cls_rep = self.encoder(input_ids, mask)[0][:, 0, :]
        vector = self.project(cls_rep)
        return vector

    def forward(self, batch=None, **kwargs):
        if batch is None:
            batch = kwargs

        c1 = self.encode_seq(batch['c1_input_ids'], batch['c1_mask'])
        c2 = self.encode_seq(batch['c2_input_ids'], batch['c2_mask'])

        neg_1 = self.encode_seq(batch['neg1_input_ids'], batch['neg1_mask'])
        neg_2 = self.encode_seq(batch['neg2_input_ids'], batch['neg2_mask'])

        q = self.encode_seq(batch['q_input_ids'], batch['q_mask'])
        q_sp1 = self.encode_seq(batch['q_sp_input_ids'], batch['q_sp_mask'])
        vectors = {'q': q, 'c1': c1, "c2": c2, "neg_1": neg_1, "neg_2": neg_2, "q_sp1": q_sp1}
        return vectors

    def encode_q(self, input_ids, q_mask, q_type_ids):
        return self.encode_seq(input_ids, q_mask)
