import torch
from torch import nn
from torch.nn import functional as F
from transformers import  AutoModel, AutoTokenizer
from typing import List, Dict, Optional
from text_module.utils import generate_padding_mask

class Pretrained_Embedding(nn.Module):
    def __init__(self, config: Dict):
        super(Pretrained_Embedding,self).__init__()
        self.pretrained_name=config["text_embedding"]["text_encoder"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_name)
        self.embedding = AutoModel.from_pretrained(self.pretrained_name)
        self.freeze = config['text_embedding']['freeze']
        # freeze all parameters of pretrained model
        if self.freeze:
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.proj = nn.Linear(config["text_embedding"]['d_features'], config["text_embedding"]['d_model'])
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config["text_embedding"]['dropout'])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.padding = config["tokenizer"]["padding"]
        self.max_length = config["tokenizer"]["max_length"]
        self.truncation = config["tokenizer"]["truncation"]
        self.return_attention_mask = config["tokenizer"]["return_attention_mask"],

    def forward(self, text: List[str]):
        inputs = self.tokenizer(
            text = text,
            padding = self.padding,
            max_length = self.max_length,
            truncation = self.truncation,
            return_tensors = 'pt',
            return_attention_mask = self.return_attention_mask,
        ).to(self.device)
        if 't5' in self.pretrained_name:
            features = self.embedding.encoder(**inputs).last_hidden_state
        else:
            features = self.embedding(**inputs).last_hidden_state

        padding_mask = generate_padding_mask(inputs.input_ids, padding_idx=self.tokenizer.pad_token_id)
        out = self.proj(features)
        out = self.dropout(self.gelu(out))
        return out, padding_mask
