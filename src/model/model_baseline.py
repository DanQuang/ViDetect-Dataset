from typing import List, Dict, Optional,Text
import torch
import torch.nn as nn
from text_module.build_text_embedding import build_text_embedding
from torch.nn import functional as F

class Baseline(nn.Module):
    def __init__(self,config: Dict):
        super(Baseline, self).__init__()
        self.intermediate_dims = config["model"]["intermediate_dims"]
        self.dropout=config["model"]["dropout"]
        self.d_text = config["text_embedding"]['d_features']
        self.text_embbeding = build_text_embedding(config)
        self.attention_weights = nn.Linear(self.intermediate_dims, 1)
        self.classifier = nn.Linear(self.intermediate_dims, 1)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, text: List[str], labels: Optional[torch.LongTensor] = None):
        embbed, mask = self.text_embbeding(text)
        logits = self.classifier(torch.mean(embbed,dim=1)).squeeze(1)
        logits = F.sigmoid(logits)

        if labels is not None:
            loss = self.criterion(logits, labels)
            return logits,loss
        else:
            return logits


