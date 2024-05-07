from typing import List, Dict, Optional,Text
import torch
import torch.nn as nn
from encoder_module.encoder import UniModalEncoder
from text_module.build_text_embedding import build_text_embedding
from torch.nn import functional as F

class Model_Trans(nn.Module):
    def __init__(self,config: Dict):
        super(Model_Trans, self).__init__()
        self.intermediate_dims = config["model"]["intermediate_dims"]
        self.dropout=config["model"]["dropout"]
        self.d_text = config["text_embedding"]['d_features']
        self.text_embbeding = build_text_embedding(config)
        self.encoder = UniModalEncoder(config)
        self.attention_weights = nn.Linear(self.intermediate_dims, 1)
        self.classifier = nn.Linear(self.intermediate_dims, 1)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, text: List[str], labels: Optional[torch.LongTensor] = None):
        embbed, mask = self.text_embbeding(text)
        encoded_feature = self.encoder(embbed, mask)
        feature_attended = self.attention_weights(torch.tanh(encoded_feature))
        attention_weights = torch.softmax(feature_attended, dim=1)
        feature_attended = torch.sum(attention_weights * encoded_feature, dim=1)
        
        logits = self.classifier(feature_attended).squeeze(1)
        logits = F.sigmoid(logits)

        if labels is not None:
            loss = self.criterion(logits, labels)
            return logits,loss
        else:
            return logits


