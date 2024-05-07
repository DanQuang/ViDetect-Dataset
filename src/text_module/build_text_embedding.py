from text_module.pretrained_embedding import Pretrained_Embedding
from text_module.usual_embedding import Usual_Embedding

def build_text_embedding(config):
    if config["text_embedding"]['type']=='usual':
        return Usual_Embedding(config)
    if config["text_embedding"]['type']=='pretrained':
        return Pretrained_Embedding(config)