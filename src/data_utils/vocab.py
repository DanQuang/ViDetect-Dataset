from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Optional,Text
import pandas as pd
import os
from data_utils.utils import preprocess_text

class Vocab:
    def __init__(self, config: Dict):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.train_path=os.path.join(config['data']['dataset_folder'],config['data']['train_dataset'])
        self.val_path=os.path.join(config['data']['dataset_folder'],config['data']['val_dataset'])
        self.test_path=os.path.join(config['inference']['test_dataset'])
        self.build_vocab()
        

    def all_word(self):
        train_df= pd.read_csv(self.train_path)
        val_df=pd.read_csv(self.val_path)
        word_counts = {}
        for df in [train_df,val_df]:
            for index, row in df.iterrows():
                for word in preprocess_text(row['text']).split():
                    if word not in word_counts:
                        word_counts[word] = 1
                    else:
                        word_counts[word] += 1
        special_token=['[UNK]','[CLS]','[SEP]']
        for w in special_token:
            if w not in word_counts:
                word_counts[w]=1
            else:
                word_counts[w]+=1
        sorted_word_counts = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True))
        vocab = list(sorted_word_counts.keys())
        return vocab, sorted_word_counts

        
    def build_vocab(self):
        all_word,_=self.all_word()
        self.word_to_idx = {word: idx+1  for idx, word in enumerate(all_word)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

    def convert_tokens_to_ids(self, tokens):
        return [self.word_to_idx.get(token, 0) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.idx_to_word[idx] for idx in ids]

    def vocab_size(self):
        return len(self.word_to_idx)+1

    def pad_token_id(self):
        return 0  