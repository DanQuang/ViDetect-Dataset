from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Optional,Text
import pandas as pd
import os
from data_utils.utils import preprocess_text

class CustomDataset(Dataset):
    def __init__(self, data, with_label=True):
        self.data = data  # pandas dataframe
        self.with_label = with_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        idx=self.data.loc[index, 'id']
        sent = preprocess_text(str(self.data.loc[index, 'text']))
        if self.with_label:  # True if the dataset has labels
            labels = self.data.loc[index, 'label']
            return sent, labels, idx
        else:
            return sent, idx
        
class Get_Loader:
    def __init__(self, config):
        self.train_path=os.path.join(config['data']['dataset_folder'],config['data']['train_dataset'])
        self.train_batch=config['train']['per_device_train_batch_size']

        self.val_path=os.path.join(config['data']['dataset_folder'],config['data']['val_dataset'])
        self.val_batch=config['train']['per_device_valid_batch_size']

        self.test_path=os.path.join(config['inference']['test_dataset'])
        self.test_batch=config['inference']['batch_size']
        self.with_label = config['inference']['with_label']

    def load_train_dev(self):
        train_df=pd.read_csv(self.train_path)
        val_df=pd.read_csv(self.val_path)
        print("Reading training data...")
        train_set = CustomDataset(train_df)
        print("Reading validation data...")
        val_set = CustomDataset(val_df)
    
        train_loader = DataLoader(train_set, batch_size=self.train_batch, num_workers=2,shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.val_batch, num_workers=2,shuffle=True)
        return train_loader, val_loader
    
    def load_test(self):
        test_df=pd.read_csv(self.test_path)
        print("Reading testing data...")
        test_set = CustomDataset(test_df,with_label=self.with_label)
        test_loader = DataLoader(test_set, batch_size=self.test_batch, num_workers=2, shuffle=False)
        return test_loader
