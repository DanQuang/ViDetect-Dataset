from typing import List, Dict, Optional,Text
from data_utils.load_data import Get_Loader
import torch
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from eval_metric.evaluate import ScoreCalculator
from model.build_model import build_model

class Predict:
    def __init__(self, config: Dict):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path=os.path.join(config["train"]["output_dir"], "best_model.pth")
        self.model = build_model(config)
        self.dataloader = Get_Loader(config)
        self.compute_score = ScoreCalculator()
    def predict_submission(self):
        # Load the model
        print("Loading the best model...")
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        # Obtain the prediction from the model
        print("Obtaining predictions...")
        test =self.dataloader.load_test()
        submits=[]
        gts=[]
        predicts=[]
        ids=[]
        self.model.eval()
        with torch.no_grad():
            for it, (sents, labels,id) in enumerate(tqdm(test)):
                with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=True):
                    logits = self.model(sents)
                    preds = torch.round(logits)
                submits.extend(logits.cpu().numpy())
                gts.extend(labels.cpu().numpy())
                predicts.extend(preds.cpu().numpy())
                if isinstance(id, torch.Tensor):
                    ids.extend(id.tolist())
                else:
                    ids.extend(id)
        gts=torch.tensor(gts)
        predicts=torch.tensor(predicts)
        test_acc=self.compute_score.acc(gts,predicts)
        test_f1=self.compute_score.f1(gts,predicts)  
        test_auc=self.compute_score.auc(gts,predicts)      
        print(f"test acc: {test_acc:.4f} test f1: {test_f1:.4f} test auc: {test_auc:.4f}")             
        data = {'id': ids,'generated': submits,'predicts':predicts}
        df = pd.DataFrame(data)
        df.to_csv('./submission.csv', index=False)