from typing import List, Dict, Optional,Text
from data_utils.load_data import Get_Loader
import torch
import os
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from tqdm import tqdm
from eval_metric.evaluate import ScoreCalculator
from model.build_model import build_model

class LLM_Detec_Gen_Task:
    def __init__(self, config: Dict):
        self.num_epochs = config['train']['num_train_epochs']
        self.patience = config['train']['patience']
        self.learning_rate = config['train']['learning_rate']
        self.save_path = config['train']['output_dir']
        self.best_metric= config['train']['metric_for_best_model']
        self.weight_decay=config['train']['weight_decay']
        self.dataloader = Get_Loader(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model=build_model(config)
        self.base_model.to(self.device)
        self.compute_score = ScoreCalculator()
        self.optimizer = optim.Adam(self.base_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler()
        lambda1 = lambda epoch: 0.95 ** epoch
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
    
    def training(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    
        train, valid = self.dataloader.load_train_dev()

        if os.path.exists(os.path.join(self.save_path, 'last_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'last_model.pth'))
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['self.optimizer_state_dict'])
            print('loaded the last saved model!!!')
            initial_epoch = checkpoint['epoch'] + 1
            print(f"continue training from epoch {initial_epoch}")
        else:
            initial_epoch = 0
            print("first time training!!!")

        if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'))
            best_score = checkpoint['score']
        else:
            best_score = 0.
            
        threshold=0
        self.base_model.train()
        for epoch in range(initial_epoch, self.num_epochs + initial_epoch):
            valid_acc = 0.
            valid_f1 = 0.
            train_loss = 0.
            for it, (sents, labels, id) in enumerate(tqdm(train)):
                with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=True):
                    logits, loss = self.base_model(sents, labels.to(dtype=torch.float32, device=self.device))
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                train_loss += loss
            self.scheduler.step()
            train_loss /=len(train)

            with torch.no_grad():
                for it, (sents, labels, id) in enumerate(tqdm(valid)):
                    with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=True):
                        logits = self.base_model(sents)
                        preds = torch.round(logits)
                    valid_acc+=self.compute_score.acc(labels,preds)
                    valid_f1+=self.compute_score.f1(labels,preds)
            valid_acc /= len(valid)
            valid_f1 /= len(valid)
            # valid_auc /= len(valid)

            print(f"epoch {epoch + 1}/{self.num_epochs + initial_epoch}")
            print(f"train loss: {train_loss:.4f}")
            print(f"valid acc: {valid_acc:.4f} valid f1: {valid_f1:.4f}")

            with open('log.txt', 'a') as file:
                file.write(f"epoch {epoch + 1}/{self.num_epochs + initial_epoch}\n")
                file.write(f"train loss: {train_loss:.4f}\n")
                file.write(f"valid acc: {valid_acc:.4f} valid f1: {valid_f1:.4f}\n")

            if self.best_metric =='accuracy':
                score=valid_acc
            if self.best_metric=='f1':
                score=valid_f1
            # save the last model
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.base_model.state_dict(),
                'self.optimizer_state_dict': self.optimizer.state_dict(),
                'score': score}, os.path.join(self.save_path, 'last_model.pth'))
            
            # save the best model
            if epoch > 0 and score <= best_score:
                threshold += 1
            else:
                threshold = 0

            if score > best_score:
                best_score = score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.base_model.state_dict(),
#                     'self.optimizer_state_dict': self.optimizer.state_dict(),
                    'score':score}, os.path.join(self.save_path, 'best_model.pth'))
                print(f"saved the best model with {self.best_metric} of {score:.4f}")
            
            # early stopping
            if threshold >= self.patience:
                print(f"early stopping after epoch {epoch + 1}")
                break
