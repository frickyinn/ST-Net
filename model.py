import torch
torch.cuda.current_device()
from torch import nn
from torch.utils.data import DataLoader
import torchvision

import os
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support


class STNet(nn.Module):
    def __init__(self, n_classes, backbone='densenet'):
        super(STNet, self).__init__()
        if backbone == 'densenet':
            feature_extracter = torchvision.models.densenet121(pretrained=True)
            layers = list(feature_extracter.children())[:-1]
            self.feature_extracter = nn.Sequential(*layers)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(1024, n_classes)
        else:
            feature_extracter = torchvision.models.resnet101(pretrained=True)
            layers = list(feature_extracter.children())[:-1]
            self.feature_extracter = nn.Sequential(*layers)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(2048, n_classes)
    
    def forward_feature(self, x):
        x = self.feature_extracter(x)
        x = self.avgpool(x).squeeze()
        return x
    
    def forward(self, x):
        x = self.feature_extracter(x)
        x = self.avgpool(x).squeeze()
        x = self.classifier(x)
        return x


class Trainer():
    def __init__(self, n_classes, lr, backbone='densenet', device='cuda'):
        self.model = STNet(n_classes, backbone)
        self.model = self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
    
    def get_features(self, x):
        return self.model.forward_feature(x)
    
    def fit(self, trainset, validset, epochs, valid_epoch=1, num_workers=1):
        save_dir = './output'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=True)
        valid_loader = DataLoader(validset, batch_size=128, num_workers=num_workers, pin_memory=True)
        
        max_f1 = 0
        for epoch in range(epochs):
            with tqdm(total=len(train_loader)) as t:
                self.model.train()
                train_loss = 0
                train_precision = 0
                train_recall = 0
                train_f1 = 0
                train_cnt = 0

                for i, (x, y) in enumerate(train_loader):
                    t.set_description('Epoch {} train'.format(epoch))

                    self.optimizer.zero_grad()
                    x, y = x.to(self.device), y.to(self.device)
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
                    loss.backward()

                    train_cnt += 1
                    train_loss += loss.float().item()
                    precision, recall, f1_micro, _ = precision_recall_fscore_support(y.cpu().numpy(),
                                                           pred.detach().argmax(dim=1).cpu().numpy(), average='macro', zero_division=0)
                    train_precision += precision
                    train_recall += recall
                    train_f1 += f1_micro

                    self.optimizer.step()

                    t.set_postfix(loss='{:.3f}'.format(train_loss/train_cnt), pre='{:.3f}'.format(train_precision/train_cnt),
                                 rec='{:.3f}'.format(train_recall/train_cnt), f1='{:.3f}'.format(train_f1/train_cnt))
                    t.update(1)
            
            if (epoch+1) % valid_epoch == 0:
                with torch.no_grad():
                    with tqdm(total=len(valid_loader)) as t:
                        self.model.eval()
                        valid_loss = 0
                        valid_precision = 0
                        valid_recall = 0
                        valid_f1 = 0
                        valid_cnt = 0

                        for i, (x, y) in enumerate(valid_loader):
                            t.set_description('Epoch {} valid'.format(epoch))

                            x, y = x.to(self.device), y.to(self.device)
                            pred = self.model(x)
                            loss = self.criterion(pred, y)

                            valid_cnt += 1
                            valid_loss += loss.float().item()
                            precision, recall, f1_micro, _ = precision_recall_fscore_support(y.cpu().numpy(), 
                                                           pred.detach().argmax(dim=1).cpu().numpy(), average='macro', zero_division=0)
                            valid_precision += precision
                            valid_recall += recall
                            valid_f1 += f1_micro

                            t.set_postfix(loss='{:.3f}'.format(valid_loss/valid_cnt), pre='{:.3f}'.format(valid_precision/valid_cnt),
                                     rec='{:.3f}'.format(valid_recall/valid_cnt), f1='{:.3f}'.format(valid_f1/valid_cnt))
                            t.update(1)

                        if valid_f1 / valid_cnt > max_f1:
                            max_f1 = valid_f1 / valid_cnt
                            torch.save({
                                'epoch': epoch,
                                'valid_f1': max_f1,
                                'model_state_dict': self.model.state_dict()}, 
                                os.path.join(save_dir, f'checkpoints_f1{max_f1:.3f}.ckpt'))
