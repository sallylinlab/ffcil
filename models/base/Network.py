import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *

class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        # self.num_features = 512
        if self.args.dataset in ['cifar100']:
            self.encoder = resnet20()
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet']:
            self.encoder = resnet18(False, args)  # pretrained=False
            self.num_features = 512
        if self.args.dataset in ['auo', 'stanford_dogs']:
            if self.args.model_dir is not None:
                if 'resnet50' in args.model_dir:
                    self.encoder = resnet50(False, args) 
                    self.num_features = 2048
                else:
                    self.encoder = resnet18(False, args) 
                    self.num_features = 512
            else:
                self.encoder = resnet18(True, args) 
                self.num_features = 512


        if self.args.dataset == 'cub200':
            self.encoder = resnet18(True, args)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 512
            
        print(f'num of features: {self.num_features}')
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)

        self.projection = nn.Sequential(
            nn.Linear(self.num_features, self.num_features*2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.num_features*2, self.num_features*2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.num_features*2, self.num_features, bias=False)
        )


    def projection_layer(self, x):
        identity = x
        x = self.projection(x)


        return x


    def forward_metric(self, x):
        x = self.encode(x)
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)

        return x

    def pre_encode(self,x):
        if self.args.dataset in ['cifar100','manyshotcifar']:
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)
            
        elif self.args.dataset in ['mini_imagenet','manyshotmini','cub200','auo','stanford_dogs']:
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.maxpool(x)
            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)
        
        return x
    
    def post_encode(self,x , feat=False, project =False):
        if self.args.dataset in ['cifar100','manyshotcifar']:
            x = self.encoder.layer3(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)

        elif self.args.dataset in ['mini_imagenet','manyshotmini','cub200','auo','stanford_dogs']:
            x = self.encoder.layer3(x)
            x = self.encoder.layer4(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)
            x = torch.flatten(x, 1)

        if project:
            x = self.projection_layer(x)
        
        if feat:
            return x
        
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x

            
        return x

    def encode(self, x, project=False):
        if project:
            x = self.projection_layer(x)
            return x

        x = self.encoder(x)
        if self.args.model_dir is not None:
            if 'swin' in self.args.model_dir:
                x = x.mean(dim=1)
                return x
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        x = torch.flatten(x, 1)

        
        
        return x

    def forward(self, input, project=False):
        if self.mode != 'encoder':
            input = self.forward_metric(input)
            return input
        elif self.mode == 'encoder':
            input = self.encode(input, project)
            return input
        else:
            raise ValueError('Unknown mode')

    def update_fc(self,dataloader,class_list,session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data).detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
            # print('not data init \nnew_fc param:', new_fc)
        else:
            new_fc = self.update_fc_avg(data, label, class_list)
            # print('update fc avg \nnew_fc param:', new_fc)

        if 'ft' in self.args.new_mode:  # further finetune
            self.update_fc_ft(new_fc,data,label,session)
            print(' further finetune ')

    def update_fc_avg(self,data,label,class_list):
        new_fc=[]
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    def get_logits(self,x,fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x,fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def update_fc_ft(self,new_fc,data,label,session):
        new_fc=new_fc.clone().detach()
        new_fc.requires_grad=True
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)

        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):
                old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data,fc)
                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

        self.fc.weight.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(new_fc.data)
