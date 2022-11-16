#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 21:29:47 2022

@author: lingang
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import accuracy_score,f1_score
import os
import random
import h5py
from biosppy.signals import ecg
from biosppy.signals import tools as st 
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

class GetLoader2(torch.utils.data.Dataset):
    def __init__(self):
        with h5py.File("./data1/data_hiddenall3.hdf5", 'r') as f:
            x_data = (np.array(f['data']))
            y_data = np.array(f['label'])
        
        self.dataset=torch.from_numpy(x_data)
        self.labels=torch.from_numpy(y_data)
        self.n_data = len(x_data)
        
    def __getitem__(self, item):
        return self.dataset[item], self.labels[item]

    def __len__(self):
        return self.n_data
    

def init_weight(net,restore):
    net.load_state_dict(torch.load(restore))
    print("Restore model from: {}".format(restore))
    return net

def load_hidden_data():
    dataset1 = GetLoader2()
    dataloader1 = torch.utils.data.DataLoader(dataset=dataset1,batch_size=128,shuffle=False)
    return dataloader1

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv1d(2, 1, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

def conv_1d(in_planes, out_planes, stride=1, size=3):
    return nn.Conv1d(in_planes, out_planes, kernel_size=size, stride=stride, padding=(size-stride)//2, bias=False)

class BasicBlock1d(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, size=3, res=True):
        super(BasicBlock1d, self).__init__()
        self.inplanes=inplanes
        self.planes=planes
        self.conv1 = conv_1d(inplanes, planes, stride, size=size)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU()
        self.conv2 = conv_1d(planes, planes, size=size)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv_1d(planes, planes, size=size)
        self.bn3 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        self.res = res
        self.se = nn.Sequential(
            nn.Linear(planes, planes//4),
            nn.ReLU(),
            nn.Linear(planes//4, planes),
            nn.Sigmoid())
        
        self.shortcut = nn.Sequential(
            nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(planes))
        

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)   
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out) 
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out) 

        if self.res:
            if self.downsample is not None:
                residual = self.downsample(x)
            if self.inplanes!=self.planes:
                residual = self.shortcut(residual)
            out += residual
        return out


class ECGNet(nn.Module):
    def __init__(self,num_classes=6):
        super(ECGNet, self).__init__()
        self.model_name=4
        self.conv1 = nn.Conv1d(1, 32, kernel_size=60, stride=2, padding=29,bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=20, stride=2, padding=9,bias=False)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        layers = []
        layers.append(BasicBlock1d(32, 32, stride=1, downsample=None, size=5, res=True))
        layers.append(BasicBlock1d(32, 32, stride=1, downsample=None, size=5, res=True))
        layers.append(nn.MaxPool1d(2))
        layers.append(BasicBlock1d(32, 64, stride=1, downsample=None, size=5, res=True))
        layers.append(BasicBlock1d(64, 64, stride=1, downsample=None, size=5, res=True))
        layers.append(nn.MaxPool1d(2))
        layers.append(BasicBlock1d(64, 128, stride=1, downsample=None, size=5, res=True))
        layers.append(BasicBlock1d(128, 128, stride=1, downsample=None, size=5, res=True))
        layers.append(nn.MaxPool1d(2))
        layers.append(BasicBlock1d(128, 256, stride=1, downsample=None, size=5, res=True))
        layers.append(BasicBlock1d(256, 256, stride=1, downsample=None, size=5, res=True))
        layers.append(nn.MaxPool1d(2))
        self.layers1=nn.Sequential(*layers)
        
        self.attn_1 = SpatialAttention(kernel_size=3)
        
        self.lstm1 = nn.LSTM(input_size=256,hidden_size=256,batch_first=True,bidirectional=True)
        self.sigmoid=nn.Sigmoid()
        self.maxpool = nn.MaxPool1d(2)
        self.bottle = nn.Sequential(
            nn.Linear(512*64, 64),
            nn.ReLU())
        
        self.bottle1 = nn.Sequential(
            nn.Linear(256*64, 64),
            nn.ReLU())
        
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x0):
        batch_size = x0.size()[0]
        x0 = self.conv1(x0)     
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.conv2(x0)
        conv_out=self.layers1(x0)
        attn_1_out = self.attn_1(conv_out)
        attn_1_out = conv_out*attn_1_out
        attn_1_out = attn_1_out.permute(0, 2, 1)  
        lstm_output, (final_hidden_state, final_cell_state) = self.lstm1(attn_1_out)
        lstm_output = lstm_output.permute(0, 2, 1)
            
        out = self.fc(self.bottle(lstm_output.contiguous().view(batch_size,512*64)))
        return out
    
def round_compute(prev):
    prev1=np.zeros((len(prev),len(prev[0])))
    for i in range(len(prev)):
        for j in range(len(prev[i])):
            if prev[i][j]>=0.5:
                prev1[i][j]=1
            else:
                prev1[i][j]=0
    return prev1

def test(model):
    dataloader1 = load_hidden_data()
    model.eval()
    true_value=np.zeros((1,6))
    pred_value=np.zeros((1,6))
    with torch.no_grad():
        for _, (t_img, t_label) in enumerate(dataloader1):
            t_img, t_label = t_img.type(torch.FloatTensor),t_label.type(torch.FloatTensor)
            t_img, t_label = t_img.to(DEVICE), t_label.to(DEVICE)
            class_output = model(t_img)
            class_output=nn.Sigmoid()(class_output)
            class_output_np=class_output.to(torch.device('cpu')).numpy()
            label_np=t_label.to(torch.device('cpu')).numpy()
            true_value=np.concatenate((true_value, label_np))
            pred_value=np.concatenate((pred_value, round_compute(class_output_np)))
            
    true_value=true_value[1:]
    pred_value=pred_value[1:]
    for i in range(len(pred_value)):
        if pred_value[i][0]==1 and sum(pred_value[i][1:])>0:
            pred_value[i][0]=0
        if sum(pred_value[i][:])==0:
            pred_value[i][0]=1
    return pred_value

def getPoints(x_data):
    signal=x_data[1,:]
    a=ecg.ecg(signal=signal, sampling_rate=500.0, show=False)
    b=a.as_dict()
    rpeaks = b['rpeaks']
    return rpeaks

def heartratejudge2(rpeaks):
    rrtimes=[]
    for i in range(len(rpeaks)-3):
        rrtimes.append(round(2*60/((rpeaks[i+2]-rpeaks[i])/500)))
    low,normal,high=0,0,0
    for rrtime in rrtimes:
        if rrtime <60:
            low+=1
        elif rrtime >100:
            high+=1
        else:
            normal+=1
    RRTime=80
    
    if low<int(len(rrtimes)*0.4) and high<int(len(rrtimes)*0.4):
        RRTime=80
    else:
        if low>high:
            RRTime=50
        else:
            RRTime=120
    return rrtimes,RRTime

def rule1(rpeaks,label,RRtime):
    rpeaks1=np.ediff1d(rpeaks)
    if (np.max(rpeaks1)-np.min(rpeaks1))>60:
        label[0]=0
        
    if RRtime<60:
        label[0]=0
        label[2]=1
        
    elif RRtime>100:
        label[0]=0
        label[1]=1
        
    else:
        a=1
    return label

def rule2(rpeaks,label,RRtime):
    if RRtime<60:
        label[1]=0
        label[2]=1
    elif RRtime>100:
        a=1
    else:
        label[1]=0
        label[0]=1
    return label

def rule3(label,RRtime):
    if RRtime>=60 and RRtime<=100:
        label[2]=0
        label[0]=1
        
    return label

def postprocess(pred_value):
    filenames=pd.read_csv("./data1/hiddensetnames.csv",encoding='utf-8').loc[:,'FileName'].tolist()
    for i in range(len(pred_value)):
        path="./data1/original_data/"
        filename=filenames[i]
        try:
            if pred_value[i][0]==1 and sum(pred_value[i][1:])>0:
                pred_value[i][0]=0
                
            if sum(pred_value[i][:])==0:
                pred_value[i][0]=1
    
            if pred_value[i][0]==1:
                x_data=np.load(path+filename)
                rpeaks=getPoints(x_data)
                rrtimes,RRtime=heartratejudge2(rpeaks)
                
                label=rule1(rpeaks,pred_value[i],RRtime)
                
                pred_value[i]=label
    
            else:
                if pred_value[i][2]==1:
                    x_data=np.load(path+filename)
                    rpeaks=getPoints(x_data)
                    rrtimes,RRtime=heartratejudge2(rpeaks)
                    label=rule3(pred_value[i],RRtime)
                    pred_value[i]=label
                    
                if pred_value[i][1]==1:
                    x_data=np.load(path+filename)
                    rpeaks=getPoints(x_data)
                    rrtimes,RRtime=heartratejudge2(rpeaks)  
                    label=rule2(rpeaks,pred_value[i],RRtime)
                    pred_value[i]=label
                    
        except:
            continue
    return pred_value
    
if __name__ == '__main__':
    model = ECGNet(num_classes=6).to(DEVICE)
    model=init_weight(model,'./model_submit/ECGNet/best_model.pth')
    print('test set:')
    pred_value=test(model)
    del model
    
    true_value=np.load("./data1/true_value1.npy")
    pred_value=postprocess(pred_value)
    
    F1_macro=f1_score(true_value,pred_value,average='macro')

    acc1=accuracy_score(true_value,pred_value)
    print('Acc: {:.4f},F1_macro: {:.4f}'.format(acc1,  F1_macro))

    