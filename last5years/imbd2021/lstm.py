#%%
# -*- coding: utf-8 -*-
"""
2021 IMBD
"""
#通用函式庫
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import lr_scheduler
from sklearn.metrics import mean_absolute_error

step = 30 # 時序步數

train_temp = pd.read_csv('./data/train.csv')
test2_temp = pd.read_csv('./data/test.csv')
#%%
#依照訓練集數據長度建立空的資料結構
n_data = train_temp.shape[0]
time_step = step
data = np.zeros((n_data,12,time_step),dtype = 'float32')

row_start = 0
row_end = 0
tar_cols = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12']
#依照設定的步數處理數據
row_end = len(train_temp)
for j in range(time_step):
    zero_cols = time_step - j - 1
    train = pd.DataFrame([[0] * len(tar_cols)]*zero_cols,columns = tar_cols)
    if zero_cols == 0:
        train = train.append(train_temp[tar_cols][:])
    else:
        train = train.append(train_temp[tar_cols][:-zero_cols])
    data[row_start:row_end,:,j] = train

tar_cols = ['O']
data_y = np.zeros((n_data,1),dtype = 'float32')
row_start = 0
row_end = len(train_temp)
data_y[row_start:row_end,:] = train_temp[tar_cols]

#%%
#依照測試集數據長度建立空的資料結構
n_data = test2_temp.shape[0]
tar_cols = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12']

time_step = step
test2_data = np.zeros((n_data,12,time_step),dtype = 'float32')
row_start = 0
row_end = 0
#依照設定的步數處理數據
row_end = len(test2_temp)
for j in range(time_step):
    zero_cols = time_step - j - 1
    test = pd.DataFrame([[0] * len(tar_cols)]*zero_cols,columns = tar_cols)
    if zero_cols == 0:
        test = test.append(test2_temp[tar_cols][:])
    else:
        test = test.append(test2_temp[tar_cols][:-zero_cols])
    test2_data[row_start:row_end,:,j] = test

tar_cols = ['O']
test2_y = np.zeros((n_data,1),dtype = 'float32')
row_start = 0
row_end = len(test2_temp)
test2_y[row_start:row_end,:] = test2_temp[tar_cols]

#%%
#建立數據集張量與定義網路
train_data = torch.from_numpy(data)
train_y = torch.from_numpy(data_y)
torch_dataset = Data.TensorDataset(train_data,  train_y)


test2_data = torch.from_numpy(test2_data)
test2_y = torch.from_numpy(test2_y)

train_loader = Data.DataLoader(
    dataset=torch_dataset,     
    batch_size=1024,      
    shuffle=True,               
    num_workers=0,             
)
#%%
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        #input_size = 特徵量
        #hidden_size = 節點數
        #num_layers = 隱藏層數量
        self.rnn = nn.LSTM(input_size = 12, hidden_size = 16,num_layers = 1,batch_first  = False, bidirectional  = True)
        
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(16, 16)
        
        self.bn2 = nn.BatchNorm1d(16)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(16, 16)
        
        self.bn3 = nn.BatchNorm1d(16)
        self.relu3 = nn.ReLU()
        self.fc3 = nn.Linear(16, 16)
        
        self.bn4 = nn.BatchNorm1d(16)
        self.relu4 = nn.ReLU()
        self.fc4 = nn.Linear(16, 16)
        
        self.bn5 = nn.BatchNorm1d(16)
        self.relu5 = nn.ReLU()
        
        self.fc5 = nn.Linear(16, 1)
    def forward(self, x):
        outputs, (ht, ct) = self.rnn(x)
        
        x = ht[-1]
        residual = x
        
        x = self.fc1(self.relu1(self.bn1(x)))
        x = self.fc2(self.relu2(self.bn2(x)))
        x += residual
        
        residual = x
        x = self.fc3(self.relu3(self.bn3(x)))
        x = self.fc4(self.relu4(self.bn4(x)))
        x += residual
        
        x = x = self.fc5(self.relu5(self.bn5(x)))
        return x

#%%
#訓練
net = Net()
epoch_iter = 1000
loss_fn = torch.nn.MSELoss(reduction='mean')
learning_rate = 1e-4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
optimizer = torch.optim.Adam(net.parameters() ,lr=learning_rate,weight_decay = 1)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=list(range(epoch_iter,epoch_iter,12)), gamma=0.1)

pths_path = './model/'

for epoch in range(epoch_iter):
    net.train()
    epoch_loss = 0
    val_loss = 0
    for i, (x,y) in enumerate(train_loader):
        x,y = x.to(device), y.to(device)
        x = x.permute(2,0,1)
        pred_y = net(x)
        loss1 = loss_fn(y[:,0], pred_y[:,0])
        loss = loss1
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    net.eval()
    train_loss = epoch_loss/((i+1)*2)
    print('Epoch is {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))
state_dict = net.state_dict()
torch.save(state_dict,os.path.join(pths_path, 'model_step'+str(step)+'.pth'))

#%%
#測試
net = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
loss_fn = torch.nn.MSELoss(reduction='mean')

pths_path = './model/'
net.load_state_dict(torch.load(os.path.join(pths_path, 'model_step'+str(step)+'.pth')))
net.eval()


x = test2_data.to(device)
x = x.permute(2,0,1)
pred_y = net(x)
pred_y = pred_y.to('cpu')

loss_fn(pred_y,test2_y)
loss_fn(pred_y[:,0],test2_y[:,0])

pred_y = pred_y.detach().numpy()

mae = mean_absolute_error(test2_y, pred_y)
print('mae=',mae)#8.607
resultPath = './pred_result/'
np.savetxt(resultPath+'testdata_Step'+str(step)+'.csv', pred_y, delimiter=',')
