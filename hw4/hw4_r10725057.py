!wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
!tar -xzvf ta-lib-0.4.0-src.tar.gz
%cd ta-lib
!./configure --prefix=/usr
!make
!make install
!pip3 install Ta-Lib
!pip3 install yfinance
!pip3 install mpl_finance
import pandas as pd
import numpy as np
import torch
import talib
import math
import torch.nn as nn
import yfinance as yf
import mpl_finance as mpf
from pandas_datareader import data as pdr
yf.pdr_override()
from datetime import datetime
import matplotlib.pyplot as plt
def main():
  #collect data by yfinance
  sp500=yf.Ticker("^GSPC")
  data=sp500.history(start="1994-01-01",end="2020-12-31")
  data=data.drop(['Dividends','Stock Splits'],axis=1)
  data=data.reset_index()
  #1.Candlestick,KDline,Volume bar
  idxlist_2019=[]
  for i in range(len(data)):  
    if(data['Date'][i]>datetime.strptime('2019-01-01', "%Y-%m-%d").date() and data['Date'][i]<datetime.strptime('2020-01-01', "%Y-%m-%d").date()):    
      idxlist_2019.append(i)
  start_2019=idxlist_2019[0]
  end_2019=idxlist_2019[len(idxlist_2019)-1]
  data_2019=data[start_2019:end_2019+1]
  data_2019=data_2019.set_index(data_2019['Date'])
  data_2019.index = data_2019.index.format(formatter=lambda x: x.strftime('%Y-%m-%d')) 
  MA_10_2019 = talib.SMA(data_2019['Close'], timeperiod=10)
  MA_30_2019= talib.SMA(data_2019['Close'], timeperiod=30)
  data_2019['K'], data_2019['D'] = talib.STOCH(data_2019['High'], data_2019['Low'], data_2019['Close'])
  data_2019['K'].fillna(value=0, inplace=True)
  data_2019['D'].fillna(value=0, inplace=True)
  fig = plt.figure(figsize=(30, 20))
  ax = fig.add_axes([0,0.3,1,0.4])
  ax2 = fig.add_axes([0,0.2,1,0.1])
  ax3 = fig.add_axes([0,0,1,0.2])
  ax.set_xticks(range(0, len(data_2019.index), 10))
  ax.set_xticklabels(data_2019.index[::10])
  mpf.candlestick2_ochl(ax, data_2019['Open'], data_2019['Close'], data_2019['High'],data_2019['Low'], width=0.6, colorup='r', colordown='g', alpha=0.75)
  ax.plot(MA_10_2019,label='MA10')
  ax.plot(MA_30_2019,label='MA30')
  ax2.plot(data_2019['K'],label='K')
  ax2.plot(data_2019['D'],label='D')
  ax2.set_xticks(range(0, len(data_2019.index), 10))
  ax2.set_xticklabels(data_2019.index[::10])
  mpf.volume_overlay(ax3, data_2019['Open'], data_2019['Close'], data_2019['Volume'], colorup='r', colordown='g', width=0.5, alpha=0.8)
  ax3.set_xticks(range(0, len(data_2019.index), 10))
  ax3.set_xticklabels(data_2019.index[::10])
  ax.legend(loc="upper right",shadow=True)
  ax2.legend(loc="upper right",shadow=True)
  plt.show()
  #2.Data Preprocessing
  data['Ma10'] = talib.SMA(data['Close'], timeperiod=10)
  data['Ma30'] = talib.SMA(data['Close'], timeperiod=30)
  data['Close_change']=0
  data['Close_change'][1:len(data)]=[((data['Close'][i]-data['Close'][i-1])*100/data['Close'][i-1])for i in range(1,len(data))]
  data['K'],data['D']=talib.STOCH(data['High'], data['Low'], data['Close'])
  data['RSI']=talib.RSI(data['Close'],timeperiod=14)
  data=data.dropna()
  data=data.reset_index(drop=True)
  idxlist_1819=[]
  idxlist_2020=[]
  for i in range(len(data)):
    if(data['Date'][i]>datetime.strptime('2020-01-01', "%Y-%m-%d").date()):    
      idxlist_2020.append(i)
    elif(data['Date'][i]>datetime.strptime('2018-01-01', "%Y-%m-%d").date() and data['Date'][i]<datetime.strptime('2019-12-31', "%Y-%m-%d").date()):
      idxlist_1819.append(i)
  start_2020=idxlist_2020[0]
  end_2020=idxlist_2020[len(idxlist_2020)-1]
  start_1819=idxlist_1819[0]
  end_1819=idxlist_1819[len(idxlist_1819)-1]
  data_train=data[0:end_1819+1]
  for col in ['Open','High','Low','Ma10','Ma30','Volume','K','D','RSI','Close_change','Close']:
    _max=max(data_train[col])
    _min=min(data_train[col])
    for i in range(len(data)):
      data[col][i]=(data[col][i]-_min)/(_max-_min)
  data_y=data['Close']
  training_data=data[0:start_1819-1]
  training_data=training_data.set_index(training_data['Date'])
  training_data.index = training_data.index.format(formatter=lambda x: x.strftime('%Y-%m-%d')) 
  training_x_data=training_data.drop(['Date','Close'],axis=1)
  training_y_data=data_y[0:start_1819-1]
  training_y_data=training_y_data.reset_index(drop=True)
  valid_data=data[start_1819:end_1819+1]
  valid_data=valid_data.set_index(valid_data['Date'])
  valid_data.index =valid_data.index.format(formatter=lambda x: x.strftime('%Y-%m-%d')) 
  valid_x_data=valid_data.drop(['Date','Close'],axis=1)
  valid_y_data=data_y[start_1819:end_1819+1]
  valid_y_data=valid_y_data.reset_index(drop=True)
  testing_data=data[start_2020:end_2020+1]
  testing_data=testing_data.set_index(testing_data['Date'])
  testing_data.index = testing_data.index.format(formatter=lambda x: x.strftime('%Y-%m-%d')) 
  testing_x_data=testing_data.drop(['Date','Close'],axis=1)
  testing_y_data=data_y[start_2020:end_2020+1]
  testing_y_data=testing_y_data.reset_index(drop=True)
  #3.shift data
  def make_data(data, timestep, data_type = 'x'):
      if data_type == 'x':
          return (np.array([data[i:i+timestep] for i in range(data.shape[0]-(timestep))]))
      elif data_type == 'y':
          return (np.array([data[i+timestep:i+timestep+1] for i in range(data.shape[0]-(timestep))]))
  timestep=30
  training_x_shift = make_data(training_x_data, timestep, data_type = 'x')
  training_y_shift = make_data(training_y_data, timestep, data_type = 'y')
  valid_x_shift = make_data(valid_x_data, timestep, data_type = 'x')
  valid_y_shift = make_data(valid_y_data, timestep, data_type = 'y')
  testing_x_shift = make_data(testing_x_data, timestep, data_type = 'x')
  testing_y_shift = make_data(testing_y_data, timestep, data_type = 'y')
  #train 
  training_x=torch.from_numpy(training_x_shift.astype(np.float32))
  valid_x=torch.from_numpy(valid_x_shift.astype(np.float32))
  testing_x=torch.from_numpy(testing_x_shift.astype(np.float32))
  training_y=torch.from_numpy(training_y_shift.astype(np.float32))
  valid_y=torch.from_numpy(valid_y_shift.astype(np.float32))
  testing_y=torch.from_numpy(testing_y_shift.astype(np.float32))
  trainset=torch.utils.data.TensorDataset(training_x, training_y)
  testset=torch.utils.data.TensorDataset(valid_x, valid_y)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False)
  testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
  criterion = nn.MSELoss()
  def train(model,optimizer,trainloader,testloader,epoch_num):
    train_loss_list=[]
    val_loss_list=[]
    train_loss=0
    valid_loss=0
    total_train_loss=0
    total_val_loss=0
    for epoch in range(epoch_num):
      epoch+=1
      for train_set in trainloader:
        train,target=train_set
        pred=model.forward(train)
        loss =criterion(pred,target)
        del(pred)
        train_loss+=loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      with torch.no_grad():
        for valid_set in testloader:
          input,valid_target=valid_set
          val_pred=model.forward(input)
          val_loss=criterion(val_pred,valid_target)
          del(val_pred)
          valid_loss+=val_loss
          del(val_loss)
      total_val_loss=valid_loss/len(testloader)
      total_train_loss=train_loss/len(trainloader)
      train_loss_list.append(total_train_loss)
      val_loss_list.append(total_val_loss)
      print(f'epoch:{epoch} train_loss:{total_train_loss} val_loss:{total_val_loss}')
      total_train_loss=0
      total_val_loss=0
      train_loss=0
      valid_loss=0
    predict=model.forward(valid_x).detach().numpy()
    fig=plt.figure(figsize=(40,10),linewidth=2)
    axa = fig.add_subplot(1, 2, 1)
    axb = fig.add_subplot(1, 2, 2)
    axa.set_title('model loss')
    axa.plot(train_loss_list,label='train')
    axa.plot(val_loss_list,label='valid')
    axa.legend()
    axb.set_xticks(range(0, len(valid_data.index), 30))
    axb.set_xticklabels(valid_data.index[30::30])
    axb.set_title('price prediction')
    axb.plot(valid_y,label='real')
    axb.plot(predict,label='predict')
    axb.legend()
    #4-5.RNN
    class RNN(nn.Module):
        def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
            super(RNN, self).__init__()
            self.hidden_dim = hidden_dim
            self.layer_dim = layer_dim
            self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, nonlinearity='relu',batch_first=True, dropout=0.33)
            self.fc = nn.Linear(hidden_dim, output_dim)
        def forward(self, x):
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
            out, h0 = self.rnn(x, h0.detach())
            out = out[:, -1, :]
            out = self.fc(out)
            return out
    rnn=RNN(10,100,1,1)
    optimizer= torch.optim.Adam(rnn.parameters(),lr=1e-4)
    train(rnn,optimizer,trainloader,testloader,20)
    #6.LSTM
    class LSTM(nn.Module):
      def __init__(self,feature_dim,hidden_dim,num_layers,output_dim):
        super(LSTM, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers =num_layers
        self.output_dim = output_dim
        self.lstm = nn.LSTM(self.feature_dim, self.hidden_dim, self.num_layers, dropout=0.33,batch_first=True)
        self.fc = nn.Linear(self.hidden_dim , self.output_dim)
      def forward(self, x):      
        h0 = torch.zeros([self.num_layers, x.shape[0], self.hidden_dim]).requires_grad_()
        c0 = torch.zeros([self.num_layers, x.shape[0], self.hidden_dim]).requires_grad_()
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out
    lstm=LSTM(10,100,1,1)
    optimizer_lstm= torch.optim.Adam(lstm.parameters(),lr=1e-4)
    train(lstm,optimizer_lstm,trainloader,testloader,20)
    #7.GRU
    class GRU(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
            super(GRU, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,dropout=0.33)
            self.fc = nn.Linear(hidden_dim, output_dim)
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            out,_= self.gru(x, (h0.detach()))
            out = self.fc(out[:, -1, :])
            return out
    gru=GRU(10,100,1,1)
    optimizer_gru= torch.optim.Adam(gru.parameters(),lr=1e-4)
    train(gru,optimizer_gru,trainloader,testloader,20)
    #8.Test 2020 data
    predict2020_rnn=rnn.forward(testing_x).detach().numpy()
    predict2020_lstm=lstm.forward(testing_x).detach().numpy()
    predict2020_gru=gru.forward(testing_x).detach().numpy()
    fig_2020=plt.figure(figsize=(45,10),linewidth=2)
    ax10 = fig_2020.add_subplot(1, 3, 1)
    ax11= fig_2020.add_subplot(1, 3, 2)
    ax12= fig_2020.add_subplot(1, 3, 3)
    ax10.set_title('RNN')
    ax10.set_xticks(range(0, len(testing_data.index), 30))
    ax10.set_xticklabels(testing_data.index[30::30])
    ax10.plot(testing_y,label='real')
    ax10.plot(predict2020_rnn,label='predict')
    ax10.legend()
    ax11.set_title('LSTM')
    ax11.set_xticks(range(0, len(testing_data.index), 30))
    ax11.set_xticklabels(testing_data.index[30::30])
    ax11.plot(testing_y,label='real')
    ax11.plot(predict2020_lstm,label='predict')
    ax11.legend()
    ax12.set_title('GRU')
    ax12.set_xticks(range(0, len(testing_data.index), 30))
    ax12.set_xticklabels(testing_data.index[30::30])
    ax12.plot(testing_y,label='real')
    ax12.plot(predict2020_gru,label='predict')
    ax12.legend()
    #10.improve model
    training_x_10d = make_data(training_x_data, 10, data_type = 'x')
    training_y_10d = make_data(training_y_data, 10, data_type = 'y')
    valid_x_10d = make_data(valid_x_data, 10, data_type = 'x')
    valid_y_10d = make_data(valid_y_data, 10, data_type = 'y')
    testing_x_10d = make_data(testing_x_data, 10, data_type = 'x')
    testing_y_10d = make_data(testing_y_data, 10, data_type = 'y')
    training_x_10d=torch.from_numpy(training_x_10d.astype(np.float32))
    valid_x_10d=torch.from_numpy(valid_x_10d.astype(np.float32))
    testing_x_10d=torch.from_numpy(testing_x_10d.astype(np.float32))
    training_y_10d=torch.from_numpy(training_y_10d.astype(np.float32))
    valid_y_10d=torch.from_numpy(valid_y_10d.astype(np.float32))
    testing_y_10d=torch.from_numpy(testing_y_10d.astype(np.float32))
    trainset_10d=torch.utils.data.TensorDataset(training_x_10d, training_y_10d)
    testset_10d=torch.utils.data.TensorDataset(valid_x_10d, valid_y_10d)
    trainloader_10d = torch.utils.data.DataLoader(trainset_10d, batch_size=32, shuffle=False)
    testloader_10d = torch.utils.data.DataLoader(testset_10d, batch_size=32, shuffle=False)
    criterion = nn.MSELoss()
    gru_10d=GRU(10,100,1,1)
    optimizer_gru10d= torch.optim.Adam(gru_10d.parameters(),lr=1e-4)
    train_loss_list=[]
    val_loss_list=[]
    train_loss=0
    valid_loss=0
    total_train_loss=0
    total_val_loss=0
    for epoch in range(20):
      epoch+=1
      for train_set in trainloader_10d:
        train,target=train_set
        pred=gru_10d.forward(train)
        loss =criterion(pred,target)
        del(pred)
        train_loss+=loss
        optimizer_gru10d.zero_grad()
        loss.backward()
        optimizer_gru10d.step()
      with torch.no_grad():
        for valid_set in testloader_10d:
          input,valid_target=valid_set
          val_pred=gru_10d.forward(input)
          val_loss=criterion(val_pred,valid_target)
          del(val_pred)
          valid_loss+=val_loss
          del(val_loss)
      total_val_loss=valid_loss/len(testloader_10d)
      total_train_loss=train_loss/len(trainloader_10d)
      train_loss_list.append(total_train_loss)
      val_loss_list.append(total_val_loss)
      print(f'epoch:{epoch} train_loss:{total_train_loss} val_loss:{total_val_loss}')
      total_train_loss=0
      total_val_loss=0
      train_loss=0
      valid_loss=0
    predict=gru_10d.forward(valid_x_10d).detach().numpy()
    fig=plt.figure(figsize=(40,10),linewidth=2)
    axmodel = fig.add_subplot(1, 2, 1)
    axpredict = fig.add_subplot(1, 2, 2)
    axmodel.set_title('model loss')
    axmodel.plot(train_loss_list,label='train')
    axmodel.plot(val_loss_list,label='valid')
    axmodel.legend()
    axpredict.set_xticks(range(0, len(valid_data.index), 60))
    axpredict.set_xticklabels(valid_data.index[10::60])
    axpredict.set_title('price prediction')
    axpredict.plot(valid_y_10d,label='real')
    axpredict.plot(predict,label='predict')
    axpredict.legend()
if __name__=='__main__':
    main()