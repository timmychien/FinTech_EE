import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision.models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from keras.datasets import fashion_mnist
def main():
    #load data
    print("[INFO] loading Fashion MNIST...")
    ((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
    trainX=trainX/255.0
    testX=testX/255.0
#1 CNN
    train_X,val_X,train_y,val_y=train_test_split(trainX,trainY,test_size=0.2,random_state=0)
    class CNN(nn.Module):
      def __init__(self):
        super(CNN, self).__init__()
        #input shape(1,28,28)
        self.conv1=nn.Conv2d(1,16,3,stride=1,padding=2)
        self.batchnorm1=nn.BatchNorm2d(16)
        self.relu1=nn.ReLU()
        self.drop1=nn.Dropout2d(p=0.25),
        self.Maxpool1=nn.MaxPool2d(2,2)
        #conv2
        self.conv2=nn.Conv2d(16,32,3,stride=1,padding=2) #(40,16,16)
        self.batchnorm2=nn.BatchNorm2d(32)
        self.relu2=nn.ReLU()
        self.drop2=nn.Dropout2d(p=0.25),
        self.Maxpool2=nn.MaxPool2d(2,2)
        #conv3
        self.conv3=nn.Conv2d(32,64,3,stride=1,padding=2)
        self.batchnorm3=nn.BatchNorm2d(64)
        self.relu3=nn.ReLU()
        self.drop3=nn.Dropout2d(p=0.25),
        self.Maxpool3=nn.MaxPool2d(2,2)#(20,5,5)
        self.fc=nn.Sequential(
            nn.Linear(64*5*5,40),
            nn.Dropout2d(p=0.3),
            nn.ReLU(),
            nn.Linear(40,10)
        )
      def forward(self,x):
        #conv1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = F.dropout(x,p=0.25)
        x = self.Maxpool1(x)
        #conv2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = F.dropout(x,p=0.25)
        x = self.Maxpool2(x)
        #conv3
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)
        x = F.dropout(x,p=0.25)
        x = self.Maxpool3(x)
        x = x.view(x.size(0),-1)
        output = self.fc(x)
        return output
    model=CNN()
    model=model.to(device)
    train_X=train_X.reshape(len(train_X),1,28,28)
    val_X =val_X.reshape(len(val_X),1,28,28)
    test_X=testX.reshape(len(testX),1,28,28)
    train_X=torch.from_numpy(train_X.astype(np.float32))
    val_X=torch.from_numpy(val_X.astype(np.float32))
    test_X=torch.from_numpy(test_X.astype(np.float32))
    train_y=torch.from_numpy(train_y.astype(np.float32)).long()
    val_y=torch.from_numpy(val_y.astype(np.float32)).long()
    test_y=torch.from_numpy(testY.astype(np.float32)).long()
    train_dataset=Data.TensorDataset(train_X,train_y)
    test_dataset=Data.TensorDataset(test_X,test_y)
    trainloader=Data.DataLoader(
      dataset=train_dataset,
      batch_size=128,
      shuffle=True,
      num_workers=2  
    );
    testloader=Data.DataLoader(
      dataset=test_dataset,
      batch_size=128,
      shuffle=False,
      num_workers=2  
    );
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    ## CNN traininig
    trainloss=0
    testloss=0
    train_correct=0
    test_correct=0
    epochs=25
    batch_size=128
    train_loss=[]
    test_loss=[]
    train_acc=[]
    test_acc=[]
    for epoch in range(epochs):
      epoch+=1
      for train,target in trainloader:
        train=train.to(device)
        target=target.to(device)
        y_pred=model.forward(train)
        train_pred =y_pred.argmax(dim=1, keepdim=True)
        train_correct += train_pred.eq(target.view_as(train_pred)).sum().item()
        loss=criterion(y_pred,target)
        trainloss+=loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      with torch.no_grad():
        for test,test_target in testloader:
          test=test.to(device)
          test_target=test_target.to(device)
          y_eval=model.forward(test)
          loss_test=criterion(y_eval,test_target)
          testloss+=loss_test
          test_eval = y_eval.argmax(dim=1, keepdim=True)
          test_correct += test_eval.eq(test_target.view_as(test_eval)).sum().item()
      train_loss.append(trainloss/(len(train_dataset)/batch_size))
      test_loss.append(testloss/(len(test_dataset)/batch_size))
      train_acc.append(train_correct/len(train_dataset))
      test_acc.append(test_correct/len(test_dataset))
      print(f"epoch:{epoch} train_loss:{trainloss/(len(train_dataset)/batch_size)} test_loss:{testloss/(len(test_dataset)/batch_size)} train_acc:{train_correct/len(train_dataset)} test_acc:{test_correct/len(test_dataset)}")
      trainloss=0
      testloss=0
      train_correct=0
      test_correct=0
#2.Plot the learning and the accuracy curve
    epoch_=[i for i in range(25)]
    fig=plt.figure(figsize=(30,10),linewidth=2)
    axis1 = fig.add_subplot(1, 2, 1)
    axis2 = fig.add_subplot(1, 2, 2)
    axis1.plot(epoch_,train_loss, label='train_loss')
    #plt.plot(epoch_,valid_loss,label='valid_loss')
    axis1.plot(epoch_,test_loss,label='test_loss')
    axis1.set_xlabel("Epoch")
    axis1.set_ylabel("loss")
    axis1.set_title('Learning Curve')
    axis1.legend(loc="upper right",shadow=True)
    axis2.plot(epoch_,train_acc, label='train_acc')
    axis2.plot(epoch_,test_acc,label='test_acc')
    axis2.set_xlabel("Epoch")
    axis2.set_ylabel("Accuracy")
    axis2.set_title('Accuracy')
    axis2.legend(loc="upper right",shadow=True)
    plt.show()
#3.plot activations of the first layer
    plt.imshow(train_X[1,0,:,:].cpu().detach().numpy(),cmap=plt.cm.binary)
    conv1=model.conv1(train_X)
    axes=[]
    fig=plt.figure(figsize=(12,12))
    for i in range(4*4):
      axes.append(fig.add_subplot(4,4,i+1))
      plt.imshow(conv1[1,i,:,:].cpu().detach().numpy(),cmap=plt.cm.binary)
    fig.tight_layout()    
    plt.show()
#4.classify the clothing and plot the corresponding image and label
    labels=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
    test_X=test_X.to(device)
    out=model.forward(test_X)
    out_eval = out.argmax(dim=1, keepdim=True)
    out_eval=[out.item() for out in out_eval]
    axes_pred=[]
    fig_pred=plt.figure(figsize=(8,8))
    for i in range(16):
       axes_pred.append(fig_pred.add_subplot(4,4,i+1))
       if out_eval[i]==testY[i]:
           plt.imshow(test_X[i,0,:,:].cpu().detach().numpy(),cmap=plt.cm.binary)
           plt.text(5, 5, labels[out_eval[i]], fontsize=15, color='green')
       else:
           plt.imshow(test_X[i,0,:,:].cpu().detach().numpy(),cmap=plt.cm.binary)
           plt.text(5, 5, labels[out_eval[i]], fontsize=15, color='red')
    fig_pred.tight_layout()    
    plt.show()
    resnet18 = models.resnet18(pretrained=True)
    resnet18.fc = nn.Linear(512, 10)
    resnet18.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resnet18=resnet18.to(device)
    optimizer_resnet=torch.optim.Adam(resnet18.parameters(),lr=1e-3,weight_decay=1e-5)
    #resnet traininig
    trainloss_resnet=0
    testloss_resnet=0
    train_correct_resnet=0
    test_correct_resnet=0
    train_loss_resnet=[]
    test_loss_resnet=[]
    train_acc_resnet=[]
    test_acc_resnet=[]
    batch_size=128
    epochs=25
    for epoch in range(epochs):
      epoch+=1
      for inputs, targets in trainloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        pred = resnet18(inputs)
        train_pred_resnet=pred.argmax(dim=1, keepdim=True)
        train_correct_resnet += train_pred_resnet.eq(targets.view_as(train_pred_resnet)).sum().item()
        loss_resnet = criterion(pred, targets)
        trainloss_resnet+=loss_resnet
        optimizer_resnet.zero_grad()
        loss_resnet.backward()
        optimizer_resnet.step()
      with torch.no_grad():
          for test_resnet,test_target_resnet in testloader:
            test_resnet=test_resnet.to(device)
            test_target_resnet=test_target_resnet.to(device)
            y_eval_resnet=resnet18(test_resnet)
            loss_test_resnet=criterion(y_eval_resnet,test_target_resnet)
            testloss_resnet+=loss_test_resnet
            test_eval_resnet = y_eval_resnet.argmax(dim=1, keepdim=True)
            test_correct_resnet += test_eval_resnet.eq(test_target_resnet.view_as(test_eval_resnet)).sum().item()
      train_loss_resnet.append(trainloss_resnet/(len(train_dataset)/batch_size))
      test_loss_resnet.append(testloss_resnet/(len(test_dataset)/batch_size))
      train_acc_resnet.append(train_correct_resnet/len(train_dataset))
      test_acc_resnet.append(test_correct_resnet/len(test_dataset))
      print(f"epoch:{epoch} train_loss:{trainloss_resnet/(len(train_dataset)/batch_size)} test_loss:{testloss_resnet/(len(test_dataset)/batch_size)} train_acc:{train_correct_resnet/len(train_dataset)} test_acc:{test_correct_resnet/len(test_dataset)}")
      trainloss_resnet=0
      testloss_resnet=0
      train_correct_resnet=0
      test_correct_resnet=0
      #plot learning curve and accuracy curve
      fig=plt.figure(figsize=(30,10),linewidth=2)
      axis3 = fig.add_subplot(1, 2, 1)
      axis4 = fig.add_subplot(1, 2, 2)
      axis3.plot(epoch_,train_loss_resnet, label='train_loss')
      axis3.plot(epoch_,test_loss_resnet,label='test_loss')
      axis3.set_xlabel("Epoch")
      axis3.set_ylabel("loss")
      axis3.set_title('Learning Curve')
      axis3.legend(loc="upper right",shadow=True)
      axis4.plot(epoch_,train_acc_resnet, label='train_acc')
      axis4.plot(epoch_,test_acc_resnet,label='test_acc')
      axis4.set_xlabel("Epoch")
      axis4.set_ylabel("Accuracy")
      axis4.set_title('Accuracy')
      axis4.legend(loc="upper right",shadow=True)
      plt.show()
      # plot activations of the first layer
      plt.imshow(train_X[1,0,:,:].cpu().detach().numpy(),cmap=plt.cm.binary)
      conv1_resnet=resnet18.conv1(train_X)
      axes_resnet=[]
      fig_resnet=plt.figure(figsize=(8,8))
      for i in range(4*4):
        axes_resnet.append(fig.add_subplot(4,4,i+1))
        plt.imshow(conv1_resnet[1,i,:,:].cpu().detach().numpy(),cmap=plt.cm.binary)
      fig_resnet.tight_layout()    
      plt.show()
      # classify the clothing and plot the corresponding image and label
      out_resnet=resnet18(test_X)
      out_eval_resnet = out_resnet.argmax(dim=1, keepdim=True)
      out_eval_resnet=[out.item() for out in out_eval_resnet]
      axes_pred_resnet=[]
      fig_pred_resnet=plt.figure(figsize=(8,8))
      for i in range(16):
          axes_pred_resnet.append(fig_pred_resnet.add_subplot(4,4,i+1))
          if out_eval_resnet[i]==testY[i]:
            plt.imshow(test_X[i,0,:,:].cpu().detach().numpy(),cmap=plt.cm.binary)
            plt.text(5, 5, labels[out_eval_resnet[i]], fontsize=15, color='green')
          else:
            plt.imshow(test_X[i,0,:,:].cpu().detach().numpy(),cmap=plt.cm.binary)
            plt.text(5, 5, labels[out_eval_resnet[i]], fontsize=15, color='red')
      plt.show()
if __name__=='__main__':
    main()