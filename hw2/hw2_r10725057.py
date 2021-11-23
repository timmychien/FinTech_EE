#!pip3 install skorch
#!pip3 install torch
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.utils.data as Data
import seaborn as sns
import matplotlib.pyplot as plt
from skorch import NeuralNetBinaryClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report,roc_curve, auc,precision_recall_curve,average_precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
def main():
    #i
    data=pd.read_csv('./Data.csv')
    y=data['Class']
    X=data.drop(['Class'],axis=1)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
    scaler=StandardScaler().fit(X_train)
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.fit_transform(X_test)
    def FindLayerNodesLinear(n_layers, first_layer_nodes, last_layer_nodes):
        layers = []
        nodes_increment = (last_layer_nodes - first_layer_nodes)/ (n_layers-1)
        nodes = first_layer_nodes
        for i in range(1, n_layers+1):
            layers.append(math.ceil(nodes))
            nodes = nodes + nodes_increment
        return layers
    first,second,third=FindLayerNodesLinear(3,30,10)
    class Model(nn.Module):
      def __init__(self):
        super(Model,self).__init__()
        self.layers=nn.Sequential(
        nn.Linear(first,second),
        nn.ReLU(),
        nn.Linear(second,third),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(third,1),
        nn.Sigmoid()
        )
      def forward(self,x):
        pred=self.layers(x)
        return pred
    model=Model()
    X_train=torch.from_numpy(X_train.astype(np.float32))
    y_train=torch.from_numpy(y_train.values.astype(np.float32))
    X_test=torch.from_numpy(X_test.astype(np.float32))
    y_test=torch.from_numpy(y_test.values.astype(np.float32))
    net = NeuralNetBinaryClassifier(
          model,
          max_epochs=50,
         )
    params= {
    'lr': [5e-4,0.001,0.005, 0.01,0.02],
    'max_epochs': list(range(20,30,50)),
    'batch_size':list(range(32,64))
    }
    grid = GridSearchCV(net, params, refit=True,scoring='accuracy',n_jobs=-1, cv=5)
    grid_result=grid.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    batch_size=grid_result.best_params_['batch_size']
    learning_rate=grid_result.best_params_['lr']
    epoch_size=grid_result.best_params_['max_epochs']
    y_train=torch.reshape(y_train,(len(y_train),1))
    y_test=torch.reshape(y_test,(len(y_test),1))
    train_dataset=Data.TensorDataset(X_train,y_train)
    valid_dataset=Data.TensorDataset(X_test,y_test)
    trainloader=Data.DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
    );
    validloader=Data.DataLoader(
        dataset=valid_dataset,
        batch_size=32,
        shuffle=True,
    );
    criterion=nn.BCELoss(size_average=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    train_acc=[]
    val_acc=[]
    trainloss=0
    valloss=0
    train_loss=[]
    val_loss=[]
    for epoch in range(epoch_size):
      epoch+=1
      for train_data in trainloader:
        train,target=train_data
        y_pred=model.forward(train)
        loss=criterion(y_pred,target)
        loss.backward()
        trainloss+=loss
        optimizer.step()
        optimizer.zero_grad()
      with torch.no_grad():
        for valid_data in validloader:
          val,val_target=valid_data
          y_eval=model.forward(val)
          loss_val=criterion(y_eval,val_target)
          valloss+=loss_val
      train_loss.append((trainloss/batch_size).detach().numpy())
      val_loss.append((valloss/batch_size).detach().numpy())
      y_train_pred=model(X_train)
      y_train_pred_class=y_train_pred.round()
      train_accuracy=(y_train_pred_class.eq(y_train).sum())/float(y_train.shape[0])
      train_acc.append(train_accuracy)
      y_pred=model(X_test)
      y_pred_class=y_pred.round()
      valid_accuracy=(y_pred_class.eq(y_test).sum())/float(y_test.shape[0])
      val_acc.append(valid_accuracy)
      print(f'epoch:{epoch} train_loss:{trainloss/batch_size} val_loss:{valloss/batch_size} train_acc:{train_accuracy.item()} val_acc:{valid_accuracy.item()}')
      trainloss=0
      valloss=0
    
    plt.style.use("ggplot")
    epoch=[i for i in range(epoch_size)]
    fig=plt.figure(figsize=(20,10),linewidth=2)
    axis1 = fig.add_subplot(1, 2, 1)
    axis2 = fig.add_subplot(1, 2, 2)
    axis1.plot(epoch,train_loss,'-',color='blue',label="train_loss",linewidth=3)
    axis1.plot(epoch,val_loss,'-',color='#66FFE6',label="valid_loss",linewidth=3)
    axis1.legend(loc="upper right",shadow=True)
    axis1.set_xlabel("Epoch",fontsize=15,labelpad=15)
    axis1.set_ylabel("loss",fontsize=15,labelpad=20)
    axis1.set_title('Epoch Loss')
    axis2.plot(epoch,train_acc,'-',color='blue',label="train_accuracy",linewidth=3)
    axis2.plot(epoch,val_acc,'-',color='#66FFE6',label="valid_accuracy",linewidth=3)
    axis2.legend(loc="upper right",shadow=True)
    axis2.set_xlabel("Epoch",fontsize=15,labelpad=15)
    axis2.set_ylabel("Accuracy",fontsize=15,labelpad=20)
    axis2.set_title('Epoch Accuracy')
    plt.show()
#ii 
    y_train_pred=model(X_train)
    y_train_pred_class=y_train_pred.round()
    train_accuracy=(y_train_pred_class.eq(y_train).sum())/float(y_train.shape[0])
    print("Train Accuracy:",train_accuracy.item())
    with torch.no_grad():
        y_pred=model(X_test)
        y_pred_class=y_pred.round()
        valid_accuracy=(y_pred_class.eq(y_test).sum())/float(y_test.shape[0])
    print("valid Accuracy:",valid_accuracy.item())
    train_confusion=confusion_matrix(y_train.detach().numpy(),y_train_pred_class.detach().numpy())
    tc=sns.heatmap(train_confusion, annot=True,fmt='g', cmap='Blues',cbar=False)
    tc.set_title('train_confusion_matrix');
    tc.xaxis.set_ticklabels(['predict=0','predict=1'])
    tc.yaxis.set_ticklabels(['True=1','True=0'])
    plt.show()
    valid_confusion=confusion_matrix(y_test.detach().numpy(),y_pred_class.detach().numpy())
    vc=sns.heatmap(valid_confusion, annot=True,fmt='g', cmap='Blues',cbar=False)
    vc.set_title('validation_confusion_matrix');
    vc.xaxis.set_ticklabels(['predict=0','predict=1'])
    vc.yaxis.set_ticklabels(['True=1','True=0'])
    plt.show()
#iii
    target_names = ['class 0', 'class 1']
    train_eval=classification_report(y_train.detach().numpy(),y_train_pred_class.detach().numpy(),target_names=target_names)
    print('Train Performance:')
    print(train_eval)
    val_eval=classification_report(y_test.detach().numpy(),y_pred_class.detach().numpy(),target_names=target_names)
    print('Valid Performance:')
    print(val_eval)
#v
    y2=data['Class']
    X2=data.drop(['Class'],axis=1)
    X_train2,X_test2,y_train2,y_test2=train_test_split(X2,y2,test_size=0.2,random_state=0)    
    #Decision Tree
    decisiontree=DecisionTreeClassifier()
    decisiontree.fit(X_train2,y_train2)
    dt_pred=decisiontree.predict(X_test2)
    dt_eval=classification_report(y_test2,dt_pred,target_names=target_names)
    print('DecisionTree Performance:')
    print(dt_eval)
    #Random Forest
    randomforest=RandomForestClassifier()
    randomforest.fit(X_train2,y_train2)
    rf_pred=randomforest.predict(X_test2)
    rf_eval=classification_report(y_test2,rf_pred,target_names=target_names)
    print('Random Forest Performance:')
    print(rf_eval)
#vi
    #DNN
    fpr_dnn,tpr_dnn,threshold_dnn = roc_curve(y_test, y_pred)
    precision_dnn, recall_dnn, thresholds_dnn = precision_recall_curve(y_test,y_pred)
    auc_score_dnn=auc(fpr_dnn, tpr_dnn)
    ap_dnn=average_precision_score(y_test, y_pred)
    fig_dnn=plt.figure(figsize=(20,10),linewidth=2)
    axis3 = fig_dnn.add_subplot(1, 2, 1)
    axis4 = fig_dnn.add_subplot(1, 2, 2)
    axis3.plot(fpr_dnn, tpr_dnn, color='blue', label=f'AUC={round(auc_score_dnn,2)}')
    axis3.plot([0, 1], [0, 1], color='red', linestyle='--')
    axis3.set_xlabel('False Positive Rate')
    axis3.set_ylabel('True Positive Rate')
    axis3.set_title('Receiver Operating Characteristic')
    axis3.legend(loc="lower right",shadow=True)
    axis4.plot(precision_dnn,recall_dnn,color='blue')
    axis4.fill(precision_dnn,recall_dnn,color='#66FFE6')
    axis4.set_xlabel('Recall')
    axis4.set_ylabel('Precision')
    axis4.set_title(f'Precision-Recall Curve (AP={round(ap_dnn,2)})')
    plt.fill_between(precision_dnn,recall_dnn,color='#66FFE6')
    plt.show()
    #Decision Tree
    fpr_dt,tpr_dt,threshold_t = roc_curve(y_test2,dt_pred)
    precision_dt, recall_dt, thresholds_dt = precision_recall_curve(y_test2,dt_pred)
    auc_score_dt=auc(fpr_dt,tpr_dt)
    ap_dt=average_precision_score(y_test2, dt_pred)
    fig_dt=plt.figure(figsize=(20,10),linewidth=2)
    axis5 = fig_dt.add_subplot(1, 2, 1)
    axis6 = fig_dt.add_subplot(1, 2, 2)
    axis5.plot(fpr_dt, tpr_dt, color='blue', label=f'AUC={round(auc_score_dt,2)}')
    axis5.plot([0, 1], [0, 1], color='red', linestyle='--')
    axis5.set_xlabel('False Positive Rate')
    axis5.set_ylabel('True Positive Rate')
    axis5.set_title('Receiver Operating Characteristic')
    axis5.legend(loc="lower right",shadow=True)
    axis6.plot(precision_dt,recall_dt,color='blue')
    axis6.fill(precision_dt,recall_dt,color='#66FFE6')
    axis6.set_xlabel('Recall')
    axis6.set_ylabel('Precision')
    axis6.set_title(f'Precision-Recall Curve (AP={round(ap_dt,2)})')
    plt.fill_between(precision_dt, recall_dt,color='#66FFE6')
    plt.show()
    #Random Forest
    fpr_rf,tpr_rf,threshold_rf = roc_curve(y_test2,rf_pred)
    precision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_test2,rf_pred)
    auc_score_rf=auc(fpr_rf,tpr_rf)
    ap_rf=average_precision_score(y_test2, rf_pred)
    fig_rf=plt.figure(figsize=(20,10),linewidth=2)
    axis7 = fig_rf.add_subplot(1, 2, 1)
    axis8 = fig_rf.add_subplot(1, 2, 2)
    axis7.plot(fpr_rf, tpr_rf, color='blue', label=f'AUC={round(auc_score_rf,2)}')
    axis7.plot([0, 1], [0, 1], color='red', linestyle='--')
    axis7.set_xlabel('False Positive Rate')
    axis7.set_ylabel('True Positive Rate')
    axis7.set_title('Receiver Operating Characteristic')
    axis7.legend(loc="lower right",shadow=True)
    axis8.plot(precision_rf,recall_rf,color='blue')
    axis8.fill(precision_rf,recall_rf,color='#66FFE6')
    axis8.set_xlabel('Recall')
    axis8.set_ylabel('Precision')
    axis8.set_title(f'Precision-Recall Curve (AP={round(ap_rf,2)})')
    plt.fill_between(precision_rf, recall_rf,color='#66FFE6')
    plt.show()
if __name__=='__main__':
    main()