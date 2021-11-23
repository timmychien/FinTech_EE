import pandas as pd
import numpy as np

def one_hot_encoding(data):
    data_dum = pd.get_dummies(data)
    data=pd.DataFrame(data_dum)
    return data

def train_test_split(x,y,train_percent):
    train_len=int(len(x)*train_percent)
    num_list=[num for num in range(len(x))]
    train_num=np.random.choice(num_list,train_len)
    X_train=x.iloc[train_num,:]
    Y_train=y[train_num]
    mean=pd.Series();
    std=pd.Series();
    for col in X_train.iloc[:0]:
        mean[col]=np.mean(X_train[col])
        std[col]=np.std(X_train[col])             
    for col in X_train.iloc[:0]:
        X_train[col]=(X_train[col]-mean[col])/std[col]
    test_num=[num for num in range(len(x)) if num not in train_num]
    X_test=x.iloc[test_num,:]
    Y_test=y[test_num]
    for col_test in X_test.iloc[:0]:
        X_test[col_test]=(X_test[col_test]-mean[col_test])/std[col_test]
    return X_train,X_test,Y_train,Y_test

def rmse(y_pred,y):
    return np.sqrt(((y_pred-y)**2).mean())

def regularization(X,y,_lambda,bias):
    if(bias==True):
        X = np.c_[np.ones((np.shape(X)[0], 1)), X]
        #X=np.c_[np.random.normal(0,np.ndarray.var(X.values),np.shape(X)[0]), X]
    I=np.identity(X.shape[1])
    W=np.linalg.inv(X.T.dot(X)+(_lambda*X.shape[0])*I).dot(X.T).dot(y)
        
    #for current_iteration in np.arange(epochs):
     #   y_est=X.dot(w)
      #  err=y_est-y
       # regu_term=(_lambda/2)*w.T.dot(w)
        #j_w=((y_est-y)**2).mean()+regu_term
       # gradient=(1/np.shape(X)[0])*(X.T.dot(err)+(_lambda*w))
       # w=w-lr*gradient
    
    
    return W

def bayesian(X,y,alpha):
    X = np.c_[np.ones((np.shape(X)[0], 1)), X]
    mu_zero=0
    I=np.identity(X.shape[1])
    lambda_m=np.linalg.inv(X.T.dot(X)+np.linalg.inv(alpha*I))
    mu_m=lambda_m.dot(X.T).dot(y)
    #w=np.linalg.inv(I.T.dot(I)+2*I).dot(I.T).dot(mu_m)
    return mu_m