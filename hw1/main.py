import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from func import one_hot_encoding,train_test_split,rmse,regularization,bayesian
def main():
    #1-a
    data=pd.read_csv('train.csv')
    data=data.drop(['ID','address','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','schoolsup','famsup','nursery','G1','G2','cat'],axis=1)
    data=one_hot_encoding(data)
    target=data['G3']
    data=data.drop(['G3'],axis=1)
    X_train,X_test,Y_train,Y_test=train_test_split(data,target,0.8)
    
    #1-b
    weight=np.linalg.pinv(X_train).dot(Y_train)
    pred_linear=X_test.dot(weight)
    rmse_linear=rmse(pred_linear,Y_test)
    print('(1-b) RMSE=',rmse_linear)
    
    #1-c
    w_no_bias=regularization(X_train,Y_train,1.0,bias=False)
    y_pred_no_bias=X_test.dot(w_no_bias)
    rmse_regu_without_bias=rmse(y_pred_no_bias,Y_test)
    print('(1-c) RMSE=',rmse_regu_without_bias)
    
    #1-d
    w_bias=regularization(X_train,Y_train,1.0,bias=True)
    X_test_bias=np.c_[np.ones((X_test.shape[0], 1)), X_test]
    #X=np.c_[np.random.normal(0,np.ndarray.var(X_test.values),np.shape(X_test)[0]), X_test]
    y_pred=X_test_bias.dot(w_bias)
    rmse_regu_with_bias=rmse(y_pred,Y_test)
    print('(1-d) RMSE=',rmse_regu_with_bias)
    #1-e
    W_bayesian=bayesian(X_train,Y_train,1.0)
    #X_test_bias=np.c_[np.random.normal(0,1,len(X_test)), X_test]
    X_test_bias = np.c_[np.ones((np.shape(X_test)[0], 1)), X_test]
    pred_bayesian=X_test_bias.dot(W_bayesian)
    rmse_bayesian=rmse(pred_bayesian,Y_test)
    print('(1-e) RMSE=',rmse_bayesian)
    
    #1-f
    plt.style.use("ggplot")               
    sample_index=[i for i in range(len(Y_test))]
    plt.figure(figsize=(15,10),linewidth=2)
    plt.plot(sample_index,Y_test,'-',color='blue',label="ground truth",linewidth=3)
    plt.plot(sample_index,pred_linear,'-',color='#66FFE6',label=f'({round(rmse_linear,3)})Linear Regression',linewidth=3)
    plt.plot(sample_index,y_pred_no_bias,'-',color='#8CE600',label=f'({round(rmse_regu_without_bias,3)})Linaer Regression(reg)',linewidth=3)
    plt.plot(sample_index,y_pred,'-',color='#F08080',label=f'({round(rmse_regu_with_bias,3)})Linear Regression(r/b)',linewidth=3)
    plt.plot(sample_index,pred_bayesian,'-',color='#FFFF4D',label=f'({round(rmse_bayesian,3)})Bayesian Linear Regression',linewidth=5)
    plt.legend(loc="upper right",shadow=True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Sample Index",fontsize=30,labelpad=15)
    plt.ylabel("Values",fontsize=30,labelpad=20)
    plt.show()
    #1-g
    test_data=pd.read_csv('test_no_G3.csv')
    test_data=test_data.drop(['ID','address','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','schoolsup','famsup','nursery','G1','G2','cat'],axis=1)
    test_data=one_hot_encoding(test_data)
    for col in test_data.iloc[:0]:
        test_data[col]=(test_data[col]-np.mean(test_data[col]))/np.std(test_data[col])
    test_data_bias= np.c_[np.ones((np.shape(test_data)[0], 1)), test_data]
    pred_bayesian_test=test_data_bias.dot(W_bayesian)
    output=open('r10725057_1.txt','w')
    for i in range(len(test_data)):
        output.write(f'{i}   {pred_bayesian_test[i]} \n')
    output.close()


    #2-a
    col_list=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary']
    data_2=pd.read_csv('adult.data',names=col_list)
    data_2['salary']=data_2['salary'].astype('category')
    data_2['salary']=data_2['salary'].cat.codes
    target_2=data_2['salary']
    data_2=data_2.drop(['salary'],axis=1)
    for col in data_2.iloc[:0]:
        if data_2[col].dtype=="object":
            data_2[col]=data_2[col].astype('category').cat.codes
    X_train2,X_test2,Y_train2,Y_test2=train_test_split(data_2,target_2,0.8)
    #2-b
    weight2=np.linalg.pinv(X_train2).dot(Y_train2)
    pred_linear2=X_test2.dot(weight2)
    rmse_linear2=rmse(pred_linear2,Y_test2)
    print('(2-b) RMSE=',rmse_linear2)
    #2-c regularization without bias
    w_no_bias2=regularization(X_train2,Y_train2,1.0,bias=False)
    y_pred_no_bias2=X_test2.dot(w_no_bias2)
    rmse_regu_without_bias2=rmse(y_pred_no_bias2,Y_test2)
    print('(2-c) RMSE=',rmse_regu_without_bias2)
    #2-d regularization with bias
    w_bias2=regularization(X_train2,Y_train2,1.0,bias=True)
    X_test_bias2=np.c_[np.ones((np.shape(X_test2)[0], 1)), X_test2]
    y_pred2=X_test_bias2.dot(w_bias2)
    rmse_regu_with_bias2=rmse(y_pred2,Y_test2)
    print('(2-d) RMSE=',rmse_regu_with_bias2)
    #2-e
    W_bayesian2=bayesian(X_train2,Y_train2,1.0)
    X_test_bias2=np.c_[np.ones((np.shape(X_test2)[0], 1)), X_test2]
    pred_bayesian2=X_test_bias2.dot(W_bayesian2)
    rmse_bayesian2=rmse(pred_bayesian2,Y_test2)
    print('(2-e) RMSE=',rmse_bayesian2)
    # result of 2
    col_name=[col for col in X_train2.iloc[:0]]
    for i in range(X_train2.shape[1]):
        print(f'column-name:{col_name[i]}  weight:{W_bayesian2[i]}')

    
    test_2=pd.read_csv('adult.test',names=col_list)
    test_2= test_2.drop(labels=0, axis=0)
    test_y=test_2['salary'].astype('category').cat.codes
    test_x=test_2.drop(['salary'],axis=1)
    for col in test_x.iloc[:0]:
        if test_x[col].dtype=="object":
            test_x[col]=test_x[col].astype("category").cat.codes
    for col in test_x.iloc[:0]:
        test_x[col]=(test_x[col]-np.mean(test_x[col]))/np.std(test_x[col])
    test_x_bias2=np.c_[np.ones((np.shape(test_x)[0], 1)), test_x]
    pred_bayesian_test2=test_x_bias2.dot(W_bayesian2)
    for i in range(len(pred_bayesian_test2)):
        if pred_bayesian_test2[i]<0.5:
            pred_bayesian_test2[i]=int(0)
        else:pred_bayesian_test2[i]=int(1)
    pred=pd.Series(pred_bayesian_test2).astype('int8')
    print('RMSE of Test data:',rmse(test_y,pred_bayesian_test2))
    #for i,data in enumerate(test_y):
     #   print(f'{i}  {data} {pred[i]}')
    correct_index=[]
    for i,data in enumerate(test_y):
        if data==pred[i]:
            correct_index.append(i)
    print('Accuracy:',len(correct_index)/len(test_2))
        
if __name__=='__main__':
    main()