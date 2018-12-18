#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder#Now let's import encoder from sklearn library
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("BlackFriday.csv")
data.head()


# In[2]:


def model(subdata):
    
    b = ['Product_Category_2','Product_Category_3'] 
    for i in b:
        exec("data.%s.fillna(data.%s.value_counts().idxmax(), inplace=True)" %(i,i))
        
    X = data.drop(["Purchase","User_ID","Product_ID"], axis=1)
    LE = LabelEncoder()
    X = X.apply(LE.fit_transform)#Here we applied encoder onto data
    X.Gender = pd.to_numeric(X.Gender)
    X.Age = pd.to_numeric(X.Age)
    X.Occupation = pd.to_numeric(X.Occupation)
    X.City_Category = pd.to_numeric(X.City_Category)
    X.Stay_In_Current_City_Years = pd.to_numeric(X.Stay_In_Current_City_Years)
    X.Marital_Status = pd.to_numeric(X.Marital_Status)
    X.Product_Category_1 = pd.to_numeric(X.Product_Category_1)
    X.Product_Category_2 = pd.to_numeric(X.Product_Category_2)
    X.Product_Category_3 = pd.to_numeric(X.Product_Category_3)
    
    Y = data["Purchase"]#Here we will made a array named as Y consisting of data from purchase column
    
    SS = StandardScaler()
    Xs = SS.fit_transform(X)
    #You must to transform X into numeric representation (not necessary binary).Because all machine learning methods operate on matrices of number
    
    pc = PCA(4)#here 4 indicates the number of components you want it into.
    
    principalComponents = pc.fit_transform(X)#Here we are applying PCA to data/fitting data to PCA
    
    principalDf = pd.DataFrame(data = principalComponents, columns = ["component 1", "component 2", "component 3", "component 4"])
    
    kf = KFold(10)
    #Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default).
    #Each fold is then used once as a validation while the k - 1 remaining folds form the training set.
    
    for a,b in kf.split(principalDf):
        X_train, X_test = Xs[a],Xs[b]
        y_train, y_test = Y[a],Y[b]
    
    rfr = RandomForestRegressor()
    
    fit1 = rfr.fit(X_train,y_train)#Here we fit training data to RandomForest Regressor
    ypred_1 = rfr.predict(X_test)
    
    print("Accuracy Score of RandomForest on train set",fit1.score(X_train,y_train)*100)
    print("Accuracy Score of RandomForest on test set",fit1.score(X_test,y_test)*100)
    print("RMSE of RandomFores on test set", mean_squared_error(y_test, ypred_1)**0.5)
    print("MAE of RandomFores on test set", mean_absolute_error(y_test, ypred_1))


# In[3]:


df_1 = data[:int((len(data.index) / 5)/10)]  # 10k (approx)
df_2 = data[:int(3 * ((len(data.index) / 5)/10))]  # 20k (approx)
df_3 = data[:int(10 * ((len(data.index) / 5)/10))]  # 30k (approx)
df_4 = data[:int(16 * ((len(data.index) / 5)/10))]  # 40k (approx)
df_5 = data  # 50k (approx)
range_list=[df_1,df_2,df_3,df_4,df_5]


# In[4]:


for x in range_list:
     model(x)

