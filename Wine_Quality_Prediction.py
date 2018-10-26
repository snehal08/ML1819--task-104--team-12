#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv('winequality.csv')
df = df.drop(['good','color'],axis = 1)
small_df = df[:int(len(df.index)/3)]
med_df = df[:int(2 * len(df.index)/3)]


# In[4]:


def model(data):
    bins = [1,4,6,10]
    #0 for low quality, 1 for average, 2 for great quality
    quality_labels=[0,1,2]
    data['quality_categorical'] = pd.cut(data['quality'], bins=bins, labels=quality_labels, include_lowest=True)
    #Displays the first 2 columns
    #display(data.head(n=2))
    # Split the data into features and target label
    quality_raw = data['quality_categorical']
    features_raw = data.drop(['quality', 'quality_categorical'], axis = 1)
    
    X_train, X_test, y_train, y_test = train_test_split(features_raw,quality_raw,test_size = 0.3,random_state = 0)
    
    clf_A = GaussianNB()
    clf_B = DecisionTreeClassifier(max_depth=None, random_state=None)
    clf_C = RandomForestClassifier(max_depth=None, random_state=None)
    
    gauss = clf_A.fit(X_train,y_train)
    predict_gauss = clf_A.predict(X_test)
    
    descision = clf_B.fit(X_train,y_train)
    predict_decision = clf_B.predict(X_test)
    
    rand = clf_C.fit(X_train,y_train)
    predict_rand = clf_C.predict(X_test)
    result = pd.DataFrame()

    acc_gauss = float(accuracy_score(y_test, predict_gauss))
    acc_decision = accuracy_score(y_test, predict_decision)
    acc_rand = accuracy_score(y_test, predict_rand)
    
    fs_gauss = fbeta_score(y_test, predict_gauss,beta = 0.5,average = 'weighted')
    fs_decision = fbeta_score(y_test, predict_decision,beta = 0.5,average = 'weighted')
    fs_rand = fbeta_score(y_test, predict_rand,beta = 0.5,average = 'weighted')
    
    acc_result = [acc_gauss,acc_decision,acc_rand]
    f_result = [fs_gauss,fs_decision,fs_rand]
    return acc_result,f_result


# In[5]:


small_acc,small_fs = model(small_df)
med_acc,med_fs = model(med_df)
df_acc,df_fs = model(df)


# In[6]:


df_plot_acc = pd.DataFrame({'x': [len(small_df.index),len(med_df.index),len(df.index)],'gaussian':small_acc,'decision':med_acc,'random':df_acc})
df_plot_fs = pd.DataFrame({'x': [len(small_df.index),len(med_df.index),len(df.index)],'gaussian':small_fs,'decision':med_fs,'random':df_fs})
print("Accuracy Matrix: \n ")
print(df_plot_acc)
print("\n F1 Score Matrix: \n")
print(df_plot_fs)

# In[7]:


plt.plot( 'x', 'gaussian', data=df_plot_acc, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.plot( 'x', 'decision', data=df_plot_acc, marker='', color='olive', linewidth=2)
plt.plot( 'x', 'random', data=df_plot_acc, marker='', color='olive', linewidth=2, linestyle='dashed')
plt.legend()
plt.show()


# In[8]:


plt.plot( 'x', 'gaussian', data=df_plot_fs, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.plot( 'x', 'decision', data=df_plot_fs, marker='', color='olive', linewidth=2)
plt.plot( 'x', 'random', data=df_plot_fs, marker='', color='olive', linewidth=2, linestyle='dashed')
plt.legend()
plt.show()