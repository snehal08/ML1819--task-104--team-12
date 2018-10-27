#!/usr/bin/env python

#!@Time    :2018/10/26  
#!@Author  :XINYING HU 
#!@File    :.py

# Importing Required Module for Data Preperation And Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import accuracy_score


data = pd.read_csv('E:\Mushroom\mushrooms.csv')
data2 = pd.read_csv('E:\Mushroom\mushrooms2.csv')
data3 = pd.read_csv('E:\Mushroom\mushrooms3.csv')
data4 = pd.read_csv('E:\Mushroom\mushrooms4.csv')

# Number of CLasses Counts
counts=data['class'].value_counts()
counts2=data2['class'].value_counts()
counts3=data3['class'].value_counts()
counts4=data4['class'].value_counts()
print(counts)
print(counts2)
print(counts3)
print(counts4)

# Getting all Poisonous Mushroom odor
Poisonous_Odor = data[data['class'] == 'p']['odor'].value_counts()
Poisonous_Odor = data2[data2['class'] == 'p']['odor'].value_counts()
Poisonous_Odor = data3[data3['class'] == 'p']['odor'].value_counts()
Poisonous_Odor = data4[data4['class'] == 'p']['odor'].value_counts()

# Getting all Edible Mushroom odor
Edible_Odor = data[data['class'] == 'e']['odor'].value_counts()
Edible_Odor = data2[data2['class'] == 'e']['odor'].value_counts()
Edible_Odor = data3[data3['class'] == 'e']['odor'].value_counts()
Edible_Odor = data4[data4['class'] == 'e']['odor'].value_counts()


# Given data is in character or string.To apply Machine Learning Algorithm we need to convert int integers
label_encoder = LabelEncoder()
for column in data.columns:
    data[column] = label_encoder.fit_transform(data[column])

label_encoder2 = LabelEncoder()
for column in data2.columns:
    data2[column] = label_encoder2.fit_transform(data2[column])

label_encoder3 = LabelEncoder()
for column in data3.columns:
    data3[column] = label_encoder3.fit_transform(data3[column])

label_encoder4 = LabelEncoder()
for column in data4.columns:
    data4[column] = label_encoder4.fit_transform(data4[column])


# Seaparating the labels and Features
Label = data['class']
Features = data.drop(['class'],axis=1)

Label2 = data2['class']
Features2 = data2.drop(['class'],axis=1)

Label3 = data3['class']
Features3 = data3.drop(['class'],axis=1)

Label4 = data4['class']
Features4 = data4.drop(['class'],axis=1)

ftwo_scorer = make_scorer(fbeta_score, beta=2)
acc_scorer = make_scorer(accuracy_score)

# Spltting Data in Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(Features, Label, random_state=1, train_size=0.6)
X_train2, X_test2, y_train2, y_test2 = train_test_split(Features2, Label2, random_state=1, train_size=0.6)
X_train3, X_test3, y_train3, y_test3 = train_test_split(Features3, Label3, random_state=1, train_size=0.6)
X_train4, X_test4, y_train4, y_test4 = train_test_split(Features4, Label4, random_state=1, train_size=0.6)

mushroom=[]
mushroom.append(('LogisticRegression', LogisticRegression()))
mushroom.append(('GaussianNB', GaussianNB()))
mushroom.append(('KNeighborsClassifier', KNeighborsClassifier()))
mushroom.append(('SVC', SVC()))
mushroom.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
mushroom.append(('RandomForestClassifier', RandomForestClassifier()))
acc = []
names = []
result = []
f1score = []
resultf1 = []

for name, model in mushroom:
    acc_of_model = cross_val_score(model, X_train, y_train, cv=10, scoring=acc_scorer)

    acc.append(acc_of_model)

    names.append(name)

    Out = "%s: %f (%f)" % (name, acc_of_model.mean(), acc_of_model.std())

    result.append(acc_of_model)
    print(Out)

for name, model in mushroom:
    f1_score_of_model = cross_val_score(model, X_train, y_train, cv=10, scoring=ftwo_scorer)

    f1score.append(f1_score_of_model)

    names.append(name)

    Out = "%s: %f (%f)" % (name, f1_score_of_model.mean(), f1_score_of_model.std())
    resultf1.append(f1_score_of_model)
    print(Out)


mushroom2=[]
mushroom2.append(('LogisticRegression', LogisticRegression()))
mushroom2.append(('GaussianNB', GaussianNB()))
mushroom2.append(('KNeighborsClassifier', KNeighborsClassifier()))
mushroom2.append(('SVC', SVC()))
mushroom2.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
mushroom2.append(('RandomForestClassifier', RandomForestClassifier()))
acc2 = []
names2 = []
result2 = []
f1score_2 = []
resultf1_2 = []

for name, model in mushroom2:
    acc_of_model_2 = cross_val_score(model, X_train2, y_train2, cv=10, scoring=acc_scorer)

    acc2.append(acc_of_model_2)

    names2.append(name)

    Out = "%s: %f (%f)" % (name, acc_of_model_2.mean(), acc_of_model_2.std())

    result2.append(acc_of_model_2)
    print(Out)

for name, model in mushroom2:
    f1_score_of_model_2 = cross_val_score(model, X_train2, y_train2, cv=10, scoring=ftwo_scorer)

    f1score_2.append(f1_score_of_model_2)

    names2.append(name)

    Out = "%s: %f (%f)" % (name, f1_score_of_model_2.mean(), f1_score_of_model_2.std())
    resultf1_2.append(f1_score_of_model_2)
    print(Out)


mushroom3=[]
mushroom3.append(('LogisticRegression', LogisticRegression()))
mushroom3.append(('GaussianNB', GaussianNB()))
mushroom3.append(('KNeighborsClassifier', KNeighborsClassifier()))
mushroom3.append(('SVC', SVC()))
mushroom3.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
mushroom3.append(('RandomForestClassifier', RandomForestClassifier()))
acc3 = []
names3 = []
result3 = []
f1score_3 = []
resultf1_3 = []

for name, model in mushroom3:
    acc_of_model_3 = cross_val_score(model, X_train3, y_train3, cv=10, scoring=acc_scorer)

    acc3.append(acc_of_model_3)

    names3.append(name)

    Out = "%s: %f (%f)" % (name, acc_of_model_3.mean(), acc_of_model_3.std())

    result3.append(acc_of_model_3)
    print(Out)

for name, model in mushroom3:
    f1_score_of_model_3 = cross_val_score(model, X_train3, y_train3, cv=10, scoring=ftwo_scorer)

    f1score_3.append(f1_score_of_model_3)

    names3.append(name)

    Out = "%s: %f (%f)" % (name, f1_score_of_model_3.mean(), f1_score_of_model_3.std())
    resultf1_3.append(f1_score_of_model_3)
    print(Out)


mushroom4=[]
mushroom4.append(('LogisticRegression', LogisticRegression()))
mushroom4.append(('GaussianNB', GaussianNB()))
mushroom4.append(('KNeighborsClassifier', KNeighborsClassifier()))
mushroom4.append(('SVC', SVC()))
mushroom4.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
mushroom4.append(('RandomForestClassifier', RandomForestClassifier()))
acc4 = []
names4 = []
result4 = []
f1score_4 = []
resultf1_4 = []

for name, model in mushroom4:
    acc_of_model_4 = cross_val_score(model, X_train4, y_train4, cv=10, scoring=acc_scorer)

    acc4.append(acc_of_model_4)

    names4.append(name)

    Out = "%s: %f (%f)" % (name, acc_of_model_4.mean(), acc_of_model_4.std())

    result4.append(acc_of_model_4)
    print(Out)

for name, model in mushroom4:
    f1_score_of_model_4 = cross_val_score(model, X_train4, y_train4, cv=10, scoring=ftwo_scorer)

    f1score_4.append(f1_score_of_model_4)

    names4.append(name)

    Out = "%s: %f (%f)" % (name, f1_score_of_model_4.mean(), f1_score_of_model_4.std())
    resultf1_4.append(f1_score_of_model_4)
    print(Out)




