#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn


# In[21]:


df = pd.read_csv('C:/Users/S/Desktop/BlackFriday_Processed.csv')


# In[22]:


df.describe()
df.head(10)


# In[23]:


cleanup_city = {"City_Category": {"A": 1, "B": 2, "C": 3}}
df.replace(cleanup_city, inplace=True)
cleanup_age = {"Age": {"0-17": 1, "18-25": 2, "26-35": 3, "36-45": 4,"46-50": 5, "51-55":6, "55+":7 }}
df.replace(cleanup_age, inplace=True)
df.describe()
df.head(10)


# In[ ]:


df.to_csv('C:/Users/S/Desktop/BlackFriday_PreProcessed_Final.csv', index=False, sep=',')


# In[ ]:




