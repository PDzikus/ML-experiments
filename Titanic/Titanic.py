#!/usr/bin/env python
# coding: utf-8

# # Problem description
# This is my take on an introductory Kaggle Competition: https://www.kaggle.com/c/titanic/ <br>
# I will be using Keras (on Tenserflow) to solve this problem.<br>
# 
# This is binary classification problem, I'll be using binary crossentropy loss function, with sigmoid activation on last network layer. Also data will require some preprocessing before it's usable for Deep Learning

# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras import models
from keras import layers

get_ipython().run_line_magic('matplotlib', 'inline')


# #### Meet and greet data

# In[13]:


train_ds = pd.read_csv('train.csv')
test_ds = pd.read_csv('test.csv')


# In[15]:


print(f"Train data shape: {train_ds.shape}")
print(f"Test data shape: {test_ds.shape}")
train_ds.sample(10)


# In[24]:


print("          Data summary")
print(train_ds.info())
print('='*40)
print("          NaN values summary")
print("--- Train data: ")
print(train_ds.isnull().sum())
print("--- Test data: ")
print(test_ds.isnull().sum())


# In[25]:


dataset_cleaner = [train_ds, test_ds]
drop_columns = ['Cabin', 'PassengerId','Ticket']
for dataset in dataset_cleaner:
    dataset.drop(drop_columns, axis = 1, inplace = True)
    dataset['Age'].fillna(dataset['Age'].mean(), inplace = True)
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)


# In[29]:


train_ds.isnull().sum()


# In[30]:


train_ds.sample(5)


# #### now I need to encode qualitative data for use in neural network

# In[32]:


for dataset in dataset_cleaner:
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
title_names = (train_ds['Title'].value_counts() < 10)


# In[48]:


for dataset in dataset_cleaner:
    dataset['Title'] = dataset['Title'].apply(lambda x: 'Misc' if (x not in title_names) or (title_names.loc[x]) else x)
    dataset.drop(['Name'], axis = 1, inplace = True)


# In[49]:


test_ds['Title'].value_counts()


# In[57]:


train_ds.sample(5)


# In[56]:


from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
for dataset in dataset_cleaner:
    dataset['Sex Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked Code'] = label.fit_transform(dataset['Embarked'])


# In[ ]:




