#!/usr/bin/env python
# coding: utf-8

# # Problem description
# This is my take on an introductory Kaggle Competition: https://www.kaggle.com/c/titanic/ <br>
# I will be using Keras (on Tenserflow) to solve this problem.<br>
# 
# This is binary classification problem, I'll be using binary crossentropy loss function, with sigmoid activation on last network layer. Also data will require some preprocessing before it's usable for Deep Learning

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras import models
from keras import layers

get_ipython().run_line_magic('matplotlib', 'inline')


# #### Meet and greet data

# In[2]:


train_ds = pd.read_csv('train.csv')
test_ds = pd.read_csv('test.csv')


# In[3]:


print(f"Train data shape: {train_ds.shape}")
print(f"Test data shape: {test_ds.shape}")
train_ds.sample(10)


# In[4]:


print("          Data summary")
print(train_ds.info())
print('='*40)
print("          NaN values summary")
print("--- Train data: ")
print(train_ds.isnull().sum())
print("--- Test data: ")
print(test_ds.isnull().sum())


# In[5]:


dataset_cleaner = [train_ds, test_ds]
drop_columns = ['Cabin', 'PassengerId','Ticket']
for dataset in dataset_cleaner:
    dataset.drop(drop_columns, axis = 1, inplace = True)
    dataset['Age'].fillna(dataset['Age'].mean(), inplace = True)
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)


# In[6]:


train_ds.isnull().sum()


# In[7]:


train_ds.sample(5)


# #### now I need to encode qualitative data for use in neural network

# In[8]:


for dataset in dataset_cleaner:
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
title_names = (train_ds['Title'].value_counts() < 10)


# In[9]:


for dataset in dataset_cleaner:
    dataset['Title'] = dataset['Title'].apply(lambda x: 'Misc' if (x not in title_names) or (title_names.loc[x]) else x)
    dataset.drop(['Name'], axis = 1, inplace = True)


# In[10]:


test_ds['Title'].value_counts()


# In[11]:


train_ds.sample(5)


# In[12]:


from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
for dataset in dataset_cleaner:
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset.drop(['Sex', 'Embarked', 'Title'], axis = 1, inplace = True)


# In[20]:


train_data.sample(5)


# In[17]:


train_label = train_ds['Survived']
train_data = train_ds.drop(['Survived'], axis = 1)

dataset_cleaner = [train_data, test_ds]


# Let's split data and normalize :)

# In[18]:


train_label = train_ds['Survived']

for dataset in dataset_cleaner:
    mean = dataset.mean()
    std = dataset.std()
    dataset -= mean
    dataset /= std


# In[63]:


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(32, activation = 'relu', input_shape = (train_data.shape[1], )))
    model.add(layers.Dense(32, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))

    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model


# We don't really have that much data for training, so let's use K-Fold validation.

# In[64]:


k = 4
num_val_samples = train_data.shape[0] // k
num_epochs = 10
all_history = []

for i in range(k):
    print(f"processing fold #{i}")
    val_data = train_data[i*num_val_samples : (i+1)*num_val_samples]
    val_targets = train_label[i*num_val_samples : (i+1)*num_val_samples]
    
    partial_train_data = np.concatenate([train_data[0:i*num_val_samples], train_data[(i+1)*num_val_samples:]], axis = 0)
    partial_train_targets = np.concatenate([train_label[0:i*num_val_samples], train_label[(i+1)*num_val_samples:]], axis = 0)
    
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, epochs = num_epochs, batch_size = 1,
                       validation_data = (val_data, val_targets), verbose = 0)
    all_history.append(history)


# In[65]:


def mean_history(histories, column):
    array = np.array([x.history[column] for x in histories])
    average = np.mean(array, axis = 0)
    return average

average_val_loss = mean_history(all_history, 'val_loss')
average_loss = mean_history(all_history, 'loss')
average_acc = mean_history(all_history, 'acc')
average_val_acc = mean_history(all_history, 'val_acc')


# In[66]:


plt.plot(range(num_epochs), average_loss, 'bo', label = 'Training loss')
plt.plot(range(num_epochs), average_val_loss, 'b', label = 'Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[67]:


plt.plot(range(num_epochs), average_acc, 'bo', label = 'Training accuracy')
plt.plot(range(num_epochs), average_val_acc, 'b', label = 'Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[68]:


model = build_model()
model.fit(train_data, train_label, epochs = 7, batch_size = 1)
results = model.evaluate(train_data, train_label)
print(results)


# In[71]:


result = model.predict(test_ds)
print(result)


# In[ ]:




