
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


# From now on I'll work on copy of dataset for various experiments with encoding

# In[12]:


def copy_dataset():
    train_data = train_ds.drop(['Survived'], axis = 1).copy(deep = True)
    train_targets = train_ds['Survived'].copy(deep = True)
    test_data = test_ds.copy(deep = True)
    return (train_data, train_targets, test_data)


# In[13]:


from sklearn.preprocessing import LabelEncoder

train_data, train_targets, test_data = copy_dataset()
dataset_cleaner = [train_data,test_data]

label = LabelEncoder()
for dataset in dataset_cleaner:
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset.drop(['Sex', 'Embarked', 'Title'], axis = 1, inplace = True)


# In[14]:


test_data.sample(5)


# # Experiment 1

# Let's split data and normalize :)

# In[15]:


for dataset in dataset_cleaner:
    mean = dataset.mean()
    std = dataset.std()
    dataset -= mean
    dataset /= std


# In[16]:


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(16, activation = 'relu', input_shape = (train_data.shape[1], )))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(16, activation = 'relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(1, activation = 'sigmoid'))

    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model


# We don't really have that much data for training, so let's use K-Fold validation.

# In[17]:


def test_model(k = 4, model_builder=build_model):
    k = 4
    num_val_samples = train_data.shape[0] // k
    num_epochs = 30
    all_history = []

    for i in range(k):
        print(f"processing fold #{i}")
        val_data = train_data[i*num_val_samples : (i+1)*num_val_samples]
        val_targets = train_targets[i*num_val_samples : (i+1)*num_val_samples]

        partial_train_data = np.concatenate([train_data[0:i*num_val_samples], train_data[(i+1)*num_val_samples:]], axis = 0)
        partial_train_targets = np.concatenate([train_targets[0:i*num_val_samples], train_targets[(i+1)*num_val_samples:]], axis = 0)

        model = model_builder()
        history = model.fit(partial_train_data, partial_train_targets, epochs = num_epochs, batch_size = 1,
                           validation_data = (val_data, val_targets), verbose = 0)
        all_history.append(history)
    return all_history


# In[18]:


def prepare_plot_data(history):
    
    def mean_history(histories, column):
        array = np.array([x.history[column] for x in histories])
        average = np.mean(array, axis = 0)
        return average

    average_val_loss = mean_history(history, 'val_loss')
    average_loss = mean_history(history, 'loss')
    average_acc = mean_history(history, 'acc')
    average_val_acc = mean_history(history, 'val_acc')
    return (average_loss, average_val_loss, average_acc, average_val_acc)


# In[ ]:


def plot_loss(avg_loss, avg_val_loss):
    plt.clf()
    plt.plot(range(len(avg_loss)), avg_loss, 'bo', label = 'Training loss')
    plt.plot(range(len(avg_val_loss)), avg_val_loss, 'b', label = 'Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# In[ ]:


def plot_acc(avg_acc, avg_val_acc):
    plt.clf()
    plt.plot(range(len(avg_acc)), avg_acc, 'bo', label = 'Training accuracy')
    plt.plot(range(len(avg_val_acc)), avg_val_acc, 'b', label = 'Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# In[ ]:


all_history = test_model(model_builder = build_model)


# In[ ]:


avg_loss, avg_val_loss, avg_acc, avg_val_acc = prepare_plot_data(all_history)
plot_loss(avg_loss, avg_val_loss)
plot_acc(avg_acc, avg_val_acc)


# In[ ]:


model = build_model()
model.fit(train_data, train_label, epochs = 30, batch_size = 1, verbose = 0)


# In[ ]:


from sklearn.metrics import roc_curve, roc_auc_score

print(model.evaluate(train_data, train_label))
predict = model.predict(train_data)

auc = roc_auc_score(train_label, predict)
print (f"AUC: {auc}")
fpr, tpr, thresholds = roc_curve(train_label, predict)

plt.plot([0,1], [0,1], linestyle = '--')
plt.plot(fpr, tpr, marker = '.')
plt.show()


# In[ ]:


i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index = i), 'tpr': pd.Series(tpr, index = i),
                   '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i),
                   'thresholds': pd.Series(thresholds, index = i)})
print(roc.iloc[(roc.tf-0).abs().argsort()[:1]])

fig, ax = plt.subplots()
plt.plot(roc['tpr'])
plt.plot(roc['1-fpr'], color = 'red')
plt.xlabel('1-False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
ax.set_xticklabels([])


# In[ ]:


roc.iloc[(roc.tf-0).abs().argsort()[:1]]


# In[337]:


result = model.predict(test_data)
cutoff = 0.375958

results = pd.DataFrame([0 if result[x] < cutoff else 1 for x in range(len(result))], columns = ['Survived'])
submittion = pd.read_csv('test.csv', usecols=[0])
submittion = pd.concat([submittion, results], axis = 1)
submittion.to_csv("submittion_1.csv", index = False, encoding = 'utf-8')


# Interesting note: after submitting this file with a cutoff point set at 0.5, my accuracy was at 77.99%.
# With cutoff point set to 0.4, my accuracy was 78.94%

# # Experiment 2
# one-hot encoding the data

# In[327]:


train_data, train_targets, test_data = copy_dataset()


# In[328]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

def process(dataset):
    columns_to_standardize = ['Age','SibSp','Parch', 'Fare']
    dataset = pd.get_dummies(dataset, columns = ['Pclass', 'Sex', 'Embarked', 'Title'], sparse = False)
    dataset[columns_to_standardize] = dataset[columns_to_standardize].astype('float64')
    dataset[columns_to_standardize] = scaler.fit_transform(dataset[columns_to_standardize])
    return dataset


# In[329]:


train_data = process(train_data)
test_data = process(test_data)


# In[330]:


print(f"Train data shape: {train_data.shape}")
train_data.sample(5)


# In[331]:


all_history = test_model(model_builder = build_model)


# In[332]:


avg_loss, avg_val_loss, avg_acc, avg_val_acc = prepare_plot_data(all_history)
plot_loss(avg_loss, avg_val_loss)
plot_acc(avg_acc, avg_val_acc)

