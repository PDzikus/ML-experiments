#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import numpy as np
import matplotlib.pyplot as plt


# ### 1. Dataset
# Loading dataset from keras, dataset description.

# In[2]:


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)


# In[3]:


help(imdb)
print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")


# ### 2. Preparing the data with one-hot-encoding
# labels also vectorized and converted to floats

# In[4]:


def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# ### 3. Building and training a model

# In[5]:


model = models.Sequential()
model.add(layers.Dense(128, activation = 'relu', input_shape = (10000, )))
model.add(layers.Dense(128, activation = 'relu'))
model.add(layers.Dense(128, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))


# In[6]:


model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[7]:


x_val = x_train[:10000]
y_val = y_train[:10000]
partial_x_train = x_train[10000:]
partial_y_train = y_train[10000:]


# In[8]:


history = model.fit(partial_x_train, partial_y_train,                     epochs = 20,                     batch_size = 512,                     validation_data = (x_val, y_val))


# ### 4. History of training

# In[9]:


# help(history)
history_dict = history.history
print(history_dict.keys())


# In[10]:


loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label  ='Training loss')
plt.plot(epochs, val_loss_values, 'b', label = 'Validation_loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[11]:


plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc_values, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc_values, 'b', label = 'Validation acc')
plt.title('Training and validation acuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend
plt.show()


# ### 5. Retraining new model

# In[12]:


model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape = (10000, )))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop',
             loss = 'binary_crossentropy',
             metrics = ['accuracy'])
model.fit(x_train, y_train, epochs=4, batch_size = 512)
results = model.evaluate(x_test, y_test)
print(results)


# In[13]:


results


# In[15]:


model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape = (10000, )))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop',
             loss = 'binary_crossentropy',
             metrics = ['accuracy'])
model.fit(x_train, y_train, epochs=4, batch_size = 512)
results = model.evaluate(x_test, y_test)
print(results)


# In[ ]:




