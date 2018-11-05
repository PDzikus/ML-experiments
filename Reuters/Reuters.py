#!/usr/bin/env python
# coding: utf-8

# ## Reuters dataset
# Single-label, multiclass classification problem

# In[1]:


from keras.datasets import reuters
help(reuters)


# In[14]:


(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = 10000)
print(f'Train data size: {train_data.shape}')
print(f'Test data size: {test_data.shape}')


# ### 2. Data preparation
# I'm going to vectorize the X data - vector of 0s and 1s, coding which words were find in the sequence <br>
#     For labels we'll use one-hot-encoding with a keras built in function

# In[10]:


import numpy as np

def vectorize_sequence(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)

from keras.utils.np_utils import to_categorical

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)


# ### 3. Building model

# In[41]:


from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation = 'relu', input_shape = (10000, )))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(46, activation = 'softmax'))

model.compile(optimizer = 'rmsprop',
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])


# In[42]:


x_val = x_train[:1000]
y_val = one_hot_train_labels[:1000]
partial_x_train = x_train[1000:]
partial_y_train = one_hot_train_labels[1000:]


# In[43]:


history = model.fit(partial_x_train, partial_y_train,
                   epochs=20,
                   batch_size=512,
                   validation_data = (x_val, y_val))


# In[44]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[45]:


plt.clf()

acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# ### 4. Training the model
# network overfits around 9th epoch, so we'll just retrain it until that point

# In[47]:


model = models.Sequential()
model.add(layers.Dense(64, activation = 'relu', input_shape=(10000, )))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(46, activation = 'softmax'))

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(partial_x_train, partial_y_train, epochs = 9, batch_size = 512,
         validation_data = (x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)
print(results)


# In[58]:


model = models.Sequential()
model.add(layers.Dense(92, activation = 'relu', input_shape=(10000, )))
#model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(46, activation = 'softmax'))

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(x_train, one_hot_train_labels, epochs = 9, batch_size = 512,
         validation_data = (x_test, one_hot_test_labels))
results = model.evaluate(x_test, one_hot_test_labels)
print(results)


# In[59]:


predictions = model.predict(x_test)
predictions[0].shape
np.sum(predictions[0])


# In[60]:


np.argmax(predictions[0])


# In[63]:


plt.plot(range(1,47), predictions[3])
plt.show()


# In[ ]:




