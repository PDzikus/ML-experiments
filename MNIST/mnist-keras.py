#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist
help(mnist)


# In[10]:


(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
print(f"Train images shape: {train_images.shape}")
print(f"Train labels shape: {train_labels.shape}")
print(f"Test images shape: {test_images.shape}")


# In[11]:


from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))


# In[12]:


network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[13]:


train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32') / 255


# In[14]:


from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# In[15]:


network.fit(train_images, train_labels, epochs=5, batch_size=128)


# In[16]:


test_loss, test_accuracy = network.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_accuracy}')


# In[ ]:


print("test")

