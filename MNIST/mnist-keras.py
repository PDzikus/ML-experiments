#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist
help(mnist)


# In[2]:


(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
print(f"Train images shape: {train_images.shape}")
print(f"Train labels shape: {train_labels.shape}")
print(f"Test images shape: {test_images.shape}")


# In[3]:


from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))


# In[4]:


network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[5]:


train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32') / 255


# In[6]:


from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# In[7]:


network.fit(train_images, train_labels, epochs=5, batch_size=128)


# In[8]:


test_loss, test_accuracy = network.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_accuracy}')


# ## Now let's do the same with Convolutions!
# Building new model with 3 layers of convnet on top of dense layer

# In[11]:


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))


# In[12]:


model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))

model.summary()


# In[15]:


(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# In[16]:


model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_images, train_labels, epochs = 5, batch_size = 64)


# In[17]:


test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Convnet test accuracy: {test_acc}')


# In[ ]:




