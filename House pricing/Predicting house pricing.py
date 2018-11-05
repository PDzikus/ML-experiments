#!/usr/bin/env python
# coding: utf-8

# # Boston housing prices prediction
# regression problem with Deep Learning/Keras

# ## 1. Loading and preparing data

# In[2]:


from keras.datasets import boston_housing
help(boston_housing)


# In[3]:


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print(f"Train data shape: {train_data.shape}")
print(f"Test data shape:  {test_data.shape}")


# #### normalizing data

# In[4]:


mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


# ## 2. building Keras model

# In[5]:


from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation = 'relu', input_shape = ( train_data.shape[1], )))
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['mae'])
    return model


# Because of small amount of data we'll be using for this model, we need to use K-fold cross-validation scheme for training to avoid (reduce) overfitting

# In[ ]:


import numpy as np

k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []


# In[9]:


for i in range(k):
    print(f"processing fold #{i}")
    val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets = train_targets[i*num_val_samples:(i+1)*num_val_samples]
    
    partial_train_data = np.concatenate([train_data[:i*num_val_samples], train_data[(i+1)*num_val_samples:]], axis = 0)
    partial_train_targets = np.concatenate([train_targets[:i*num_val_samples], train_targets[(i+1)*num_val_samples:]], axis = 0)
    
    model = build_model()
    model.fit(partial_train_data, partial_train_targets, epochs = num_epochs, batch_size = 1, verbose = 0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose = 0)
    all_scores.append(val_mae)


# In[11]:


print(f"Mean Square Error per fold: {all_scores}")
print(f"mean MSE: {np.mean(all_scores)}")


# #### new version of model training
#     This time I will train model with saved history of training

# In[13]:


num_epochs = 500
all_mae_histories = []
for i in range(k):
    print(f"processing fold #{i}")
    val_data = train_data[i*num_val_samples: (i+1)*num_val_samples]
    val_targets = train_targets[i*num_val_samples: (i+1)*num_val_samples]
    
    partial_train_data = np.concatenate([train_data[:i*num_val_samples], train_data[(i+1)*num_val_samples:]], axis = 0)
    partial_train_targets = np.concatenate([train_targets[:i*num_val_samples], train_targets[(i+1)*num_val_samples:]], axis = 0)
    
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, validation_data = (val_data, val_targets),
                        epochs = num_epochs, batch_size = 1, verbose = 0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)


# In[17]:


import matplotlib.pyplot as plt

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range (num_epochs)]
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# Because above plot is hard to read, we'll remove first 10 data points, and the rest of the plot will be smoothed - ech point will be replaced with an exponential moving average of the previous points

# In[20]:


def smooth_curve(points, factor = 0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history)+1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# It looks like the model starts to overfit around 80th epoch, so we'll build a fresh new model with only 80 epochs and all train_data.

# In[21]:


model = build_model()
model.fit(train_data, train_targets, epochs = 80, batch_size = 1, verbose = 0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(f"Final model MAE score: {test_mae_score}")


# In[ ]:




