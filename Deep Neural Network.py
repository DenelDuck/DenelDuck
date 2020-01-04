#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# In[2]:


np.random.seed(0)


# In[3]:


n_pts = 500
X, y = datasets.make_circles(n_samples = n_pts, random_state = 123, noise = 0.1, factor = 0.2)  #2 circles, een buitenste en binnenste


# In[4]:


plt.scatter(X[y==0, 0], X[y==0, 1])   #Buitenste cirkel
plt.scatter(X[y==1, 0], X[y==1, 1])   #Binnenste cirkel


# In[5]:


model = Sequential()
model.add(Dense(4, input_shape =(2,), activation = 'sigmoid'))    #Hidden layer neuronen
model.add(Dense(1, activation = 'sigmoid'))   #Output layer
model.compile(Adam(lr = 0.01), 'binary_crossentropy', metrics = ['accuracy'])


# In[6]:


h = model.fit(x = X, y = y, verbose = 1, batch_size = 20, epochs = 100, shuffle = 'true')   #Trainen network, shuffle zodat niet in lokaal minimum blijft


# In[7]:


plt.plot(h.history['accuracy'])    #Visueel accuracy
plt.xlabel('epoch')
plt.legend(['accuracy'])
plt.title('accuracy')


# In[8]:


plt.plot(h.history['loss'])   #Visueel loss
plt.xlabel('epoch')
plt.legend(['loss'])
plt.title('loss')


# In[9]:


def plot_decision_boundary(X, y, model):
    x_span = np.linspace(min(X[:,0]) - 0.25, max(X[:,0]) + 0.25)
    y_span = np.linspace(min(X[:,1]) - 0.25, max(X[:,1]) + 0.25)
    xx, yy = np.meshgrid(x_span, y_span)
    grid = np.c_[xx.ravel(), yy.ravel()]
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)


# In[10]:


plot_decision_boundary(X, y, model)
plt.scatter(X[:n_pts,0], X[:n_pts,1])
plt.scatter(X[n_pts:,0], X[n_pts:,1])


# In[12]:


plot_decision_boundary(X, y, model)
plt.scatter(X[:n_pts,0], X[:n_pts,1])
plt.scatter(X[n_pts:,0], X[n_pts:,1])
x = 0.1
y = 0
point = np.array([[x, y]])
prediction = model.predict(point)
plt.plot([x], [y], marker = 'o', markersize = 10, color = 'red')
print('Prediction is: ', prediction)


# In[ ]:





# In[ ]:




