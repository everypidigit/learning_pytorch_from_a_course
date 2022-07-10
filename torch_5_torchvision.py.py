#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np 
from torch.utils.data import TensorDataset, DataLoader
import torchvision as vision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


# In[3]:


dataset = MNIST(root = 'data/', download = True)
print(len(dataset))
test_dataset = MNIST(root = 'data/', train = False)
print(len(test_dataset))


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


image, label = dataset[0]
plt.imshow(image, cmap = 'gray')
print('Label:', label)


# In[6]:


image, label = dataset[20]
plt.imshow(image, cmap = 'gray')
print('Label:', label)


# In[7]:


import torchvision.transforms as transforms


# In[8]:


dataset = MNIST(root='data/', 
                train=True,
                transform=transforms.ToTensor())


# In[9]:


img_tensor, label = dataset[0]
print(img_tensor.shape, label)


# In[ ]:




