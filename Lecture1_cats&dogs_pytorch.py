
# coding: utf-8

# In[1]:


from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
# For showing and formatting images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# For importing datasets into pytorch
import torchvision.datasets as dataset

# Used for dataloaders
from torch.utils.data import DataLoader

# For pretrained resnet34 model
import torchvision.models as models

# For optimisation function
import torch.nn as nn
import torch.optim as optim

# For turning data into tensors
import torchvision.transforms as transforms

# For loss function
import torch.nn.functional as F

# Tensor to wrap data in
from torch.autograd import Variable


# In[2]:


PATH = "Data/dogscats/"
get_ipython().system('ls {PATH+"train"}')


# In[3]:


#!wget http://files.fast.ai/data/dogscats.zip
#!unzip dogscats.zip


# In[4]:


batch_size = 256
sz = 224


# In[6]:


## Image loaders
## Dataset transforms puts the images in tensor form
normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_raw = dataset.ImageFolder(PATH+"train", transform=transforms.Compose([transforms.RandomResizedCrop(224),
                                                                            transforms.ToTensor(),
                                                                           normalise]))
train_loader = DataLoader(train_raw, batch_size=16, shuffle=True, num_workers=4)


# In[ ]:


## Create resnet model
resnet34=models.resnet34(pretrained=True)

## Loss function and optimiser
criterion = nn.CrossEntropyLoss().cuda()
optimiser = optim.Adam(resnet34.fc.parameters(), lr=0.001, weight_decay=0.001)


# In[ ]:


#def train(epochs):
epoch=1
resnet34.train()
for batch_idx, (data, target) in enumerate(train_loader):
    #print(batch_idx)
    data, target = Variable(data), Variable(target)
    optimiser.zero_grad()
    output = resnet34(data)
    loss=criterion(output, target)
    loss.backward()
    optimiser.step()
    if batch_idx % 10 == 0:        
        print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx*len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.data[0]))

