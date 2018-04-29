
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


# In[18]:


PATH = "Data/dogscats/"
get_ipython().system('ls {PATH+"train"}')


# In[19]:


#!wget http://files.fast.ai/data/dogscats.zip
#!unzip dogscats.zip


# In[20]:


batch_size = 32
sz = 224


# In[21]:


## Image loaders
## Dataset transforms puts the images in tensor form
normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_raw = dataset.ImageFolder(PATH+"train", transform=transforms.Compose([transforms.RandomResizedCrop(sz),
                                                                            transforms.ToTensor(),
                                                                           normalise]))
train_loader = DataLoader(train_raw, batch_size=batch_size, shuffle=True, num_workers=4)

valid_raw = dataset.ImageFolder(PATH+"valid", transform=transforms.Compose([transforms.CenterCrop(sz),
                                                                            transforms.ToTensor(),
                                                                           normalise]))
valid_loader = DataLoader(valid_raw, batch_size=batch_size, shuffle=False, num_workers=4)


# In[22]:


## Create resnet model
resnet34=models.resnet34(pretrained=True)

## Loss function and optimiser
criterion = nn.CrossEntropyLoss().cuda()
optimiser = optim.Adam(resnet34.fc.parameters(), lr=0.001, weight_decay=0.001)


# In[23]:


def train(epochs):
    #epoch=1
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


# In[28]:


def validation():
    resnet34.eval()
    test_loss = 0
    correct = 0
    for data, target in valid_loader:
        data, target = Variable(data, volatile = True), Variable(target)
        output=resnet34(data)
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    test_loss /= len(valid_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(valid_loader.dataset),
    100. * correct / len(valid_loader.dataset)))


# In[29]:


## Loop through epochs training data and then testing it
for epoch in range(1,10):
    train(epoch)
    validation()

