#Tensors Basics

import torch
import numpy as np
#import pandas as pd
#import sklearn
#import matplotlib.pyplot as plt


#if you want to check the torch version:
#print(f"PyTorch version: {torch.__version__}")

x = torch.rand(2,2)
y = torch.rand(2,2)

z = x * y
#we can add tensors in this way:
y.add_(x)

#we can multiply tensors in this way:
y.mul_(x)
print(y)

#in my case, Torch is running on the CPU, and doing b = a.numpy() means that 'b' points to the same spot in the memory to which 'a' points,
#so they are actually the same thing
a = torch.ones(2)
b = a.numpy()

 #checking the value of 'b', which has to be equal to 'a' since they are the same exact thing
print(b) #and they are, indeed

#adding 1 to the tensor 'a' changed the values inside of 'a'
a.add_(1)

#now, by printing 'b', we can see that 'b' has changed too: it points to the same spot in the memory.
#print(b)

# in case, there is an mps device available
mps = torch.device("mps")



