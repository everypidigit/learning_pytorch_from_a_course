import torch as torch
import numpy as np

#Autograd things

#we can specify that a tensor need to be autograded, it creates a function that is used in the backprop to get gradients. 
x = torch.randn(3, requires_grad = True)
print(x)

y = x + 2
print(x)

z = y * 2 * 2
print(z)

z = z.mean()
print(z)

z.backward() #dz/dx
print(x.grad)

# Convert the numpy array to a torch tensor.
#y = torch.from_numpy(x)
#y

