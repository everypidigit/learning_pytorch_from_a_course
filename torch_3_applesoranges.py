import torch
import numpy as np

# Features 
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32')

# We should work on inputs and targets separately
#yield_apple  = w11 * temp + w12 * rainfall + w13 * humidity + b1
#yield_orange = w21 * temp + w22 * rainfall + w23 * humidity + b2

# Converting the inputs and outputs (np arrays) to the Torch tensors
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

#initialiazing weights to use in the function
#since we have 3 features for 2 functions, we will create this many weights
w = torch.randn(2, 3, requires_grad=True)

#initializing biases to use in the function. since we have just 2 functions (for orange and for apples), we will create 2 biases
b = torch.randn(2, requires_grad=True)

#creating a model for the calculations
# @ is a matris multiplication and .t returns a transposed matrix
def model(x):
    return x @ w.t() + b

#preds is our prediction. we call the model function for the inputs and wait for it to compute our predictions for the crop yieds.
preds = model(inputs)
print(preds)

#The loss function is needed to know how well a model performs.
# MSE loss (one of the simplest ways of computing the loss)
# torch.sum returns the sum of all elements in a tensor
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

# Compute the MSE
loss = mse(preds, targets)
print(loss)

# the .backward computes the gradient (derivative)
loss.backward()
#printing gradientf for weights
print(w)
print(w.grad)

#If a gradient element is positive:

#  increasing the weight element's value slightly will increase the loss
#  decreasing the weight element's value slightly will decrease the loss


#If a gradient element is negative:

#  increasing the weight element's value slightly will decrease the loss
#  decreasing the weight element's value slightly will increase the loss

with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5
    
