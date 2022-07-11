import torch 
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70], 
                   [74, 66, 43], 
                   [91, 87, 65], 
                   [88, 134, 59], 
                   [101, 44, 37], 
                   [68, 96, 71], 
                   [73, 66, 44], 
                   [92, 87, 64], 
                   [87, 135, 57], 
                   [103, 43, 36], 
                   [68, 97, 70]], 
                  dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119],
                    [57, 69], 
                    [80, 102], 
                    [118, 132], 
                    [21, 38], 
                    [104, 118], 
                    [57, 69], 
                    [82, 100], 
                    [118, 134], 
                    [20, 38], 
                    [102, 120]], 
                   dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

train_ds = TensorDataset(inputs, targets)
#print(train_ds[0:3])

batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle = True)

#for xb, yb in train_dl: 
 #   print(xb)
  #  print(yb)
   # break

model = nn.Linear(3,2)
#print(model.weight)
#print(model.bias)

#PyTorch models also have a helpful .parameters method, which returns a list containing all the weights and bias 
#matrices present in the model. For our linear regression model, we have one weight matrix and one bias matrix.

preds = model(inputs)
#print(preds)

loss_function = F.mse_loss

loss = loss_function(model(inputs), targets)
#print(loss)

optimizer = torch.optim.SGD(model.parameters(), lr = 1e-5)

# Overall structure of a program:

    #Generate predictions

    #Calculate the loss

    #Compute gradients w.r.t the weights and biases

    #Adjust the weights by subtracting a small quantity proportional to the gradient

    #Reset the gradients to zero

def fit (epochs, model, loss_function, optimizer, train_dl):
    for epoch in range (epochs):
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_function(pred, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))
            
fit(100, model, loss_function, optimizer, train_dl)     

preds = model(inputs)
print(preds)
print(targets)