import torch
import numpy as np
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F

dataset = MNIST(root = 'data/', download = True)
print(len(dataset))
test_dataset = MNIST(root = 'data/', train = False, transform = torchvision.transforms.ToTensor())
print(len(test_dataset))

image, label = dataset[0]
plt.imshow(image, cmap = 'gray')
#plt.show() #this method is used to open the image and display it

dataset = MNIST(root='data/',
                train=True,
                transform=torchvision.transforms.ToTensor())

img_tensor, label = dataset[0]
print(img_tensor.shape, label)

plt.imshow(img_tensor[0, 5:13, 5:13], cmap = 'gray')
#plt.show()

#splitting the data into training and validation datasets
train_data, validation_data = random_split(dataset, [50000, 10000])

# setting the batch size for training batches and loading datasets into DataLoader instances
batch_size = 128
train_loader = DataLoader(train_data, batch_size, shuffle=True)
val_loader = DataLoader(validation_data, batch_size)
input_size = 28 * 28
num_classes = 10
img_tensor.reshape(1, 784).shape

class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out

logistic_model = MnistModel()

for images, labels in train_loader:
    print(images.shape)
    outputs = logistic_model(images)
    break
    print('outputs.shape : ', outputs.shape)
    print('Sample outputs : \n', outputs[:2].data)

probs = F.softmax(outputs, dim = 1)

max_probs, preds = torch.max(probs, dim = 1)
print(preds)
print(max_probs)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim = 1)
    return torch.tensor(torch.sum(preds==labels).item()/len(preds))

loss_funct = F.cross_entropy
loss = loss_funct(outputs, labels)
print(loss)

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_funct=torch.optim.SGD):
    optimizer = opt_funct(model.parameters(), lr)
    history = []  # for the recording of results

    for epoch in range(epochs):

        # Training
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)

    return history

class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out

    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # combine batch losses and find the average loss on 1 batch
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(
            batch_accs).mean()  # combine the batch accuracies and find the average accuracy on 1 batch of data
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

model = MnistModel()

#history3 = fit(15, 0.001, model, train_loader, val_loader)
#history2 = fit(5, 0.001, model, train_loader, val_loader)
#history1 = fit(5, 0.001, model, train_loader, val_loader)

#test_loader = DataLoader(test_dataset, batch_size=256)
#result = evaluate(model, test_loader)
#print(result)

#torch.save(model.state_dict(), 'mnist_logistic_1.pth')

model2 = MnistModel()
model2.load_state_dict(torch.load('mnist_logistic_1.pth'))
model2.state_dict()

test_loader = DataLoader(test_dataset, batch_size=256)
result = evaluate(model2, test_loader)
print(result)


def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model2(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()

img, label = test_dataset[0]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))

img, label = test_dataset[10]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))

