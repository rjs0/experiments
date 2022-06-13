from pickletools import optimize
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# if gpu is available, use GPU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 28 * 28 #flattened picture
batch_size = 100

#load MNIST
train_dataset = torchvision.datasets.MNIST(root='./data',train=True
    ,transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root='./data',train=False
    ,transform=transforms.ToTensor())

# load shuffled images
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
batch_size=batch_size, shuffle=True)

example = train_dataset[26]
print(train_dataset.targets)
"""
feature = example[0]
label = example[1]
example[1]=69
print(label)
print(feature.size())
plt.imshow(feature.reshape(28,28), cmap="gray")
plt.show()
"""
