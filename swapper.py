from pickletools import optimize
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
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

# take in x 0-9, output a y 0-9, where x!=y
def swap_label(x):
    y = np.random.randint(0,10)
    if x==y:
        y=swap_label(x)
    return torch.tensor(y)

# create the swapped set
PROPORTION_SWAP = 0.0001
swap_set_size = int(len(train_dataset) * PROPORTION_SWAP)
#pure_set_size = len(train_dataset) - swap_set_size
#swap_set, pure_set = torch.utils.data.random_split(train_dataset, [swap_set_size, pure_set_size])

#print('swap data set size:', len(swap_set))
#print('pure data set size:', len(pure_set))

# iterate through swap_set and perform swap, remembering results with hashtable
swap_table = {} # tensor : (old, new)
indices = list(range(0,len(train_dataset)))
random.shuffle(indices)
for i in range(0,swap_set_size):
   index = indices[i]
   if i==0:
       index=0
   old = train_dataset.targets[index]
  # print("old: "+str(old))
   new = swap_label(old)
  # print("new: "+str(new))
   swap_table[hash(train_dataset[index])]=(old.item(),new.item())
   train_dataset.targets[index] = new

print("Done")
print(swap_table)

img=iter(torch.utils.data.DataLoader(train_dataset)).next()
print(img)
print(train_dataset.targets[0])
#print(train_dataset.targets[27])
#print(swap_label(train_dataset.targets[0]))

"""
feature = example[0]
label = example[1]
example[1]=60
print(label)
print(feature.size())
plt.imshow(feature.reshape(28,28), cmap="gray")
plt.show()
"""
