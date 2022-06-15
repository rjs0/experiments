from pickletools import optimize
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
print("running experiment")
# if gpu is available, use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 28 * 28 #flattened picture
hidden_size = 100
num_classes = 10
num_epochs = 5
batch_size = 100
eval_size = 60000
learning_rate = 0.001
proportion=0.001
def swap_label(x):
    y = np.random.randint(0,10)
    if x==y:
        y=swap_label(x)
    return torch.tensor(y)

def swap_data(dataset, proportion):
    swap_table = {} # tensor : (old, new)
    swap_set_size = int(len(train_dataset) * proportion)
    indices = list(range(0,len(dataset)))
    """Turn shuffle on and off to debug, shuffle off will make first n pictures swapped"""
  #  random.shuffle(indices)
    for i in range(0,swap_set_size):
        
        index = indices[i]
        
        old = dataset.targets[index]
        new = swap_label(old)
        if index==0:
            print("index=0 is "+str(old)+" change to "+str(new))
        swap_table[hash(str(dataset[index][0]))]=(old.item(),new.item())
        dataset.targets[index] = new
    return dataset, swap_table
#load MNIST
train_dataset = torchvision.datasets.MNIST(root='./data',train=True
    ,transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root='./data',train=False
    ,transform=transforms.ToTensor())

# data loader to allow iterating
train_dataset, swap_table = swap_data(train_dataset, proportion)
print(train_dataset[0][1])
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
batch_size=batch_size, shuffle=True)

eval_loader = torch.utils.data.DataLoader(dataset=train_dataset,
batch_size=eval_size, shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = examples.next()
#print(samples.shape,labels.shape)
"""
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(samples[i][0],cmap='gray')
plt.show()  
"""
class NeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(NeuralNet, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear_1 = torch.nn.Linear(7 * 7 * 64, 128)
        self.linear_2 = torch.nn.Linear(128, num_classes)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv_2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        pred = self.linear_2(x)
        return pred
    
    
model = NeuralNet(num_classes)  
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# training loop
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #100 x 1 x 28 x 28 -> 100 x 784
       # images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        #forward
        outputs = model(images)
    #    print(outputs.shape)
        loss=criterion(outputs,labels)
        #backward
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        if (i+1) % 100 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}, loss = {loss.item():.4f}')
    #
    
    if epoch in [0,2,4]:
        # ... run aum tests, going through entire dataset
        for i, (images, labels) in enumerate(eval_loader):
          #  images = images.reshape(-1, 28*28).to(device)
            targets = train_dataset.targets
            logits = model(images)
            target_values = logits.gather(1, targets.view(-1, 1)).squeeze()
            """ following four lines of code taken from AUM paper, author J Shapiro"""
            masked_logits = torch.scatter(logits, 1, targets.view(-1, 1), float('-inf')) # make target values -inf so not selected again
            other_logit_values, _ = masked_logits.max(1)
            other_logit_values = other_logit_values.squeeze()

            margin_values = (target_values - other_logit_values).tolist()
            print((margin_values[:10]))
            #print(logits.shape)
            #print(train_dataset.targets.shape)
        #set = iter(train_loader)
        #print("set shape: " +str(set.shape))
        #output = model()
        #print("output shape: "+str(output.shape))
        print(epoch+1)
    
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
       # images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # predicted class
        _, predictions = torch.max(outputs,1)
        n_samples += labels.shape[0] # number of samples in batch
        n_correct += (predictions == labels).sum().item()
    
acc = 100.0 * n_correct / n_samples
print(acc)