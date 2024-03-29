from pickletools import optimize
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#from swapper import swap_label
print("digit class")
# if gpu is available, use GPU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 28 * 28 #flattened picture
hidden_size = 100
num_classes = 10
num_epochs = 5
batch_size = 100
eval_size = 60000
learning_rate = 0.001

#load MNIST
train_dataset = torchvision.datasets.MNIST(root='./data',train=True
    ,transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root='./data',train=False
    ,transform=transforms.ToTensor())

# data loader to allow iterating
master_table = {}
""" 
this is really slow
"""
limit = len(train_dataset)
limit = 10
for i in range(0, 10):
    str_rep = str(train_dataset[i][0])
    master_table[hash(str_rep)]=i
    
print(train_dataset[0][0].shape)
print("Done making master hash table")
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
# create a bit mask of a tensor, zeroing out the smallest k parameters
def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


x = torch.randn(2, 3, 4, 5)
res = torch.topk(x.view(-1), k=3)

idx = unravel_index(res.indices, x.size())
print(x[idx] == res.values)

class NeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(NeuralNet, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear_1 = torch.nn.Linear(7 * 7 * 64, 128)
        self.linear_2 = torch.nn.Linear(128, num_classes)
       # self.dropout = torch.nn.Dropout(p=0.5)
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
       # x = self.dropout(x)
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
        #images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        #forward
        outputs = model(images)
    #    print(outputs.shape)
        loss=criterion(outputs,labels)
      #  print(loss)
        #backward
        optimizer.zero_grad()
        loss.backward()
        conv = model.conv_1.weight
       # print(conv.shape)
        weights=model.linear_1.weight
       # print(weights.shape)
        optimizer.step()
      #  model.linear_1.weight=nn.Parameter(torch.randn(128,3136))
        if (i+1) % 100 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}, loss = {loss.item():.4f}')
    #
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