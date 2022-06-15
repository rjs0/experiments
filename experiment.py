from pickletools import optimize
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from swapper import swap_label

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
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size,num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
    #    print(out)
        return out
    
model = NeuralNet(input_size, hidden_size, num_classes)  
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# training loop
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #100 x 1 x 28 x 28 -> 100 x 784
        images = images.reshape(-1, 28*28).to(device)
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
            images = images.reshape(-1, 28*28).to(device)
            targets = train_dataset.targets
            logits = model(images)
            target_values = logits.gather(1, targets.view(-1, 1)).squeeze()
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
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # predicted class
        _, predictions = torch.max(outputs,1)
        n_samples += labels.shape[0] # number of samples in batch
        n_correct += (predictions == labels).sum().item()
    
acc = 100.0 * n_correct / n_samples
print(acc)