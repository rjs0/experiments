"""

Randomly choose 1 class, remove 95% of pics, use diff techniques ie VOG+AUM to see how difficult this classes examples are 

"""

from pickletools import optimize
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset

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


# get indices of class, choose 5% of these to keep, then merge with where argwhere is not == to class, and use these as indices to make 
# subset
class custom_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices, labels):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.targets = labels
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)
X = train_dataset.data
Y = train_dataset.targets
#print(Y.shape)
DELETE_NUMBER = 9
delete = torch.argwhere(Y==DELETE_NUMBER)
#print(three.shape)
replaceLen = int(0.05*len(delete))
choice = np.random.choice(delete.squeeze(), replaceLen, replace=False)
#print(choice.shape)
notDeletion = torch.argwhere(Y!=DELETE_NUMBER).squeeze()
notDeletion = np.array(notDeletion)
#print(notThree.shape)

newIndices = np.concatenate((notDeletion,choice))
#print(newIndices.shape)

subset = torch.utils.data.Subset(train_dataset,newIndices)

# data loader to allow iterating
train_loader = torch.utils.data.DataLoader(dataset=subset,
batch_size=batch_size, shuffle=True)

eval_loader = torch.utils.data.DataLoader(dataset=subset,
batch_size=eval_size, shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
batch_size=batch_size, shuffle=False)



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
        #images = images.reshape(-1, 28*28).to(device)
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
    """
    NEED TO ITERATE THROUGH SUBSET DATA, NOT ALL DATA. MAYBE GO BACK TO FOR LOOP AND USE DATALOADER WITH THE SUBSET AS PARAM
    """
    if epoch in [0,2,4]:
        for i, (images, labels) in enumerate(eval_loader):

        # ... run aum tests, going through entire dataset
        
            model.eval() # no dropout, in eval mode
            logits = model(images)
            targets = labels
            target_values = logits.gather(1, targets.view(-1, 1)).squeeze()
            """ following four lines of code taken from AUM paper, author J Shapiro"""
            masked_logits = torch.scatter(logits, 1, targets.view(-1, 1), float('-inf')) # make target values -inf so not selected again
            other_logit_values, _ = masked_logits.max(1)
            other_logit_values = other_logit_values.squeeze()

            margin_values = (target_values - other_logit_values).tolist()
            #get maximum logits
            max_logits, _ = logits.max(1)
            print("Shape of logits : "+ str(logits.shape))

            print("Shape of logits max: "+ str(max_logits.shape))
            margin_val=np.array(margin_values)
            # see indices of worst 50 examples
            #k = 50
            #idx = np.argpartition(margin_values, k)
            #print("lowest margins")
            # these are lowest
            #print(idx[:k])

            # find avg/median aum by class
            avgs = []
            for i in range(10):
                #print("getting data for "+str(i))
                indices = np.argwhere(labels==i).squeeze()
               # print(indices.shape) # [306]
                correct_imgs = margin_val[indices] # [306,1,28,28]
                avgs.append(np.mean(correct_imgs))
            print(avgs)
            #print(logits.shape)
            #print(train_dataset.targets.shape)

    
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
