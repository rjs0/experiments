from pickletools import optimize
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
#from swapper import swap_label
# if gpu is available, use GPU
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
    #random.shuffle(indices)
    for i in range(0,swap_set_size):
        
        index = indices[i]
        
        old = dataset.targets[index]
        new = swap_label(old)
        if index==0:
            print("index=0 is "+str(old)+" change to "+str(new))
       #swap_table[hash(str(dataset[index][0]))]=(old.item(),new.item())
 
        swap_table[str(dataset[index][0])]=(old.item(),new.item())
        dataset.targets[index] = new
    return dataset, swap_table
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
#train_dataset, swap_table = swap_data(train_dataset, proportion)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
batch_size=batch_size, shuffle=True)

eval_loader = torch.utils.data.DataLoader(dataset=train_dataset,
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

X, Y = train_dataset.data, train_dataset.targets
vog = {}
normalized_vog = []
training_vog_stats=[]
training_labels=[]
training_class_variances = list(list() for i in range(10))
training_class_variances_stats = list(list() for i in range(10))
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #100 x 1 x 28 x 28 -> 100 x 784
        #images = images.reshape(-1, 28*28).to(device)
        model.train()
        labels = labels.to(device)
        #print(images.shape)
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
        
        #get gradients and max score
        #following code adapted from chirag126 https://github.com/chirag126/VOG/blob/master/toy_script.py

    model.eval() # no dropout, in eval mode
    grad_X=X.type(torch.FloatTensor)
    grad_X = torch.reshape(grad_X, (60000,1,28,28))
# print("evaluating with first 1000 examples")
    grad_X = grad_X[:1]
    """ if get weird errors, make sure this is ok"""
    grad_Y = train_dataset.targets
    grad_Y = grad_Y[:1]
#  print("dones splicing")

    grad_X.requires_grad = True
    # am i doing backprop on the correct thing?
    node_sel = grad_Y.shape
    ones = torch.ones(node_sel) # 60000x1, one for each training example
#  print("feeding forward")
    logits  = model(grad_X) # feed forward, keeping track of the calculation graph
#  print("done with logits")
    print("probs")
    probs = torch.nn.Softmax(dim=1)(logits) # 60000 x 10
    print(probs)
    sel_nodes = probs[torch.arange(len(grad_Y)), grad_Y.type(torch.LongTensor)]
#  print("sel nodes shape")
#  print(sel_nodes.shape)
# print("running gradients")
    sel_nodes.backward(ones) # is this correct?
    print("nodes to do backprop on")
    print(sel_nodes)
    grad = grad_X.grad.data.numpy()
    grad = grad.squeeze()
    print("gradient")
    print(grad)
# print("calculated gradients")
    for i in range(grad_X.shape[0]):
        if i not in vog.keys():
            vog[i] = []
            vog[i].append(grad[i, :].tolist())
        else:
            vog[i].append(grad[i, :].tolist())
    print("vog dictionary")
    print(vog)
    # now we have stored the gradients for each pixel of each image
    training_vog_stats=[]
    training_labels=[]
    training_class_variances = list(list() for i in range(10))
    training_class_variances_stats = list(list() for i in range(10))
    """ CHECK THIS CODE AGAIN SOMETHING GOING WRONG
        maybe with appending array. go through math
    """
    for ii in range(grad_X.shape[0]):
        
        temp_grad = np.array(vog[ii])            
        mean_grad = np.sum(np.array(vog[ii]), axis=0)/len(temp_grad)
        training_vog_stats.append(np.mean(np.sqrt(sum([(mm-mean_grad)**2 for mm in temp_grad])/len(temp_grad))))
        training_labels.append(int(grad_Y[ii].item()))

        training_class_variances[int(grad_Y[ii].item())].append(np.mean(np.sqrt(sum([(mm-mean_grad)**2 for mm in temp_grad])/len(temp_grad))))
    normalized_vog=[]
    for ii in range(grad_X.shape[0]):
        mu = np.mean(training_class_variances[int(grad_Y[ii].item())])
        std  = np.std(training_class_variances[int(grad_Y[ii].item())])
        normalized_vog.append(np.abs((training_vog_stats[ii] - mu)/std))
# print("done with vog")
    normalized_vog=a = np.ma.array(normalized_vog, mask=np.isnan(normalized_vog))
# print(normalized_vog)
# print("highest values")

    #print(normalized_vog)
    # end code adapted from https://github.com/chirag126/VOG/blob/master/toy_script.py
    print("max variances")
   # ind = np.argpartition(normalized_vog, -10)[-100:]
   # print(ind)
#print([ind])    
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