from pickletools import optimize
from this import d
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
#from swapper import swap_label
# if gpu is available, use GPU
proportion=0.008

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
num_epochs = 15
batch_size = 100
eval_size = 60000
learning_rate = 0.001

#load MNIST
train_dataset = torchvision.datasets.MNIST(root='./data',train=True
    ,transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root='./data',train=False
    ,transform=transforms.ToTensor())
train_dataset, swap_table = swap_data(train_dataset, proportion)

#print(swap_table.values())
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
       # x = self.dropout(x)
        pred = self.linear_2(x)
        return pred
    
model = NeuralNet(num_classes)  
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# training loop

n_total_steps = len(train_loader)

X, Y = train_dataset.data, train_dataset.targets
print(Y[0])
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
    grad_X = grad_X[:60000]
   # print("images:")
    #print(grad_X[0])
    """ if get weird errors, make sure this is ok"""
    grad_Y = train_dataset.targets
    grad_Y = grad_Y[:60000] # number of training examples
    print("labels")
    #print(grad_Y)
#  print("dones splicing")

    grad_X.requires_grad = True
    # am i doing backprop on the correct thing?
    node_sel = grad_Y.shape
    ones = torch.ones(node_sel) # 60000x1, one for each training example
#  print("feeding forward")
    logits  = model(grad_X) # feed forward, keeping track of the calculation graph
#  print("done with logits")
   # print("probs")
    #print("logit shape")
   # print(logits.shape)
  #  print(logits)
   # probs = torch.nn.Softmax(dim=1)(logits) # 60000 x 10
    #print(probs)
    sel_nodes = logits[torch.arange(len(grad_Y)), grad_Y.type(torch.LongTensor)] # changed from probs to logits
   # print("sel nodes shape")
    #print(sel_nodes.shape)
   # print(sel_nodes)
# print("running gradients")
    sel_nodes.backward(ones) # is this correct?
   # print("nodes to do backprop on")
   # print(sel_nodes)
    grad = grad_X.grad.data.numpy()
   # grad = grad.squeeze()
    print("gradient")
   # print(grad.shape)
   # print(grad[3]) # RUN THIS, see if it is really just 0s
# print("calculated gradients")
    for i in range(grad_X.shape[0]):

        #vog[i] = n x 1 x 28 x 28
        if i not in vog.keys():
            vog[i] = []
            vog[i].append(grad[i, :].tolist())
            
        else:
            vog[i].append(grad[i, :].tolist())
       
    #print("vog dictionary")
    #print(vog)
    # now we have stored the gradients for each pixel of each image
    training_vog_stats=[]
    training_labels=[]
    training_class_variances = list(list() for i in range(10))
    training_class_variances_stats = list(list() for i in range(10))
    """ CHECK THIS CODE AGAIN SOMETHING GOING WRONG
        maybe with appending array. go through math
    """
    for ii in range(grad_X.shape[0]):
        
        temp_grad = np.array(vog[ii])    # this will be an t x 1 x 28 x 28 size, where t = epoch number     
        mean_grad = np.sum(np.array(vog[ii]), axis=0)/len(temp_grad)  # 1 x 28 x 28, avg of previous timesteps (2)
       # if(ii==0):
          #  print("appending to vog")
          #  print(np.mean(np.sqrt(sum([(mm-mean_grad)**2 for mm in temp_grad])/len(temp_grad))))
        variance = np.mean(np.sqrt(sum([(mm-mean_grad)**2 for mm in temp_grad])/len(temp_grad)))
        training_vog_stats.append(variance) # scalar represnting (3) w sqrt
        training_labels.append(int(grad_Y[ii].item())) # append corresponding class label

        # training_class_varainces [class]  = [var 1, var 2, ... , var k], if there are k instances of this class in the dataset
        training_class_variances[int(grad_Y[ii].item())].append(variance)
    normalized_vog=[]
    for ii in range(grad_X.shape[0]):
        mu = np.mean(training_class_variances[int(grad_Y[ii].item())])
        std  = np.std(training_class_variances[int(grad_Y[ii].item())])
        normalized_vog.append(np.abs((training_vog_stats[ii] - mu)/std))
# print("done with vog")
    normalized_vog= np.ma.array(normalized_vog, mask=np.isnan(normalized_vog))
    print(normalized_vog)
# print(normalized_vog)
# print("highest values")

    #print(normalized_vog)
    # end code adapted from https://github.com/chirag126/VOG/blob/master/toy_script.py
    print("max variances")
    ind = np.argpartition(normalized_vog, -1000)[-1000:]
    print("NUMBER OF SWAPPED IN TOP")
    ind=np.array(ind)
    print((ind<481).sum())
    
    num_row=10
    num_col=10
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    count=0
   # for i in ind:
   #     x = train_dataset[i][0]
    #    ax = axes[count//num_col, count%num_col]
   #     ax.imshow(x.reshape(28,28), cmap='gray')
    #    ax.set_title('Label: {}'.format(train_dataset.targets[i]))
    #    count+=1
   # plt.tight_layout()
    #plt.show()
    print(ind)
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