import torch
import random
images = {}

# x have id 1, swapped to 4
# y has id 2, swapped to 5
# z has id 9, swapped to 7
targets = [1,2,9]
x = torch.randn(5,5)
y = torch.randn(5,5)
z = torch.randn(5,5)
w = torch.randn(5,5)
train_data = [x,y,z]

images[hash(x)]=(1,4)
images[hash(y)]=(2,5)
images[hash(z)]=(9,7)

indices = list(range(0,100))
random.shuffle(indices)
print(indices[:5])
i=1
for i in range(0,60000):
    i+=hash(str(i))
print(i)