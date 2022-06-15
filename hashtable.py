import torch
import random
images = {}

logits = torch.tensor([[1., 2.3, 3., 4., -1.], # 4 - 2.3 = 1.7
                       [1.1, 4.2, -2., 0., 0.6], # 4.2 - 1.1 = 3.1
                       [0.2, 0.2, 0.2, 0.18, 0.22]]) # 0.2 - 0.22 = -0.02
targets = torch.tensor([3,1,2])

print(logits.shape, targets.shape)

target_values = logits.gather(1, targets.view(-1, 1)).squeeze()
print(target_values)
print(target_values.shape)

masked_logits = torch.scatter(logits, 1, targets.view(-1, 1), float('-inf'))
print(masked_logits)
print(masked_logits.shape)
other_logit_values, _ = masked_logits.max(1)
other_logit_values = other_logit_values.squeeze()

margin_values = (target_values - other_logit_values).tolist()
print(margin_values)









"""
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
"""