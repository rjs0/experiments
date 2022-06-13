from numpy import double
import torch


x=torch.ones(3,requires_grad=True)
y=x+3
z=y*y
print(x)
z=z.mean()
print(z)
z.backward() # dz/dx = dz/dy * dy/dz = (2y * 1)/3 = 8/3=2.67

print(x.grad)