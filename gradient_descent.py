import torch
X = torch.tensor([1,2,3,4],dtype=torch.float32)
Y = torch.tensor([3,6,9,12],dtype=torch.float32)

# f(x) = 3x 

w = torch.tensor(0.0,dtype=torch.float32, requires_grad=True)

def forward(w,x):
    return w*X

def loss(y_hat,y):
    return ((y_hat-y)**2).mean()


lr = 0.01
n_iters=1000
for epoch in range(n_iters):
    pred = forward(w,X)
    l = loss(pred,Y)
    l.backward()

    with torch.no_grad():
        w -= lr*w.grad
    w.grad.zero_()
    print(w)