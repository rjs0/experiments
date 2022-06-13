import torch
import torch.nn as nn
X = torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
Y = torch.tensor([[3],[6],[9],[12]],dtype=torch.float32)

# f(x) = 3x 


n_samples, n_features = X.shape
model = nn.Linear(n_features,n_features)

lrate = 0.01
n_iters=1000
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=lrate)
for epoch in range(n_iters):
    pred = model(X)
    l = loss(pred,Y)
    l.backward()

    optimizer.step()
    optimizer.zero_grad()
    [w,b]=model.parameters()
    print(w)