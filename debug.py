import torch

def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))

def masker(tensor,shape,m):
    temp=tensor.abs()
    temp=-temp #now the k top values are those with the smallest magnitude
    res = torch.topk(temp.view(-1), k=m)
    idx = unravel_index(res.indices, temp.size())
    tensor[idx]=0
    return tensor

x = torch.randn(2, 3, 4)
print(x)
x = masker(x,x.shape,3)
print(x)
#print(x[idx] == res.values)