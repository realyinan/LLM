import torch


a = torch.tensor([
    [1, 2, 4],
    [3, 1, 2]
])

print(a.max(dim=-1))
print(a.argmax(dim=-1))
print(a.topk(k=1, dim=-1))