import torch


a = torch.tensor([
    [2.3, 4.6],
    [1.5, 5.7]
])

print(a.max(dim=-1))

