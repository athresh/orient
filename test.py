import torch

broadcast_size = (1000, 100, 10)
x = torch.zeros(1000, 10)
y = torch.eye(100, 10)
x = x.unsqueeze(1).expand(broadcast_size)
y = y.unsqueeze(0).expand(broadcast_size)
dists = torch.sum((x - y)**2, dim=2)

y_src = torch.randint(0, 2, (1000, 1))
y_tgt = torch.randint(0, 2, (100, 1))
print(y_src[0:100] == y_tgt)
y_src = y_src.expand(broadcast_size[0:2])
y_tgt = y_tgt.view(1, -1).expand(broadcast_size[0:2])
y_same = torch.eq(y_src, y_tgt)   # Boolean mask
y_diff = torch.logical_not(y_same)        # Boolean mask
intraclass_dists = dists * y_same   # Set 0 where classes are different
interclass_dists = dists * y_diff   # Set 0 where classes are the same
max_dists = torch.max(dists, dim=1, keepdim=True)[0]
max_dists = max_dists.expand(broadcast_size[0:2])
interclass_dists = torch.where(y_same, max_dists, interclass_dists)
print()