#!/usr/bin/env python3

'''
@Time   : 2019-08-01 22:29:09
@Author : su.zhu
@Desc   : 
'''

import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim

m = nn.Linear(20, 10)

x = torch.randn(2, 20)
y = torch.tensor([1, 9])

params = list(m.parameters())
for param in params:
    #param.requires_grad = False
    print(param.size())
params = list(filter(lambda p: p.requires_grad, params))

optimizer = optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0) # (beta1, beta2)

optimizer.zero_grad()
z = F.log_softmax(m(x), dim=1)
loss = nn.NLLLoss(size_average=False)(z, y)
loss.backward()
for param in params:
    print(param.grad)
#optimizer.step()
