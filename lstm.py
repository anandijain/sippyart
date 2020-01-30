import glob
import os
import time

import torch
import torchaudio

from torch import nn, optim
from torch.nn import functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils
import loaders
import models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


BATCH_SIZE = 64
N_LAYERS = 4
HIDDEN_SIZE = 64
LR = 1e-2


dataset = loaders.WavLSTM()
x, y = dataset[0]
loader = DataLoader(dataset, batch_size=BATCH_SIZE)
loss_fn = nn.MSELoss()
writer = SummaryWriter(f'runs/{time.asctime()}')

model = nn.LSTM(BATCH_SIZE*2, BATCH_SIZE*2, N_LAYERS).to(device)
optimizer = optim.SGD(model.parameters(), lr=LR)

optimizer.zero_grad()

h = torch.randn(N_LAYERS, 1, BATCH_SIZE*2).to(device)
c = torch.randn(N_LAYERS, 1, BATCH_SIZE*2).to(device)

all_outs = []
for i, (x, y) in enumerate(loader):
    x = x.to(device)
    y = y.to(device)
    print(f'x{x} y{y}')
    out, (h2, c2) = model(x.view(1, 1, -1), (h, c))
    loss = loss_fn(out, y.view(1, 1, -1))
    loss.backward()
    if i % 500 == 0:
        all_outs.append(out.view(2, -1))
        print(f'{i}: {loss}')
    if i >= 44100 * 50:
        break

big = torch.cat(all_outs, dim=1)
torchaudio.save('test_lstm.wav', big.view(2, -1), 44100)
torch.save(model, 'models/lstm.pth')
