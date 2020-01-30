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
dataset = loaders.WavLSTM()


BATCH_SIZE = 64
N_LAYERS = 4
HIDDEN_SIZE = BATCH_SIZE
LR = 1e-2
SAMPLE_RATE = dataset.sr

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
    out, (h, c) = model(x.view(1, 1, -1), (h, c))
    loss = loss_fn(out, y.view(1, 1, -1))
    writer.add_scalar('train_loss', loss.item(), i)
    loss.backward()
    if i % (SAMPLE_RATE // 64) == 0:
        all_outs.append(out.view(2, -1))
        print(f'{i}: {loss}')
    if i >= SAMPLE_RATE * 10:
        break

big = torch.cat(all_outs, dim=1)
torchaudio.save('test_lstm.wav', big.view(2, -1), 44100)
torch.save(model, 'models/lstm.pth')
