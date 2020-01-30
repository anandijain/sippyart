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


BPM = 110
BEATS_PER_SECOND = BPM // 60 # b/sec
N_LAYERS = 4

FN = '3seconds' 
to_read = f'data/' + FN + '.wav'
wave, SAMPLE_RATE = torchaudio.load(to_read)

WINDOW_SIZE = 440 * 4
HIDDEN_SIZE = WINDOW_SIZE
print(WINDOW_SIZE)
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-1
LOG_FN = f'LSTM_{FN}_WIN_LEN_{WINDOW_SIZE}_N_{N_LAYERS}_BATCH_{BATCH_SIZE}_LR_{LR}'
SAVE_FN = f'samples/{LOG_FN}.wav'


if __name__ == "__main__":
    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = loaders.WavLSTM(wave, SAMPLE_RATE, WINDOW_SIZE)
    x, y = dataset[0]
    print(f'x: {x.shape} y{y.shape}')
    
    print(f'len(dataset): {len(dataset)}')


    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    loss_fn = nn.MSELoss()
    writer = SummaryWriter(f'runs/{LOG_FN}{time.asctime()}')

    # model = nn.LSTM(BATCH_SIZE*2, BATCH_SIZE*2, N_LAYERS).to(device)
    model = models.LSTM(WINDOW_SIZE, WINDOW_SIZE, N_LAYERS, device).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    optimizer.zero_grad()

    hn = torch.randn(N_LAYERS, 1, HIDDEN_SIZE).to(device)
    cn = torch.randn(N_LAYERS, 1, HIDDEN_SIZE).to(device)
    all_outs = []
    for epoch in range(EPOCHS):

        model.reset()
        for i, (x, y) in enumerate(loader):
            try:
                x = x.to(device).view(BATCH_SIZE, 1, -1)
            except RuntimeError:
                print('broke')
                continue
            y = y.to(device).view(BATCH_SIZE, 1, -1)
            try:
                out, (hn, cn) = model(x)
            except RuntimeError:
                continue
            loss = loss_fn(out, y)
            writer.add_scalar('train_loss', loss.item(), i+(len(dataset) * epoch))
            loss.backward()
            # all_outs.append(out.view(2, -1))
            all_outs.append(out)
            print(f'epoch: {epoch} batch_id: {i} loss: {loss} % DONE: {(i * BATCH_SIZE) / len(dataset)}')
            if i >= SAMPLE_RATE * 10:
                break
    big = torch.cat(all_outs)
    torchaudio.save(SAVE_FN, big.view(2, -1).cpu(), 44100)
    torch.save(model, 'models/lstm.pth')
