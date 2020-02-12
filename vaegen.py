"""

"""
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
import models
import loaders
import train


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

RUN_TIME = time.asctime()

BATCH_SIZE = 15
WINDOW_SECONDS = 2  # larger window sizes wont usually work on my GPU because of the RAM
BOTTLENECK = 500

# start_saving should be less than epochs 
EPOCHS = 25
START_SAVING_AT = 0

START_FRAC = 0
END_FRAC = 0.20

SAVE_FREQ = 1
LOG_INTERVAL = 1
SHUFFLE = False

MODEL_FN = 'n_2.pth'
LOAD_MODEL = False
SAVE_MODEL = True
# LR = 0.1

FILE_NAMES = [
    # place file names here
    '/home/sippycups/Music/2020/81 - 2 8 20.wav'
    # '/home/sippycups/Music/2019/81 - 4 3 19.wav'
    # '/home/sippycups/Music/misc/81 - misc - 18 9 13 17.wav'
    # '/home/sippycups/Music/misc/81 - misc - 11 6 30 17 2.wav'

]

def train_vae(fn, epochs=EPOCHS, start_saving_at=START_SAVING_AT, save_song=True, save_model=SAVE_MODEL):
    d = prep(fn)
    short_fn = utils.full_fn_to_name(fn)
    y_hats = []

    for epoch in range(1, epochs + 1):
        print(f'epoch: {epoch} {short_fn}')
        if epoch < start_saving_at:
            train.train_epoch(d, epoch, BATCH_SIZE, device)
        else:
            train.train_epoch(d, epoch, BATCH_SIZE, device)
            y_hat = utils.gen_recon(d['m'], BOTTLENECK, device)
            y_hats.append(y_hat)

    song = torch.cat(y_hats, dim=1)
    print(song)

    if save_song:
        save_wavfn = f'vaeconv_{short_fn}_{RUN_TIME}.wav'
        torchaudio.save(d['path'] + save_wavfn, song, d['sr'])

    if save_model:
        torch.save(d["m"].state_dict(), MODEL_FN)

    return song


def prep(fn: str):
    short_fn = utils.full_fn_to_name(fn)

    path = 'samples/sound/' + short_fn + '/'
    utils.make_folder(path)

    dataset = loaders.WaveSet(
        fn, seconds=WINDOW_SECONDS, start_pct=START_FRAC, end_pct=END_FRAC)

    print(f'len(dataset): {len(dataset)} (num of windows)')
    print(f'sample_rate：{dataset.sample_rate}')
    # bs = len(dataset)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE) ##### !!!

    model = models.VAEConv1d(WINDOW_SECONDS*dataset.sample_rate*2,
                             bottleneck=BOTTLENECK).to(device)

    if LOAD_MODEL:
        model = utils.load_model(model, MODEL_FN)

    print(model)

    optimizer = optim.Adam(model.parameters())  #, lr=LR)

    writer = SummaryWriter(
        f"runs/{short_fn}_{RUN_TIME}")

    d = {
        'm': model,
        'o': optimizer,
        'data': dataset,
        'loader': train_loader,
        'sr': dataset.sample_rate,
        'path': path,
        'writer': writer
    }
    return d


if __name__ == "__main__":
    for fn in FILE_NAMES:
        train_vae(fn)
    # gen_folder()
