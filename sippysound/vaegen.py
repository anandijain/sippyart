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

from sippysound import utilz
from sippysound import models
from sippysound import loaders
from sippysound import train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

RUN_TIME = time.asctime()

BATCH_SIZE = 139
WINDOW_SECONDS = 1  # larger window sizes wont usually work on my GPU because of the RAM
BOTTLENECK = 250

# start_saving should be less than epochs 
EPOCHS = 100
START_SAVING_AT = 50

START_FRAC = 0
END_FRAC = 1

SAVE_FREQ = 1
LOG_INTERVAL = 1
SHUFFLE = False

MODEL_FN = f'{utilz.PARENT_DIR}models/2n_2.pth'
LOAD_MODEL = False
SAVE_MODEL = True
SAVE_SONG = True
# LR = 1e-3
LR = None

SAVE_RUN = False
USE_LOGGER = False

FILE_NAMES = [
    # place file names here
    '/home/sippycups/Music/81 - intro to ableton/81 - intro to ableton - 88 bye 3.wav'
    # '/home/sippycups/Music/misc/81 - misc - 25 mini 12-39 am.wav',
    # # '/home/sippycups/Music/misc/81 - misc - 27 12 31 17.wav'
    # '/home/sippycups/Music/2018/81 - 2018 - 07 part 2 not done.wav'
]

def train_vae(fns:list):
    d = prep(fns)
    y_hats = []

    for epoch in range(1, EPOCHS + 1):
        print(f'epoch: {epoch}')
        
        if epoch < START_SAVING_AT:
            train.train_epoch(d, epoch, BATCH_SIZE, device)
        else:
            train.train_epoch(d, epoch, BATCH_SIZE, device)
            y_hat = utilz.gen_recon(d['m'], BOTTLENECK, device)
            y_hats.append(y_hat)

    song = torch.cat(y_hats, dim=1)
    print(song)

    if SAVE_SONG:
        save_wavfn = f'vaeconv_{RUN_TIME}.wav'
        song_path = d['path'] + save_wavfn
        torchaudio.save(song_path, song, d['sr'])
        print(f'audio saved to {song_path}')

    if SAVE_MODEL:
        torch.save(d["m"].state_dict(), MODEL_FN)
        print(f'model saved to {MODEL_FN}')

    return song


def prep(fns: list):
    # short_fn = utilz.full_fn_to_name(fn)

    path = utilz.PARENT_DIR + 'samples/sound/'
    utilz.make_folder(path)

    dataset = loaders.WaveSet(
        fns, seconds=WINDOW_SECONDS, start_pct=START_FRAC, end_pct=END_FRAC)

    print(f'len(dataset): {len(dataset)} (num of windows)')
    print(f'sample_rate：{dataset.sample_rate}')
    # bs = len(dataset)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE) ##### !!!

    model = models.VAEConv1d(WINDOW_SECONDS*dataset.sample_rate*2,
                             bottleneck=BOTTLENECK).to(device)

    if LOAD_MODEL:
        model = utilz.load_model(model, MODEL_FN)

    print(model)
    if LR is None:
        optimizer = optim.Adam(model.parameters())
    else:
        optimizer = optim.Adam(model.parameters(), lr=LR)
    if USE_LOGGER:
        writer = SummaryWriter(
            f"runs/{RUN_TIME}")
    else:
        writer = None
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
    train_vae(FILE_NAMES)
    # gen_folder()
