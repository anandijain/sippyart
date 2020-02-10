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

# larger window sizes wont usually work on my GPU because of the RAM
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

LOG_INTERVAL = 1
BATCH_SIZE = 52
WINDOW_SECONDS = 2  # 1.5  # n
MIDDLE = 400  # 11025 # 22050 # 44100
BOTTLENECK = 300
EPOCHS = 100
START_SAVING_AT = 50
SAVE_FREQ = 1
SHUFFLE = False

RUN_TIME = time.asctime()

MODEL_FN = f'models/n_{WINDOW_SECONDS}_mid_{MIDDLE}_bot_{BOTTLENECK}.pth'

FILE_NAMES = [
    # place file names here
    # '/home/sippycups/Music/2020/81 - 2 8 20.wav'
    # '/home/sippycups/Music/2019/5 14 19 matt anand ilan.wav'
    # '/home/sippycups/Music/2019/81 - 10 18 19.wav'
    # '/home/sippycups/Music/2019/81 - 9 25 19.wav'
    '/home/sippycups/Music/2019/81 - 9 23 19.wav'
]
LOAD_MODEL = False

def train_vae(fn, epochs=EPOCHS, start_saving_at=START_SAVING_AT, save=True, save_model=True, lopass=False):
    d = prep(fn)
    short_fn = utils.full_fn_to_name(fn)
    y_hats = []
    for epoch in range(1, epochs + 1):  # [epochs, 2, n]
        print(f'epoch: {epoch} {short_fn}')
        if epoch < start_saving_at:
            train.train_epoch(d, epoch, BATCH_SIZE, device)
        else:
            train.train_epoch(d, epoch, BATCH_SIZE, device)
            y_hat = utils.gen_recon(d['m'], BOTTLENECK, device)
            y_hats.append(y_hat)

    song = torch.cat(y_hats, dim=1)
    print(song)

    MODEL_FN: f'models/n_{WINDOW_SECONDS}_mid_{MIDDLE}_bot_{BOTTLENECK}.pth'
    if save:
        save_wavfn = f'vaeconv_{short_fn}_{RUN_TIME}.wav'

        torchaudio.save(d['path'] + save_wavfn, song, d['sr'])
    if save_model:
        torch.save(d["m"].state_dict(), MODEL_FN)
    return song


def prep(fn: str, load_model=LOAD_MODEL):
    short_fn = utils.full_fn_to_name(fn)

    path = 'samples/sound/' + short_fn + '/'

    try:
        os.makedirs(path)
    except FileExistsError:
        print(f'warning: going to overwrite {path}')

    dataset = loaders.WaveSet(fn, seconds=WINDOW_SECONDS)
    print(f'len(dataset): {len(dataset)} (num of windows)')
    window_len = dataset.window_len
    sample_rate = dataset.sample_rate
    # BATCH_SIZE = len(dataset) // 2
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    # model = models.VAEConv1d(dim=window_len*2, bottleneck=BOTTLENECK,
    #                          middle=MIDDLE).to(device)

    model = models.VAEConv1d(WINDOW_SECONDS*sample_rate*2, bottleneck=BOTTLENECK).to(device)

    # model = models.VAE(dim=window_len*2, bottleneck=BOTTLENECK,
    #                 middle=MIDDLE).to(device)
    if load_model:
        try:
            model.load_state_dict(torch.load(MODEL_FN))
            print(f'loaded: {MODEL_FN}')
        except FileNotFoundError:
            pass
    print(model)
    optimizer = optim.Adam(model.parameters())
    writer = SummaryWriter(
        f"runs/{short_fn}_{RUN_TIME}")

    d = {
        'm': model,
        'o': optimizer,
        'data': dataset,
        'loader': train_loader,
        'sr': sample_rate,
        'path': path,
        'writer': writer
    }
    return d



def gen_folder(folder="/home/sippycups/Music/2019/"):
    # broken
    fns = glob.glob(f'{folder}/**.wav')
    for i, wavfn in enumerate(fns):
        print(f'{i}: {wavfn}')
        try:
            train_vae(wavfn)
        except RuntimeError:
            continue
       # if i == 5:
        #    break


if __name__ == "__main__":
    for fn in FILE_NAMES:
        train_vae(fn)
    # gen_folder()
