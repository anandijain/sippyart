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

# larger window sizes wont usually work on my GPU because of the RAM
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

#CUTOFF_FREQ = 12000
LOG_INTERVAL = 1

BATCH_SIZE = 64
WINDOW_SECONDS = 1.5  # n
MIDDLE = 300  # 11025 # 22050 # 44100
BOTTLENECK = 200
EPOCHS = 50
START_SAVING_AT = 0
SAVE_FREQ = 1

# works on gpu
CONFIG_1 = {
    'WINDOW_SECONDS': 2,  # n
    'MIDDLE': 300,
    'BOTTLENECK': 200,
}

CONFIG_2 = {
    'WINDOW_SECONDS' : 1.1,  # n
    'MIDDLE' : 600,  
    'BOTTLENECK' : 500,
}
CONFIG_3 = {
    'WINDOW_SECONDS' : 3,  # n
    'MIDDLE' : 215,  
    'BOTTLENECK' : 100,
}

MODEL_FN = f'n_{WINDOW_SECONDS}_mid_{MIDDLE}_bot_{BOTTLENECK}.pth'

FILE_NAMES = [
    '/home/sippycups/Music/2018/81 - 2018 - 70 5 9 18.wav'
    # '/home/sippycups/audio/data/3seconds.wav'
    # '/home/sippycups/Music/2018/81 - 2018 - 119 8 5 18.wav'
    # '/home/sippycups/Music/2018/81 - 2018 - 92 6 13 18 2.wav',
    # '/home/sippycups/Music/2018/81 - 2018 - 09 2 18 18.wav',
    # '/home/sippycups/Music/2018/81 - 2018 - 46 4 26 18 v2.wav'
    # '/home/sippycups/Music/misc/81 - misc - 23 12 27 17 v2.wav'
    # '/home/sippycups/Music/misc/81 - misc - 26 12 29 17 hollow pt hiking v2.wav'
    # '/home/sippycups/Music/2018/81 - 2018 - 41 self love soup loop.wav',
    # '/home/sippycups/Music/2018/81 - 2018 - 38 4 10 18.wav',
    # '/home/sippycups/Music/2018/81 - 2018 - 37 4 7 18.wav',
    # '/home/sippycups/Music/2018/81 - 2018 - 31 4 3 18 perhaps a light tune to brighten the mood.wav',
    # '/home/sippycups/Music/2018/81 - 2018 - 15 3 18 18 pt 2 music vid syncup.wav',
    # '/home/sippycups/Music/2018/81 - 2018 - 14 3 18 18 pt 1 music vid syncup.wav',
    # '/home/sippycups/Music/2018/81 - 2018 - 12 3 16 18.wav',
    # '/home/sippycups/Music/2019/81 - 9 21 19 2.wav'
]
LOAD_MODEL = False
# RESAMPLE_RATE = 44100 # TODO



def train_epoch(d, epoch: int, save=False):

    model = d['m']
    optimizer = d['o']
    dataset = d['data']
    train_loader = d['loader']
    sample_rate = d['sr']
    path = d['path']

    samples = []
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)

        loss = utils.loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        idx = len(dataset) * epoch + batch_idx

        d['writer'].add_scalar('train_loss', loss.item(), global_step=idx)

        train_loss += loss.item()
        optimizer.step()
        with torch.no_grad():
            sample = torch.randn(1, BOTTLENECK).to(device)
            sample = model.decode(sample).cpu()
            samples.append(sample.view(2, -1))
    return torch.cat(samples, dim=0)  # to stereo


def prep(fn: str, load_model=LOAD_MODEL):
    short_fn = utils.full_fn_to_name(fn)

    path = 'samples/' + short_fn + '/'

    try:
        os.makedirs(path)
    except FileExistsError:
        print(f'warning: going to overwrite {path}')

    dataset = loaders.WaveSet(fn, seconds=WINDOW_SECONDS)
    print(f'len(dataset): {len(dataset)} (num of windows)')
    window_len = dataset.window_len
    sample_rate = dataset.sample_rate

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = models.VAEConv(dim=window_len*2, bottleneck=BOTTLENECK,
                    middle=MIDDLE).to(device)
    if load_model:
        try:
            model.load_state_dict(torch.load(MODEL_FN))
            print(f'loaded: {MODEL_FN}')
        except FileNotFoundError:
            pass
    print(model)
    optimizer = optim.Adam(model.parameters())
    writer = SummaryWriter(
        f"runs/{short_fn}_n_{WINDOW_SECONDS}_{time.asctime()}")

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


def train(fn, epochs=EPOCHS, start_saving_at=START_SAVING_AT, save=True, save_model=True, lopass=False):
    d = prep(fn)
    short_fn = utils.full_fn_to_name(fn)
    y_hats = []
    for epoch in range(1, epochs + 1):  # [epochs, 2, n]
        print(f'epoch: {epoch} {short_fn}')
        if epoch < start_saving_at:
            train_epoch(d, epoch)
        else:
            y_hat = train_epoch(d, epoch)
            y_hats.append(y_hat.view(2, -1))

    song = torch.cat(y_hats, dim=1)
    print(song)

    if save:
        save_wavfn = f'vaeconv_{short_fn}_n_{WINDOW_SECONDS}_mid_{MIDDLE}_bot_{BOTTLENECK}.wav'

        torchaudio.save(d['path'] + save_wavfn, song, d['sr'])
    if save_model:
        torch.save(d["m"].state_dict(), MODEL_FN)
    return song




def gen_folder(folder="/home/sippycups/Music/2019/"):
    # broken
    fns = glob.glob(f'{folder}/**.wav')
    for i, wavfn in enumerate(fns):
        print(f'{i}: {wavfn}')
        try:
            train(wavfn)
        except RuntimeError:
            continue
       # if i == 5:
        #    break


if __name__ == "__main__":
    for fn in FILE_NAMES:
        train(fn)
    # gen_folder()
