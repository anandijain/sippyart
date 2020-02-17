"""

"""
import glob
import os
import random
import time

import torch
import torchaudio
import torchvision

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

BATCH_SIZE = 1
WINDOW_SECONDS = 1  # larger window sizes wont usually work on my GPU because of the RAM
BOTTLENECK = 300

# 3, 10 10 

# start_saving should be less than epochs
EPOCHS = 25
START_SAVING_AT = 0

START_FRAC = 0
END_FRAC = .30

SAVE_FREQ = 1
LOG_INTERVAL = 1
SHUFFLE = True

MODEL_FN = f'{utilz.PARENT_DIR}models/10n_2.pth'
LOAD_MODEL = True
SAVE_MODEL = True

SAVE_SONG = True
SAVE_VIDEO = True

# LR = 1e-3
LR = None

SAVE_RUN = False
USE_LOGGER = False

USE_GEN_APPLY = True

FILE_NAMES = [
    # place train files here
    '/home/sippycups/audio/data/sound/sentimental.wav',
    # '/home/sippycups/audio/data/sound/ascenseur.wav',

]


GEN_APPLY_FNS = [
    # test files here, used only if USE_GEN_APPLY is True
    '/home/sippycups/audio/data/sound/ascenseur.wav',
] 


def train_vae(fns: list):
    d = prep(fns)
    y_hats = []
    applyset_len = len(d['applyset'])
    all_zs = []
    for epoch in range(1, EPOCHS + 1):
        print(f'epoch: {epoch}')

        if epoch < START_SAVING_AT:
            train.train_epoch(d, epoch, BATCH_SIZE, device)
        else:
            all_zs.append(train.train_epoch(d, epoch, BATCH_SIZE, device))

            if USE_GEN_APPLY:
                apply_idx = random.randint(0, applyset_len)
                sample = d['applyset'][apply_idx].view(BATCH_SIZE, 2, -1)
                y_hat = utilz.gen_apply(d['m'], sample, device).cpu()
            else:
                y_hat = utilz.gen_recon(d['m'], BOTTLENECK, device)

            y_hats.append(y_hat)
    video = torch.cat(all_zs, dim=0)
    song = torch.cat(y_hats, dim=1)
    print(f'song: {song}')
    print(f'video.shape: {video.shape}')

    if SAVE_SONG:
        save_wavfn = f'vaeconv_{RUN_TIME}.wav'
        song_path = d['path'] + save_wavfn
        torchaudio.save(song_path, song, d['sr'])
        print(f'audio saved to {song_path}')

    if SAVE_MODEL:
        torch.save(d["m"].state_dict(), MODEL_FN)
        print(f'model saved to {MODEL_FN}')

    if SAVE_VIDEO:
        video_path = f'{utilz.PARENT_DIR}samples/videos/vaegen{RUN_TIME}.mp4'
        print(f'video shape: {video.shape}')
        video = video.view(-1, 10, 10, 3)
        torchvision.io.write_video(video_path, video, 60)
        print(f'audio saved to {video_path}')

    return song



def test_vae(test_fns):
    """
    Used with a trained model, where MODEL_FN is already a saved .pth file


    """
    d = prep(test_fns)
    y_hats = []
    length = len(d['data'])
    img_y_hats = []
    for epoch in range(1, EPOCHS + 1):
        print(f'test epoch: {epoch}')

        train.train_epoch(d, epoch, BATCH_SIZE, device)
        apply_idx = random.randint(0, length)
        sample = d['data'][apply_idx].view(BATCH_SIZE, 2, -1)
        y_hat = utilz.gen_apply(d['m'], sample, device).cpu()
        print(f'y_hat.shape: {y_hat.shape}')

        img_vers = y_hat.view(1, 3, 240, 245) # for 2 seconds
        img_y_hats.append(img_vers)

        y_hats.append(y_hat)

    song = torch.cat(y_hats, dim=1)
    print(song)

    if SAVE_SONG:
        save_wavfn = f'vaeconv_{RUN_TIME}.wav'
        song_path = d['path'] + save_wavfn
        torchaudio.save(song_path, song, d['sr'])
        print(f'audio saved to {song_path}')


    return song# , video


def prep(fns: list):
    """
    Prepares a dictionary containing the necesary items for training.

    
    """
    path = utilz.PARENT_DIR + 'samples/sound/'
    utilz.make_folder(path)

    dataset = loaders.WaveSet(
        fns, seconds=WINDOW_SECONDS, start_pct=START_FRAC, end_pct=END_FRAC)

    apply_set = loaders.WaveSet(GEN_APPLY_FNS, seconds=WINDOW_SECONDS)

    print(f'len(dataset): {len(dataset)} (num of windows)')
    print(f'sample_rate：{dataset.sample_rate}')

    train_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)  # !!!

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
        'applyset': apply_set,
        'loader': train_loader,
        'sr': dataset.sample_rate,
        'path': path,
        'writer': writer
    }
    return d


if __name__ == "__main__":
    train_vae(FILE_NAMES)
    # test_vae(FILE_NAMES)
