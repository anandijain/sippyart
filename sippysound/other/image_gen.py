import os
import scipy.misc
import time
from PIL import Image
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import skvideo.io
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from torch.utils.tensorboard import SummaryWriter

from glob import glob

import numpy as np

import image_loader
from sippysound import models
from sippysound import utilz


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
MODEL_FN = f'{utilz.PARENT_DIR}models/conv2d.pth'
DATA_PATH = 'data/images'

TIME = time.asctime()

HEIGHT, WIDTH = 256, 256
CHANNELS = 3

MIDDLE = 288
BOTTLENECK = 288

EPOCHS = 500
BATCH_SIZE = 1

# LR = 1e-2
LR = None

SAVE_MODEL = True
LOAD_MODEL = True

USE_LOGGER = False

edits = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor()])

def prep():

    images_path = f'{utilz.PARENT_DIR}data/images/'
    print(images_path)
    images = image_loader.Images(
        images_path, transforms=edits)
        
    data = images[0]
    dataloader = DataLoader(images, shuffle=True, batch_size=BATCH_SIZE)

    print(data.shape)
    dim = data.flatten().shape[0]
    if USE_LOGGER:
        writer = SummaryWriter(
            f"runs/image_gen_test_MID_{MIDDLE}_BOTTLE_{BOTTLENECK}_{TIME}")
    else:
        writer = None
    
    model = models.VAEConv2d(dim, middle=MIDDLE, bottleneck=BOTTLENECK).to(device)
    print(model)

    if LOAD_MODEL:
        model = utilz.load_model(model, MODEL_FN)

    if LR is not None:
        optimizer = optim.Adam(model.parameters(), lr=LR)
    else:
        optimizer = optim.Adam(model.parameters())

    write_to = f'samples/images/image_gen_{TIME}'

    os.makedirs(write_to)
    d = {
        'write_to': write_to,
        'writer': writer,
        'dataloader': dataloader,
        'model': model,
        'optimizer': optimizer,
        'set': images,
        'model_fn': MODEL_FN
    }
    return d

def train(d):

    samples = []
    for epoch in range(EPOCHS):
        for i, data in enumerate(d['dataloader']):
            data = data.float().to(device) / 255
            d['optimizer'].zero_grad()
            # print(f'data shape : {data.shape}')
            data = data.view(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
            recon_batch, mu, logvar = d['model'](data)
            recon_batch = recon_batch.view(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
            loss = utilz.kl_loss(recon_batch, data, mu, logvar)
            loss.backward()

            idx = len(d['set']) * epoch + i

            if d['writer'] is not None:
                d['writer'].add_scalar('train_loss', loss.item(), global_step=idx)
            
            
            if i % 500 == 0:
                print(f'{epoch} {idx}: {loss}')
            #     print(
            #         f'recon: {np.unique(recon_batch.cpu().detach().numpy())}')
            #     print(f'x: {np.unique(data.cpu().detach().numpy())}')
            d['optimizer'].step()

        with torch.no_grad():
            sample = torch.randn(1, BOTTLENECK).to(device)
            sample = d['model'].decode(sample).cpu()
            sample = sample.view(HEIGHT, WIDTH, CHANNELS)
            scipy.misc.imsave(
                f'samples/images/image_gen_{TIME}/img_{idx}.png', sample.numpy())
            samples.append(sample)

    video = torch.cat(samples).view(-1, HEIGHT, WIDTH, CHANNELS) * 255
    print(f'video.shape: {video.shape}')

    video_path = f'{utilz.PARENT_DIR}samples/videos/{TIME}.mp4'
    torchvision.io.write_video(video_path, video, 60)

    torch.save(d["model"].state_dict(), d['model_fn'])

if __name__ == "__main__":
    d = prep()
    # print(d)
    train(d)
