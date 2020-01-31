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

import matplotlib.pyplot as plt
import time
from torch.utils.tensorboard import SummaryWriter

from glob import glob

import numpy as np

import loaders
import models
import utils


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

if __name__ == "__main__":
    DATA_PATH = 'data/images'
    T = time.asctime()
    MIDDLE = 250
    BOTTLENECK = 150
    EPOCHS = 200
    # HEIGHT = 1024 #3024
    # WIDTH = 1024 #3024
    # CHANNELS = 3
    images = loaders.Images(DATA_PATH)
    data = images[0]
    print(data.shape)
    HEIGHT, WIDTH, CHANNELS = data.shape
    dim = data.flatten().shape[0]

    writer = SummaryWriter(
        f"runs/image_gen_test_MID_{MIDDLE}_BOTTLE_{BOTTLENECK}_{T}")
    os.mkdir(f'samples/images/image_gen_{T}')
    
    model = models.VAE(dim, middle=MIDDLE, bottleneck=BOTTLENECK).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    samples = []
    for epoch in range(EPOCHS):
        for i, data in enumerate(images):
            print(f'data.shape pre: {data.shape}')
            data = data.flatten().float().to(device) / 256
            print(f'data.shape post: {data.shape}')
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)
            print(f'recon: {np.unique(recon_batch.cpu().detach().numpy())}')
            print(f'x: {np.unique(data.cpu().detach().numpy())}')
            loss = utils.kl_loss(recon_batch, data, mu, logvar)
            loss.backward()

            print(f'loss: {loss}')
            writer.add_scalar('train_loss', loss.item(), global_step=i + epoch)
            idx = len(images) * epoch + i
            optimizer.step()
            with torch.no_grad():
                sample = torch.randn(1, BOTTLENECK).to(device)
                sample = model.decode(sample).cpu()
                sample = sample.view(HEIGHT, WIDTH, CHANNELS)
                # img = Image.fromarray(sample[0][i].transpose(
                #     0, 2).numpy().astype(np.uint8))
                print(f'sample: {sample}')
                print(f'sample: {sample.shape}')
                print(f'sample: {sample.dtype}')
                # img = torch.tensor(sample * 255, dtype=torch.uint8)
                scipy.misc.imsave(f'samples/images/image_gen_{T}/img_{i+epoch}.png', sample.numpy())
                samples.append(sample)


    video = torch.cat(samples).view(-1, HEIGHT, WIDTH, CHANNELS) * 255
    print(f'video.shape: {video.shape}')
    torchvision.io.write_video('tv_test.mp4', video, 60)
