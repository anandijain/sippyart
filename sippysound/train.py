import torch
import torchaudio

from sippysound import utilz


def train_epoch(d, epoch: int, batch_size, device, save=False):
    length = len(d['data'])
    
    samples = []
    d['m'].train()
    train_loss = 0
    for batch_idx, data in enumerate(d['loader']):
        data = data.to(device)
        d['o'].zero_grad()

        recon_batch, mu, logvar = d['m'](data)
        recon_batch = recon_batch.view(batch_size, 2, -1)

        loss = utilz.kl_loss(recon_batch, data, mu, logvar)
        loss.backward()
        idx = length * epoch + batch_idx
        
        if d['writer'] is not None:
            d['writer'].add_scalar('train_loss', loss.item(), global_step=idx)

        train_loss += loss.item()
        d['o'].step()
