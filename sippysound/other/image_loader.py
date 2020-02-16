import torch
import glob
import numpy as np
from torch.utils.data import Dataset

from PIL import Image

from sippysound import utilz

class Images(Dataset):

    def __init__(self, root_dir, transforms=None):
        """

        """
        self.root_dir = root_dir
        self.fns = glob.glob(self.root_dir + '**.jpg')
        print(f'fns: {self.fns}')
        self.transform = transforms
        self.imgs = get_imgs(self.fns, transform=self.transform)
        self.length = len(self.imgs)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.imgs[idx]


def get_imgs(fns, transform=None):
    imgs = []
    for fn in fns:
        image = np.array(Image.open(fn), dtype=np.uint8)
        sample = torch.from_numpy(image)
        if transform:
            sample = transform(sample)
        imgs.append(sample)
    return imgs
