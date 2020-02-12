import torch
import glob
import numpy as np
from torch.utils.data import Dataset

from PIL import Image

class Images(Dataset):

    def __init__(self, root_dir, transforms=None):
        """

        """
        self.root_dir = root_dir
        self.fns = glob.glob(self.root_dir + '/**.jpg')
        self.length = len(self.fns)
        self.transform = transforms

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        fn = self.fns[idx]
        image = np.array(Image.open(fn), dtype=np.uint8)
        sample = torch.from_numpy(image)

        if self.transform:
            sample = self.transform(sample)

        return sample
