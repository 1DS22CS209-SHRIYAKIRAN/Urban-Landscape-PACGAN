import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
import os
from PIL import Image

class LoadDataset(Dataset):
    """
    Load the dataset.
    """

    def __init__(self, batch_size: int, step: int, root: str, label_root: str, n_devices: int, sel_ids: list = []):
        self.root = root
        image_size = 4 * 2 ** step

        # Load labels from CSV
        labels_data = pd.read_csv(label_root, delimiter=',')
        if sel_ids.size == 0:
            sel_ids = labels_data['ID'].values.astype(int)
        self.labels = labels_data['Class'].values.astype(int)[sel_ids]

        # Load images
        self.img = []
        for i in sel_ids:
            img_path = os.path.join(root, f'Real{image_size}x{image_size}', f'{i}.png')
            image = Image.open(img_path)
            image = np.asarray(image)
            if image.ndim == 2:  # Convert grayscale to RGB
                image = np.stack((image,) * 3, axis=-1)
            self.img.append(image)
        self.img = np.array(self.img)

        # Pytorch dataloader expect image of size N, C, H, W (batch_size, channel, height, width):
        self.img = np.transpose(self.img, (0, 3, 1, 2))

        if n_devices > 1:
            # Check to have a number of images multiple of the batch size (needed for parallelization):
            mod = len(sel_ids) % batch_size
            if mod != 0:
                self.img = self.img[:-mod, :, :, :]
                self.labels = self.labels[:-mod]

    def __getitem__(self, index):
        return self.img[index, :, :, :], self.labels[index]

    def __len__(self):
        return self.img.shape[0]  # image shape: [batch_size x channels x height x width]
