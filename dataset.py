import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data import (
    name_label_dict,
    n_class,
    load_4ch_image,
    load_4ch_image_train,
    load_4ch_image_test,
    Stats,
    train_image_dir,
    test_image_dir
)


class HPADataset(Dataset):
    def __init__(self, df, size=(256, 256),
                 use_transform=True, use_augmentation=True,
                 image_db=None):
        self.df = df
        self.size = size
        self.use_transform = True
        self.use_augmentation = use_augmentation
        self.image_db = image_db

        if self.use_augmentation:
            print('Augmentation is enabled')
            self.transformer = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Resize(self.size),
                transforms.ToTensor(),
                transforms.Normalize(Stats.mean, Stats.std)
            ])
        else:
            self.transformer = transforms.Compose([
                transforms.Resize(self.size),
                transforms.ToTensor(),
                transforms.Normalize(Stats.mean, Stats.std)
            ])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        labels = [int(label) for label in self.df.iloc[i]['Target'].split(' ')]
        labels = np.eye(n_class, dtype=np.float32)[labels].sum(axis=0)

        image_id = self.df.iloc[i]['Id']
        if self.image_db:
            im = self.image_db[f'train/{image_id}'].value
            im = Image.fromarray(im)
        else:
            im = load_4ch_image_train(image_id)

        if self.use_transform:
            im = self.transformer(im)

        return im, labels


class HPATestDataset(Dataset):
    def __init__(self, df, size=(256, 256), image_db=None):
        self.df = df
        self.size = size
        self.use_transform = True
        self.image_db = image_db

        self.transformer = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(Stats.mean, Stats.std)
        ])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        image_id = self.df.iloc[i]['Id']
        if self.image_db:
            im = self.image_db[f'test/{image_id}'].value
            im = Image.fromarray(im)
        else:
            im = load_4ch_image_test(image_id)
        if self.use_transform:
            im = self.transformer(im)

        return im


class HPARawDataset(Dataset):
    def __init__(self, df, base_dir=train_image_dir):
        self.df = df
        self.base_dir = base_dir

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        image_id = self.df.iloc[i]['Id']
        im = load_4ch_image(self.base_dir, image_id)
        return im