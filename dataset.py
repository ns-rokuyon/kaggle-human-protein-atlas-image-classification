import torch
import h5py
import numpy as np
import albumentations as alb
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


from data import *


class HPAEnhancedDatasetMP(Dataset):
    def __init__(self, df, size=(512, 512),
                 use_augmentation=True,
                 use_cutout=False, cutout_ratio=0.2):
        self.df = df
        self.size = size
        self.use_augmentation = use_augmentation

        self.ex_image_full_db_ids = get_ex_image_full_db_supports()

        print(f'Size: {self.size}')

        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(Stats.mean, Stats.std)
        ])

        self.resizer = alb.Resize(self.size[0], self.size[1], p=1)

        if self.use_augmentation:
            print('Augmentation is enabled')
            self.augmentor = alb.Compose([
                alb.HorizontalFlip(p=0.5),
                alb.VerticalFlip(p=0.5),
                alb.RandomRotate90(p=0.5),
                alb.RandomBrightness(p=0.1),
                alb.RandomContrast(p=0.1)
            ])

            self.hard_augmentor = alb.Compose([
                alb.HorizontalFlip(p=0.5),
                alb.VerticalFlip(p=0.5),
                alb.RandomRotate90(p=0.5),
                alb.OneOf([
                    alb.IAAAdditiveGaussianNoise(p=0.5),
                    alb.GaussNoise(p=0.5)
                ], p=0.5),
                alb.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
                alb.OneOf([
                    alb.OpticalDistortion(p=1),
                    alb.GridDistortion(p=1),
                ], p=0.5),
                alb.RandomBrightness(p=0.5),
                alb.RandomContrast(p=0.5)
            ])

            if use_cutout:
                cutout_length = int(cutout_ratio * size[0])
                self.transformer.transforms.append(Cutout(n_holes=1, length=cutout_length))
                print(f'Append cutout to transformer (length: {cutout_length})')
        else:
            self.augmentor = None

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        i = int(i)
        row = self.df.iloc[i]

        labels = [int(label) for label in row['Target'].split(' ')]
        labels = np.eye(n_class, dtype=np.float32)[labels].sum(axis=0)

        image_id = row['Id']
        source = row['Source']

        if source == 'train':
            with h5py.File(str(images_h5_file), 'r') as image_db:
                im = image_db[f'train/{image_id}'].value
        elif source == 'ex':
            if image_id in self.ex_image_full_db_ids:
                with h5py.File(str(ex_full_images_h5_file), 'r') as ex_image_full_db:
                    im = ex_image_full_db[f'ex/{image_id}'].value
            else:
                with h5py.File(str(ex_images_h5_file), 'r') as ex_image_db:
                    im = ex_image_db[f'ex/{image_id}'].value

        if self.use_augmentation:
            if source == 'ex':
                aug = self.hard_augmentor(image=im)
            else:
                aug = self.augmentor(image=im)
            im = aug['image']

        im = self.resizer.apply(im)
        im = self.transformer(im)

        return im, labels


class HPAEnhancedDataset(Dataset):
    def __init__(self, df, size=(512, 512),
                 use_augmentation=True,
                 use_cutout=False, cutout_ratio=0.2,
                 image_db=None, ex_image_db=None, ex_image_full_db=None):
        self.df = df
        self.size = size
        self.use_augmentation = use_augmentation
        self.image_db = image_db
        self.ex_image_db = ex_image_db
        self.ex_image_full_db = ex_image_full_db

        self.ex_image_full_db_ids = get_ex_image_full_db_supports() if ex_image_db is not None else set()

        if self.ex_image_full_db_ids:
            print('Use ex_image_full_db')

        print(f'Size: {self.size}')

        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(Stats.mean, Stats.std)
        ])

        self.resizer = alb.Resize(self.size[0], self.size[1], p=1)

        if self.use_augmentation:
            print('Augmentation is enabled')
            self.augmentor = alb.Compose([
                alb.HorizontalFlip(p=0.5),
                alb.VerticalFlip(p=0.5),
                alb.RandomRotate90(p=0.5),
                alb.RandomBrightness(p=0.1),
                alb.RandomContrast(p=0.1)
            ])

            self.hard_augmentor = alb.Compose([
                alb.HorizontalFlip(p=0.5),
                alb.VerticalFlip(p=0.5),
                alb.RandomRotate90(p=0.5),
                alb.OneOf([
                    alb.IAAAdditiveGaussianNoise(p=0.5),
                    alb.GaussNoise(p=0.5)
                ], p=0.5),
                alb.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
                alb.OneOf([
                    alb.OpticalDistortion(p=1),
                    alb.GridDistortion(p=1),
                ], p=0.5),
                alb.RandomBrightness(p=0.5),
                alb.RandomContrast(p=0.5)
            ])

            if use_cutout:
                cutout_length = int(cutout_ratio * size[0])
                self.transformer.transforms.append(Cutout(n_holes=1, length=cutout_length))
                print(f'Append cutout to transformer (length: {cutout_length})')
        else:
            self.augmentor = None

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        i = int(i)
        row = self.df.iloc[i]

        labels = [int(label) for label in row['Target'].split(' ')]
        labels = np.eye(n_class, dtype=np.float32)[labels].sum(axis=0)

        image_id = row['Id']
        source = row['Source']
        if self.image_db:
            if source == 'train':
                im = self.image_db[f'train/{image_id}'].value
            elif source == 'ex':
                if image_id in self.ex_image_full_db_ids:
                    im = self.ex_image_full_db[f'ex/{image_id}'].value
                else:
                    im = self.ex_image_db[f'ex/{image_id}'].value
        else:
            if source == 'train':
                im = load_4ch_image_train(image_id)
            elif source == 'ex':
                im = load_4ch_image_ex(image_id)
            im = Image.fromarray(im)

        if self.use_augmentation:
            if source == 'ex':
                aug = self.hard_augmentor(image=im)
            else:
                aug = self.augmentor(image=im)
            im = aug['image']

        im = self.resizer.apply(im)
        im = self.transformer(im)

        return im, labels


class HPADataset(Dataset):
    def __init__(self, df, size=(256, 256),
                 use_transform=True, use_augmentation=True,
                 use_cutout=False, cutout_ratio=0.2,
                 image_db=None):
        self.df = df
        self.size = size
        self.use_transform = True
        self.use_augmentation = use_augmentation
        self.image_db = image_db

        print(f'Size: {self.size}')

        if self.use_augmentation:
            print('Augmentation is enabled')
            self.transformer = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Resize(self.size),
                transforms.ToTensor(),
                transforms.Normalize(Stats.mean, Stats.std)
            ])
            if use_cutout:
                cutout_length = int(cutout_ratio * size[0])
                self.transformer.transforms.append(Cutout(n_holes=1, length=cutout_length))
                print(f'Append cutout to transformer (length: {cutout_length})')
        else:
            self.transformer = transforms.Compose([
                transforms.Resize(self.size),
                transforms.ToTensor(),
                transforms.Normalize(Stats.mean, Stats.std)
            ])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        i = int(i)
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

        self.resizer = alb.Resize(self.size[0], self.size[1], p=1)
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(Stats.mean, Stats.std)
        ])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        image_id = self.df.iloc[i]['Id']
        if self.image_db:
            im = self.image_db[f'test/{image_id}'].value
        else:
            im = load_4ch_image_test(image_id)
            # TODO: to numpy

        im = self.resizer.apply(im)
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


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img