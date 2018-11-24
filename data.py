import cv2
import numpy as np
import pandas as pd
from PIL import Image

try:
    from workspace import *
    print('workspace: local')
except ImportError:
    from workspace_colab import *
    print('workspace: colab')


colors = ['red','green','blue','yellow']
name_label_dict = {
    0:  'Nucleoplasm',
    1:  'Nuclear membrane',
    2:  'Nucleoli',   
    3:  'Nucleoli fibrillar center',
    4:  'Nuclear speckles',
    5:  'Nuclear bodies',
    6:  'Endoplasmic reticulum',   
    7:  'Golgi apparatus',
    8:  'Peroxisomes',
    9:  'Endosomes',
    10:  'Lysosomes',
    11:  'Intermediate filaments',
    12:  'Actin filaments',
    13:  'Focal adhesion sites',   
    14:  'Microtubules',
    15:  'Microtubule ends',  
    16:  'Cytokinetic bridge',   
    17:  'Mitotic spindle',
    18:  'Microtubule organizing center',  
    19:  'Centrosome',
    20:  'Lipid droplets',
    21:  'Plasma membrane',   
    22:  'Cell junctions', 
    23:  'Mitochondria',
    24:  'Aggresome',
    25:  'Cytosol',
    26:  'Cytoplasmic bodies',   
    27:  'Rods & rings' 
}

n_class = len(name_label_dict)


class Stats:
    mean = np.array([0.08069, 0.05258, 0.05487, 0.08282])
    std = np.array([0.13704, 0.10145, 0.15313, 0.13814])


def get_train_df():
    df = pd.read_csv(train_csv)
    return df


def get_test_df():
    df = pd.read_csv(test_csv)
    return df


def load_4ch_image(base_path, id):
    """Get 4ch PIL.Image
    """
    bands = [Image.open(str(base_path / f'{id}_{color}.png'))
             for color in colors]
    image = Image.merge('RGBA', bands=bands)
    return image


def load_4ch_image_train(id):
    return load_4ch_image(train_image_dir, id)


def load_4ch_image_test(id):
    return load_4ch_image(test_image_dir, id)


def open_rgby(base_path, id):
    """function that reads RGBY image

    Return
    ------
    nd.array shape(512, 512, 4)
    """
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(str(base_path / f'{id}_{color}.png'), flags).astype(np.float32) / 255
           for color in colors]
    return np.stack(img, axis=-1)


def open_rgby_train(id):
    return open_rgby(train_image_dir, id)


def open_rgby_test(id):
    return open_rgby(test_image_dir, id)