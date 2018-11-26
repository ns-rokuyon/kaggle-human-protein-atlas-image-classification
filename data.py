import cv2
import tqdm
import torch
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split, KFold

try:
    from workspace import *
    on_colab = False
    print('workspace: local')
except ImportError:
    from workspace_colab import *
    on_colab = True
    print('workspace: colab')


progress_bar = tqdm.tqdm if on_colab else tqdm.tqdm_notebook


# Green filter for the target protein structure of interest
# Blue landmark filter for the nucleus
# Red landmark filter for microtubules
# Yellow landmark filter for the endoplasmatic reticulum
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


def gen_images_h5_file(df, group='train'):
    with h5py.File(str(images_h5_file), 'a') as fp:
        for i in progress_bar(range(df.shape[0])):
            image_id = df.iloc[i]['Id']
            if group == 'train':
                im = load_4ch_image_train(image_id)
            elif group == 'test':
                im = load_4ch_image_test(image_id)

            im = np.array(im)

            key = '{}/{}'.format(group, str(image_id))
            fp.create_dataset(key, data=im)


def open_images_h5_file():
    return h5py.File(str(images_h5_file), 'r')


def save_model(model, keyname):
    dict_filename = f'{keyname}_dict.model'
    torch.save(model.state_dict(), str(model_dir / dict_filename))
    
    filename = f'{keyname}.model'
    torch.save(model, str(model_dir / filename))


def get_train_df():
    df = pd.read_csv(train_csv)
    return df


def get_test_df():
    df = pd.read_csv(test_csv)
    return df


def get_train_val_df_fold(k):
    train_listfile = str(kfold_cv5_list_dir / f'train_cv{k}.csv')
    val_listfile = str(kfold_cv5_list_dir / f'val_cv{k}.csv')
    train_df = pd.read_csv(str(train_listfile))
    val_df = pd.read_csv(str(val_listfile))
    return train_df, val_df


def _kfold_dfs(k=5):
    df = get_train_df()
    kf = KFold(n_splits=k, shuffle=True, random_state=1234)
    for train_index, val_index in kf.split(df.index.values):
        fold_train_df = df.iloc[train_index]
        fold_val_df = df.iloc[val_index]
        yield fold_train_df, fold_val_df


def generate_kfold_cv5_list():
    for i, (train_df, val_df) in enumerate(_kfold_dfs(k=5)):
        train_listfile = str(kfold_cv5_list_dir / f'train_cv{i}.csv')
        val_listfile = str(kfold_cv5_list_dir / f'val_cv{i}.csv')
        train_df.to_csv(train_listfile, index=False)
        val_df.to_csv(val_listfile, index=False)
        print(f'Generated: {train_listfile}')
        print(f'Generated: {val_listfile}')


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