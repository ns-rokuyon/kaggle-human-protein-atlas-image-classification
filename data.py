import cv2
import math
import tqdm
import torch
import h5py
import scipy.sparse as sp
import numpy as np
import pandas as pd
from collections import defaultdict
from PIL import Image
from sklearn.model_selection import train_test_split, KFold
from skmultilearn.model_selection import IterativeStratification

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


def gen_images_h5_file(df):
    if images_h5_file.exists():
        print(f'Found: {images_h5_file}')
        return

    with h5py.File(str(images_h5_file), 'a') as fp:
        for i in progress_bar(range(df.shape[0])):
            image_id = df.iloc[i]['Id']
            im = load_4ch_image_train(image_id)

            im = np.array(im)

            key = 'train/{}'.format(str(image_id))
            fp.create_dataset(key, data=im)


def gen_test_images_h5_file(df):
    if test_images_h5_file.exists():
        print(f'Found: {test_images_h5_file}')
        return

    with h5py.File(str(test_images_h5_file), 'a') as fp:
        for i in progress_bar(range(df.shape[0])):
            image_id = df.iloc[i]['Id']
            im = load_4ch_image_test(image_id)

            im = np.array(im)

            key = 'test/{}'.format(str(image_id))
            fp.create_dataset(key, data=im)


def open_images_h5_file():
    return h5py.File(str(images_h5_file), 'r')


def open_test_images_h5_file():
    return h5py.File(str(test_images_h5_file), 'r')


def save_model(model, keyname, optimizer=None, scheduler=None):
    dict_filename = f'{keyname}_dict.model'
    torch.save(model.state_dict(), str(model_dir / dict_filename))

    if optimizer:
        optim_filename = f'{keyname}_dict.optim'
        torch.save(optimizer.state_dict(), str(model_dir / optim_filename))

    if scheduler:
        scheduler_filename = f'{keyname}_dict.scheduler'
        torch.save(scheduler.state_dict(), str(model_dir / scheduler_filename))
    

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


def get_multilabel_stratified_train_val_df_fold(k):
    train_listfile = str(multilabel_stratified_kfold_cv3_list_dir / f'train_cv{k}.csv')
    val_listfile = str(multilabel_stratified_kfold_cv3_list_dir / f'val_cv{k}.csv')
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


def _multilabel_stratified_kfold_dfs():
    df = get_train_df()
    label_mat = multilabel_binary_representation(df, sparse=True)

    kf = IterativeStratification(random_state=1234)  # k=3
    for train_index, val_index in kf.split(df.index.values, label_mat):
        fold_train_df = df.iloc[train_index]
        fold_val_df = df.iloc[val_index]
        yield fold_train_df, fold_val_df


def generate_multilabel_stratified_kfold_cv3_list():
    for i, (train_df, val_df) in enumerate(_multilabel_stratified_kfold_dfs()):
        train_listfile = str(multilabel_stratified_kfold_cv3_list_dir / f'train_cv{i}.csv')
        val_listfile = str(multilabel_stratified_kfold_cv3_list_dir / f'val_cv{i}.csv')
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


def get_class_weights(df, max_value=100):
    pos = [0 for c in range(n_class)]
    neg = [0 for c in range(n_class)]
    for i in progress_bar(range(df.shape[0])):
        labels = [int(label) for label in df.iloc[i]['Target'].split(' ')]
        for c in range(n_class):
            if c in labels:
                pos[c] += 1
            else:
                neg[c] += 1
    weights = [n / p for p, n in zip(pos, neg)]
    if max_value is not None:
        weights = [w if w < max_value else max_value for w in weights]
    return weights


def write_log(message, keyname):
    print(message)
    with open(str(log_dir / f'{keyname}.txt'), 'a') as fp:
        print(message, file=fp)


def multilabel_binary_representation(df, sparse=True):
    mat = []
    for i in range(df.shape[0]):
        row = df.iloc[i]
        target = row['Target']
        ts = set(map(int, target.split(' ')))

        label_array = [1 if c in ts else 0 for c in range(n_class)]
        mat.append(label_array)

    mat = np.array(mat)
    if sparse:
        return sp.lil_matrix(mat)

    return mat


def create_sampling_log_weights(df, mu=0.5):
    label_set_count = defaultdict(int)
    for i in range(df.shape[0]):
        target = str(df.iloc[i]['Target'])
        label_set_count[target] += 1

    label_set_weights = {target: math.log(mu * df.shape[0] / count)
                         for target, count in label_set_count.items()}
    weights = [label_set_weights[str(df.iloc[i]['Target'])]
               for i in range(df.shape[0])]
    return weights, label_set_count


def create_weighted_random_sampler(df):
    weights, _ = create_sampling_log_weights(df)
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
    return sampler


def create_lr_scheduler(optimizer, patience=1, factor=0.5):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience, mode='max', factor=factor, verbose=True)
    return scheduler