import torch
import gc
import numpy as np
from torchvision import transforms
from sklearn.metrics import f1_score
from data import *


def evaluate(model, loader, **kwargs):
    """F1 score evaluation
    """
    gc.collect()
    torch.cuda.empty_cache()

    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in progress_bar(loader):
            pred = predict(model, data, **kwargs)

            y_true.append(target.cpu().numpy())
            y_pred.append(pred)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    score = f1_score(y_true, y_pred, average='macro')

    return score


def predict(model, x, device=None,
            use_sigmoid=True, threshold=0.5,
            with_tta=False):
    """
    Returns
    -------
    nd.array
        {0, 1} array of prediction
    """
    if x.ndimension() == 3:
        x = x.expand(1, *x.shape)

    x = x.to(device)
    logit = model(x)

    pred = torch.sigmoid(logit) if use_sigmoid else logit
    
    if with_tta:
        logit = model(x.flip(3))
        pred += torch.sigmoid(logit) if use_sigmoid else logit
        pred = 0.5 * pred

    if threshold is None:
        return pred.cpu().numpy().astype(np.float32)

    pred = (pred > threshold).cpu().numpy()
    return pred


def compute_best_thresholds(model, loader, **kwargs):
    """
    """
    gc.collect()
    torch.cuda.empty_cache()

    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in progress_bar(loader):
            kwargs['threshold'] = None
            pred = predict(model, data, **kwargs)

            y_true.append(target.cpu().numpy())
            y_pred.append(pred)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    best_thresholds = []
    for class_index in range(n_class):
        best_score = 0.0
        best_threshold = 0.0
        for th in (0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                   0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9):
            score = f1_score(
                y_true[:, class_index],
                (y_pred[:, class_index] > th).astype(np.float32),
                average='macro')
            print(f'Class: {class_index}, th: {th}, score: {score}')
            if best_score < score:
                best_score = score
                best_threshold = th
        print(f'Class: {class_index}, Best threshold: {best_threshold}, Best score: {best_score}')
        print('-----')
        best_thresholds.append(best_threshold)

    best_f1 = f1_score(y_true, (y_pred > best_thresholds).astype(np.float32), average='macro')
    default_f1 = f1_score(y_true, (y_pred > 0.5).astype(np.float32), average='macro')

    print(f'Best F1: {best_f1}')
    print(f'Default F1: {default_f1}')

    return best_thresholds