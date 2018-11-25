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

    pred = (pred > threshold).cpu().numpy()
    return pred